"""Celery 包装层：有界重试、永久失败不重试、入口任务名保持不变。"""
import pytest

from src.constants.enums import RunStatus
from src.models import RuleRun
from src.worker import MAX_RETRIES, analyze_run
from src.services.run_execution import OUTCOME_SUCCEEDED, RetryableRunError
from tests.conftest import make_study


def test_task_contract_unchanged():
    # 沿用现有研究入口（runs.create_run 调用 analyze_run.delay）。
    assert analyze_run.name == "basket.analyze"
    assert analyze_run.max_retries == MAX_RETRIES == 3
    assert analyze_run.acks_late is True
    assert analyze_run.time_limit == 150


def test_analyze_run_happy_path(db, artifact_dir):
    _, _, run, _ = make_study(db)
    assert analyze_run.run(run.id) == OUTCOME_SUCCEEDED
    from src.config.database import SessionLocal

    with SessionLocal() as session:
        assert session.get(RuleRun, run.id).status == RunStatus.SUCCEEDED


class _SimulatedRedelivery(Exception):
    """模拟 Celery retry 抛出异常中断任务，随后由 broker 重新投递。"""

    def __init__(self, exc, countdown):
        self.exc = exc
        self.countdown = countdown


def _invoke_with_redelivery(monkeypatch, run_id, initial_retries, execute_impl):
    """模拟 Celery 的“retry 抛异常 → 计数 +1 → 重新投递执行”循环。"""
    from src import worker as worker_module

    def fake_retry(exc=None, countdown=None, **kwargs):
        # 真实 Celery 的 retry 总是抛出 Retry 异常以中断当前任务执行。
        raise _SimulatedRedelivery(exc, countdown)

    monkeypatch.setattr(worker_module, "execute_run", execute_impl)
    monkeypatch.setattr(analyze_run, "retry", fake_retry)

    retry_events = []
    analyze_run.push_request(retries=initial_retries)
    try:
        while True:
            try:
                result = analyze_run.__wrapped__(run_id)
                return result, retry_events
            except _SimulatedRedelivery as redelivery:
                retry_events.append((str(redelivery.exc), redelivery.countdown))
                # 重新投递时 Celery 把 retries 加一，然后再次执行任务体。
                analyze_run.request.retries += 1
    finally:
        analyze_run.pop_request()


def test_retryable_error_triggers_bounded_celery_retry(db, monkeypatch):
    _, _, run, _ = make_study(db)
    calls = {"n": 0}
    from src.services import run_execution

    real = run_execution.execute_run

    def flaky(run_id):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RetryableRunError("redis 抖动")
        return real(run_id)

    result, events = _invoke_with_redelivery(monkeypatch, run.id, 0, flaky)
    assert result == OUTCOME_SUCCEEDED
    assert len(events) == 2
    assert calls["n"] == 3
    # 指数退避：2^0*5=5s、2^1*5=10s。
    assert [countdown for _, countdown in events] == [5, 10]


def test_retries_exhausted_returns_failed_without_reraising(db, monkeypatch):
    """已达重试用尽（retries==max_retries）：任务直接返回 failed，不再 retry。"""
    _, _, run, _ = make_study(db)
    from src.services import run_execution

    def always_boom(*args, **kwargs):
        raise RuntimeError("文件系统持续故障")

    monkeypatch.setattr(run_execution, "mine_rules", always_boom)

    result, events = _invoke_with_redelivery(
        monkeypatch, run.id, MAX_RETRIES, run_execution.execute_run
    )
    assert result == "failed"
    assert events == []  # 已达上限，不再 retry
    from src.config.database import SessionLocal

    with SessionLocal() as session:
        stored = session.get(RuleRun, run.id)
        assert stored.status == RunStatus.FAILED
        assert "文件系统持续故障" in stored.error


def test_permanent_value_error_is_not_retried(db, monkeypatch):
    _, _, run, _ = make_study(db, receipts=[])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("永久失败不应触发 Celery retry")

    monkeypatch.setattr(analyze_run, "retry", fail_if_called)
    analyze_run.push_request(retries=0)
    try:
        with pytest.raises(ValueError):
            analyze_run.__wrapped__(run.id)
    finally:
        analyze_run.pop_request()
    assert analyze_run.request.retries == 0

    from src.config.database import SessionLocal

    with SessionLocal() as session:
        assert session.get(RuleRun, run.id).status == RunStatus.FAILED
