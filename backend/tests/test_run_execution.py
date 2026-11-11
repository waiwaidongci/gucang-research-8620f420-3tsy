"""交错执行、失败重试与终态保护的复现验证。

覆盖用户报告的原始缺陷：同一研究被重复派发时，较晚失败的尝试会把已成功
研究标成 failed（结果文件却还在）。修复后状态、规则结果与结果文件必须
始终对应同一次成功计算。
"""
import json
import threading

import pytest
from sqlalchemy import select

from src.constants.enums import RunStatus
from src.models import AuditLog, BundleProposal, RuleRun
from src.services import run_execution
from src.services.run_execution import (
    OUTCOME_LOCKED,
    OUTCOME_SKIPPED,
    OUTCOME_SUCCEEDED,
    RunLock,
    RetryableRunError,
    artifact_path,
    execute_run,
)
from src.schemas.proposal import ProposalInput
from src.services.proposals import create_proposal
from tests.conftest import make_study


def _fresh_run(run_id):
    from src.config.database import SessionLocal

    with SessionLocal() as db:
        return db.get(RuleRun, run_id)


def _artifact_json(run_id, artifact_dir):
    return json.loads(artifact_path(run_id).read_text(encoding="utf-8"))


def test_successful_run_persists_status_result_and_artifact(db, artifact_dir):
    _, _, run, _ = make_study(db)
    outcome = execute_run(run.id)

    assert outcome == OUTCOME_SUCCEEDED
    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert stored.error is None
    assert stored.result["rules"], "应挖出关联规则"
    on_disk = _artifact_json(run.id, artifact_dir)
    assert on_disk == stored.result
    assert not list(artifact_dir.glob(".run-*.tmp")), "临时文件必须清理干净"


def test_stale_duplicate_after_success_is_skipped_and_changes_nothing(db, artifact_dir):
    """复现原缺陷的交错时序：A 成功后，迟到的 B（模拟旧消息重投）再执行。"""
    _, _, run, _ = make_study(db)
    assert execute_run(run.id) == OUTCOME_SUCCEEDED
    first = _fresh_run(run.id)
    snapshot = dict(status=first.status, result=first.result)
    artifact_snapshot = artifact_path(run.id).read_bytes()

    # 迟到/重复投递：直接再次执行，必须跳过且不产生任何写入。
    outcome = execute_run(run.id)
    assert outcome == OUTCOME_SKIPPED

    second = _fresh_run(run.id)
    assert second.status == snapshot["status"] == RunStatus.SUCCEEDED
    assert second.result == snapshot["result"]
    assert artifact_path(run.id).read_bytes() == artifact_snapshot


def test_late_failing_attempt_cannot_overwrite_succeeded_run(db, artifact_dir, monkeypatch):
    """原缺陷精确复现：成功后又来一次尝试，且该尝试在计算阶段抛错。"""
    _, _, run, _ = make_study(db)
    assert execute_run(run.id) == OUTCOME_SUCCEEDED
    good = _fresh_run(run.id)
    good_artifact = artifact_path(run.id).read_bytes()

    def boom(*args, **kwargs):
        raise RuntimeError("模拟迟到尝试的基础设施故障")

    monkeypatch.setattr(run_execution, "mine_rules", boom)
    # 成功终态在任何计算发生之前就被短路，异常根本不会触发。
    assert execute_run(run.id) == OUTCOME_SKIPPED

    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert stored.error is None
    assert stored.result == good.result
    assert artifact_path(run.id).read_bytes() == good_artifact


def test_interleaved_duplicate_dispatch_only_one_computation_wins(db, artifact_dir, monkeypatch):
    """两个执行者几乎同时拿到同一 queued run：持锁者计算，另一个立即返回 locked。"""
    _, _, run, _ = make_study(db)
    outcomes = []
    entered = threading.Event()
    proceed = threading.Event()

    real_mine = run_execution.mine_rules

    def slow_mine(*args, **kwargs):
        entered.set()
        proceed.wait(timeout=5)
        return real_mine(*args, **kwargs)

    monkeypatch.setattr(run_execution, "mine_rules", slow_mine)

    def first():
        outcomes.append(execute_run(run.id))

    t = threading.Thread(target=first)
    t.start()
    assert entered.wait(timeout=5), "首个执行者应已进入计算"
    # 计算进行中，重复派发到达。
    outcomes.append(execute_run(run.id))
    proceed.set()
    t.join(timeout=10)

    assert sorted(outcomes) == sorted([OUTCOME_SUCCEEDED, OUTCOME_LOCKED])
    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert _artifact_json(run.id, artifact_dir) == stored.result


def test_cas_and_lock_together_prevent_duplicate_computation(db, artifact_dir, monkeypatch):
    """锁 + CAS 双保险：同一 run 只有一个执行者计算，成功审计仅一条。"""
    _, _, run, _ = make_study(db)
    entered = threading.Event()
    proceed = threading.Event()
    real_mine = run_execution.mine_rules
    compute_calls = {"n": 0}

    def slow_mine(*args, **kwargs):
        compute_calls["n"] += 1
        entered.set()
        proceed.wait(timeout=5)
        return real_mine(*args, **kwargs)

    monkeypatch.setattr(run_execution, "mine_rules", slow_mine)

    results = []

    def worker():
        results.append(execute_run(run.id))

    t = threading.Thread(target=worker)
    t.start()
    assert entered.wait(timeout=5)
    # 真实进程锁互斥：重复投递拿不到锁，立即返回 locked。
    results.append(execute_run(run.id))
    proceed.set()
    t.join(timeout=10)

    assert sorted(results) == sorted([OUTCOME_SUCCEEDED, OUTCOME_LOCKED])
    assert compute_calls["n"] == 1
    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert _artifact_json(run.id, artifact_dir) == stored.result
    with db:
        actions = [a.action for a in db.scalars(select(AuditLog))]
    assert actions.count("run.succeeded") == 1
    assert "run.failed" not in actions


def test_finalize_cas_rejects_stale_writer_overwrite(db, artifact_dir):
    """终态 CAS 底线：UPDATE ... WHERE status='running' 在已成功时 rowcount=0。

    模拟两个执行者先后越过领取（如极端的锁失效），先到者已提交成功，
    后到者的最终写入必须被数据库拒绝，无法覆盖已成功的规则结果。
    """
    from sqlalchemy import update as sa_update

    _, _, run, _ = make_study(db, status=RunStatus.RUNNING)
    # 先到者已把 run 提交为成功（终态）。
    db.execute(
        sa_update(RuleRun)
        .where(RuleRun.id == run.id)
        .values(status=RunStatus.SUCCEEDED, result={"rules": []}, error=None)
    )
    db.commit()

    # 后到者持旧结果执行 finalize CAS：where 条件已不匹配，rowcount 必须为 0。
    rowcount = db.execute(
        sa_update(RuleRun)
        .where(RuleRun.id == run.id, RuleRun.status == RunStatus.RUNNING)
        .values(status=RunStatus.SUCCEEDED, result={"rules": [{"stale": True}]}, error=None)
    ).rowcount
    db.commit()
    assert rowcount == 0
    stored = _fresh_run(run.id)
    assert stored.result == {"rules": []}

    # execute_run 看到成功终态直接跳过，不会重算或改写。
    assert execute_run(run.id) == OUTCOME_SKIPPED
    assert _fresh_run(run.id).result == {"rules": []}


def test_stale_running_after_hard_kill_is_taken_over_and_completes(db, artifact_dir):
    """worker 被 SIGKILL 留下 running 状态：下一次投递安全接管并成功。"""
    _, _, run, _ = make_study(db, status=RunStatus.RUNNING)
    # 模拟上次执行已写出半截临时文件。
    (artifact_dir / f".run-{run.id}.deadbeef.tmp").write_text("{半截", encoding="utf-8")

    assert execute_run(run.id) == OUTCOME_SUCCEEDED

    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert stored.error is None
    assert _artifact_json(run.id, artifact_dir) == stored.result
    assert not list(artifact_dir.glob(".run-*.tmp"))


def test_retry_after_mid_computation_failure_completes_successfully(db, artifact_dir, monkeypatch):
    """计算中途失败 → run 落 failed、无残留文件 → 重试成功并对齐三者。"""
    _, _, run, _ = make_study(db)
    calls = {"n": 0}
    real_mine = run_execution.mine_rules

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("计算进程被中断")
        return real_mine(*args, **kwargs)

    monkeypatch.setattr(run_execution, "mine_rules", flaky)

    with pytest.raises(RetryableRunError):
        execute_run(run.id)
    failed = _fresh_run(run.id)
    assert failed.status == RunStatus.FAILED
    assert "计算进程被中断" in failed.error
    assert not artifact_path(run.id).exists(), "失败后不得留下结果文件"
    assert not list(artifact_dir.glob(".run-*.tmp"))

    monkeypatch.undo()
    outcome = execute_run(run.id)
    assert outcome == OUTCOME_SUCCEEDED
    done = _fresh_run(run.id)
    assert done.status == RunStatus.SUCCEEDED
    assert done.error is None
    assert _artifact_json(run.id, artifact_dir) == done.result


def test_retry_after_artifact_write_failure_leaves_no_partial_file(db, artifact_dir, monkeypatch):
    """写结果文件中途失败：不得有半截文件，重试后完整落盘。"""
    _, _, run, _ = make_study(db)

    def broken_write(run_id, result):
        path = artifact_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".run-{run_id}.broken.tmp"
        tmp.write_text("{半截内容", encoding="utf-8")
        raise OSError("磁盘暂时不可写")

    monkeypatch.setattr(run_execution, "write_atomic_artifact", broken_write)
    with pytest.raises(RetryableRunError):
        execute_run(run.id)

    assert not artifact_path(run.id).exists()
    assert _fresh_run(run.id).status == RunStatus.FAILED

    monkeypatch.undo()
    assert execute_run(run.id) == OUTCOME_SUCCEEDED
    stored = _fresh_run(run.id)
    assert _artifact_json(run.id, artifact_dir) == stored.result


def test_permanent_value_error_marks_failed_and_is_retryable_to_completion(db, artifact_dir):
    """空批次/非法阈值等永久失败先落 failed；数据补齐后同一 run 可重试成功。"""
    _, batch, run, (sku_oat, sku_bread) = make_study(db, receipts=[])
    with pytest.raises(ValueError):
        execute_run(run.id)
    failed = _fresh_run(run.id)
    assert failed.status == RunStatus.FAILED
    assert "批次没有收据" in failed.error
    assert not artifact_path(run.id).exists()

    # 补齐收据后沿用同一条研究记录重试。
    from src.models import Receipt

    db.add_all(
        [
            Receipt(batch_id=batch.id, reference="R000", items=[sku_oat, sku_bread]),
            Receipt(batch_id=batch.id, reference="R001", items=[sku_oat, sku_bread]),
        ]
    )
    db.commit()
    assert execute_run(run.id) == OUTCOME_SUCCEEDED
    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert _artifact_json(run.id, artifact_dir) == stored.result


def test_proposal_rule_snapshot_survives_retries_and_later_runs(db, artifact_dir):
    """已有组合提案的规则依据在失败/重试/重复执行后保持原样。"""
    user, _, run, _ = make_study(db)
    assert execute_run(run.id) == OUTCOME_SUCCEEDED
    # worker 在独立会话中提交，过期本会话缓存以读取最新成功状态。
    db.expire_all()
    proposal = create_proposal(
        db,
        ProposalInput(run_id=run.id, rule_index=0, name="早餐温暖搭档", note="试点"),
        user,
    )
    snapshot = dict(proposal["rule"])

    # 迟到失败尝试的最后一步（标记失败）也不得覆盖成功终态。
    run_execution._mark_failed(run.id, "迟到尝试的失败信息")
    # 重复执行直接短路跳过。
    assert execute_run(run.id) == OUTCOME_SKIPPED

    refreshed = db.get(BundleProposal, proposal["id"])
    assert refreshed.rule == snapshot
    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert stored.error is None
    assert stored.result["rules"][0] == snapshot
    assert _artifact_json(run.id, artifact_dir) == stored.result


def test_storm_of_concurrent_duplicates_yields_single_consistent_success(db, artifact_dir):
    """N 个执行者同时抢同一 queued run：恰 1 次成功，终态/文件/审计完全一致。"""
    _, _, run, _ = make_study(db)
    n = 8
    outcomes = []
    barrier = threading.Barrier(n)

    def contender():
        barrier.wait(timeout=10)  # 尽量让所有线程同时发起，最大化交错。
        outcomes.append(execute_run(run.id))

    threads = [threading.Thread(target=contender) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
        assert not t.is_alive()

    assert outcomes.count(OUTCOME_SUCCEEDED) == 1
    assert all(o in (OUTCOME_LOCKED, OUTCOME_SKIPPED) for o in outcomes if o != OUTCOME_SUCCEEDED)

    stored = _fresh_run(run.id)
    assert stored.status == RunStatus.SUCCEEDED
    assert stored.error is None
    on_disk = _artifact_json(run.id, artifact_dir)
    assert on_disk == stored.result
    assert not list(artifact_dir.glob(".run-*.tmp"))
    with db:
        succeeded_audits = db.scalars(
            select(AuditLog).where(AuditLog.action == "run.succeeded")
        ).all()
    assert len(succeeded_audits) == 1
    assert all(a.detail == {"run_id": run.id, "rules": len(stored.result["rules"])} for a in succeeded_audits)


def test_unknown_run_is_skipped(db):
    assert execute_run(999999) == OUTCOME_SKIPPED


def test_failure_clears_stale_artifact_from_previous_attempt(db, artifact_dir):
    """失败路径必须清掉本 run 的遗留结果文件，避免文件与 failed 状态并存。"""
    _, _, run, _ = make_study(db, receipts=[])
    artifact_path(run.id).write_text('{"stale": true}', encoding="utf-8")

    with pytest.raises(ValueError):
        execute_run(run.id)

    assert not artifact_path(run.id).exists()
    assert _fresh_run(run.id).status == RunStatus.FAILED


def test_run_lock_serializes_same_run_but_allows_different_runs(db):
    """同 run 锁互斥；不同 run 的锁互不阻塞。"""
    _, _, run_a, _ = make_study(db)
    _, _, run_b, _ = make_study(db, name="另一项研究")
    lock_a = RunLock(run_a.id)
    lock_a2 = RunLock(run_a.id)
    lock_b = RunLock(run_b.id)
    assert lock_a.acquire() is True
    try:
        assert lock_a2.acquire() is False
        assert lock_b.acquire() is True
        lock_b.release()
    finally:
        lock_a.release()
    # 释放后可重新获取。
    assert lock_a2.acquire() is True
    lock_a2.release()


def test_postgres_advisory_lock_branch_uses_pg_functions(monkeypatch):
    """PG 方言分支：acquire/release 走 pg_try_advisory_lock / pg_advisory_unlock。"""
    calls = []

    class FakeConn:
        def execute(self, stmt, params):
            sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
            calls.append((sql, params))

            class Result:
                def scalar(self_inner):
                    return "pg_try_advisory_lock" in sql

            return Result()

        def close(self):
            calls.append(("close", None))

    fake_engine = type(
        "FakeEngine",
        (),
        {"dialect": type("Dialect", (), {"name": "postgresql"})(), "raw_connection": lambda self: FakeConn()},
    )()
    monkeypatch.setattr(run_execution, "engine", fake_engine)

    lock = RunLock(7)
    assert lock.acquire() is True
    lock.release()
    assert any("pg_try_advisory_lock" in sql for sql, _ in calls)
    assert any("pg_advisory_unlock" in sql for sql, _ in calls)
    assert calls[-1] == ("close", None)

    # 抢锁失败时立即归还连接且不持有锁。
    class DeniedConn(FakeConn):
        def execute(self, stmt, params):
            class Result:
                def scalar(self_inner):
                    return False

            return Result()

    denied_engine = type(
        "FakeEngine",
        (),
        {"dialect": type("Dialect", (), {"name": "postgresql"})(), "raw_connection": lambda self: DeniedConn()},
    )()
    monkeypatch.setattr(run_execution, "engine", denied_engine)
    assert RunLock(7).acquire() is False
