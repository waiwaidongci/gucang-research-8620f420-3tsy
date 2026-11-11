"""规则研究的实际计算单元。

并发与重试不变量：
1. 同一 run 在任一时刻只允许一个执行者：先尝试按 run id 加跨进程锁，
   再用数据库行上的 CAS 领取，双保险。
2. 已成功的 run 是终态，任何迟到/重复的尝试都直接跳过，不会改写状态、
   规则结果或结果文件。
3. 结果文件先写临时文件、fsync 后再原子替换，且仅在成功提交路径上落地；
   失败路径仅在确认本 run 未成功时清理残留文件，保证“状态/结果/文件”
   属于同一次成功计算。
4. 计算或写文件中途失败后可再次调用本模块重试完成；被 SIGKILL 中断而
   遗留在 running 的研究，因持锁者唯一，可被下一次投递安全接管。
5. 提案中的规则快照由 proposals 服务复制保存，本模块从不改动
   bundle_proposals，规则依据保持原样。
"""
import json
import os
import threading
import uuid
from pathlib import Path

from sqlalchemy import select, update, text
from sqlalchemy.exc import SQLAlchemyError

from src.config.database import SessionLocal, engine
from src.config.settings import ARTIFACT_DIR
from src.constants.enums import RunStatus
from src.models import Product, Receipt, RuleRun
from src.services.apriori import mine_rules
from src.services.audit import record

# 与一次成功计算对齐的结果状态。
OUTCOME_SKIPPED = "skipped"
OUTCOME_SUCCEEDED = "succeeded"
OUTCOME_LOCKED = "locked"

_LOCK_NAMESPACE = 72849105  # 固定命名空间，避免与其他 pg_advisory_lock 用途冲突


class RetryableRunError(Exception):
    """基础设施类、重试后可能恢复的错误（区别于参数/数据类的永久失败）。"""


class RunLock:
    """按 run id 串行化执行的跨进程锁。

    PostgreSQL 使用会话级咨询锁，工作进程崩溃或断连时由数据库自动释放，
    不会把 run 永久卡在 running；其他方言（如测试用 SQLite）回退到进程内
    线程锁，数据库层面的 CAS 领取仍提供第二道防线。
    """

    _thread_locks: dict[int, threading.Lock] = {}
    _registry_lock = threading.Lock()

    def __init__(self, run_id: int):
        self.run_id = run_id
        self._pg_conn = None
        self._thread_lock = None

    def acquire(self) -> bool:
        if engine.dialect.name == "postgresql":
            conn = engine.raw_connection()
            try:
                acquired = conn.execute(
                    text("SELECT pg_try_advisory_lock(:ns, :rid)"),
                    {"ns": _LOCK_NAMESPACE, "rid": self.run_id},
                ).scalar()
            except Exception:
                conn.close()
                raise
            if not acquired:
                conn.close()
                return False
            self._pg_conn = conn
            return True

        with self._registry_lock:
            lock = self._thread_locks.setdefault(self.run_id, threading.Lock())
        if not lock.acquire(blocking=False):
            return False
        self._thread_lock = lock
        return True

    def release(self) -> None:
        if self._pg_conn is not None:
            try:
                self._pg_conn.execute(
                    text("SELECT pg_advisory_unlock(:ns, :rid)"),
                    {"ns": _LOCK_NAMESPACE, "rid": self.run_id},
                )
            finally:
                self._pg_conn.close()
                self._pg_conn = None
        if self._thread_lock is not None:
            self._thread_lock.release()
            self._thread_lock = None

    def __enter__(self):
        if not self.acquire():
            return None
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def artifact_path(run_id: int) -> Path:
    return Path(ARTIFACT_DIR) / f"run-{run_id}.json"


def _temp_path(run_id: int) -> Path:
    return Path(ARTIFACT_DIR) / f".run-{run_id}.{uuid.uuid4().hex}.tmp"


def write_atomic_artifact(run_id: int, result: dict) -> Path:
    """临时文件 + fsync + 原子替换，避免半截文件被当作有效结果。"""
    directory = Path(ARTIFACT_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    final = artifact_path(run_id)
    tmp = _temp_path(run_id)
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
        # 替换后回读校验，落盘内容必须与本次计算结果逐字节一致。
        if final.read_text(encoding="utf-8") != json.dumps(
            result, ensure_ascii=False
        ):
            raise RetryableRunError("结果文件回读校验失败")
        return final
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _remove_artifact(run_id: int) -> None:
    artifact_path(run_id).unlink(missing_ok=True)
    for leftover in Path(ARTIFACT_DIR).glob(f".run-{run_id}.*.tmp"):
        leftover.unlink(missing_ok=True)


def _safe_cleanup_on_failure(run_id: int) -> None:
    """失败清理：若 run 已被其他路径置为成功，则保留其结果文件不动。"""
    try:
        with SessionLocal() as db:
            run = db.get(RuleRun, run_id)
            if run is not None and run.status == RunStatus.SUCCEEDED:
                return
    except SQLAlchemyError:
        # 连库失败时宁可不删，避免误删已成功文件；下一次重试会整体覆盖。
        return
    _remove_artifact(run_id)


def _load_inputs(db, run: RuleRun):
    baskets = list(
        db.scalars(
            select(Receipt.items)
            .where(Receipt.batch_id == run.batch_id)
            .order_by(Receipt.id)
        )
    )
    products = {
        p.sku: {"price": p.price, "cost": p.cost}
        for p in db.scalars(select(Product))
    }
    return baskets, products


def _mark_failed(run_id: int, message: str) -> None:
    """把 run 标记为失败；已成功的终态绝不被失败覆盖。"""
    try:
        with SessionLocal() as db:
            run = db.get(RuleRun, run_id)
            if run is None or run.status == RunStatus.SUCCEEDED:
                return
            run.status = RunStatus.FAILED
            run.error = message[:1000]
            db.commit()
            record(db, "worker", "run.failed", {"run_id": run_id, "error": run.error})
    except SQLAlchemyError:
        # 连库本身失败时无法落状态，交给 Celery 重试/消息重投，不吞掉现场。
        pass


def execute_run(run_id: int) -> str:
    """执行一次研究计算，返回 OUTCOME_*；可安全重复调用。

    - 已成功：返回 skipped，不做任何写入。
    - 其他执行者持锁：返回 locked，不做任何状态变更。
    - 成功：状态、result、结果文件与审计在同一成功路径落地。
    - 失败：状态置 failed（成功终态除外）、清理本 run 残留文件后向上抛出，
      永久失败（ValueError）原样抛出，基础设施失败包装为 RetryableRunError。
    """
    lock = RunLock(run_id)
    if not lock.acquire():
        return OUTCOME_LOCKED

    try:
        with SessionLocal() as db:
            run = db.get(RuleRun, run_id)
            if run is None:
                return OUTCOME_SKIPPED
            if run.status == RunStatus.SUCCEEDED:
                return OUTCOME_SKIPPED

            # CAS 领取：queued/failed/running 可转为 running。
            # 持锁者唯一，因此看到 running 只可能是上次执行者被硬杀（如
            # Celery time_limit/SIGKILL）留下的陈旧状态，安全接管。
            # 若另一执行者已抢先成功（status=succeeded），rowcount=0，本次退出。
            claimed = db.execute(
                update(RuleRun)
                .where(
                    RuleRun.id == run_id,
                    RuleRun.status.in_(
                        [RunStatus.QUEUED, RunStatus.FAILED, RunStatus.RUNNING]
                    ),
                )
                .values(status=RunStatus.RUNNING, error=None)
            ).rowcount
            db.commit()
            if not claimed:
                # 已被其他执行者完成（succeeded 不在领取范围内）：重新读取确认。
                db.expire_all()
                current = db.get(RuleRun, run_id)
                if current is not None and current.status == RunStatus.SUCCEEDED:
                    return OUTCOME_SKIPPED
                return OUTCOME_LOCKED

            # 接管后清掉上次执行（可能被硬杀）遗留的临时文件；
            # 已存在的正式结果文件由本次成功计算原子覆盖，失败则由清理逻辑移除。
            for leftover in Path(ARTIFACT_DIR).glob(f".run-{run_id}.*.tmp"):
                leftover.unlink(missing_ok=True)

            try:
                baskets, products = _load_inputs(db, run)
                result = mine_rules(
                    baskets,
                    products,
                    run.min_support,
                    run.min_confidence,
                    run.min_lift,
                    run.max_size,
                )
                # 先原子落盘，再提交数据库；提交失败的下一次重试会整体重算并覆盖。
                write_atomic_artifact(run_id, result)

                # 终态写入同样带 CAS：只允许本执行者持有的 running -> succeeded，
                # 任何越权/过期的状态变更都无法把已成功研究改写成别的状态。
                finalized = db.execute(
                    update(RuleRun)
                    .where(
                        RuleRun.id == run_id,
                        RuleRun.status == RunStatus.RUNNING,
                    )
                    .values(status=RunStatus.SUCCEEDED, result=result, error=None)
                ).rowcount
                db.commit()
                if not finalized:
                    db.rollback()
                    db.expire_all()
                    current = db.get(RuleRun, run_id)
                    if current is not None and current.status == RunStatus.SUCCEEDED:
                        return OUTCOME_SKIPPED
                    raise RetryableRunError("研究状态在提交时发生冲突")

                record(
                    db,
                    "worker",
                    "run.succeeded",
                    {"run_id": run_id, "rules": len(result["rules"])},
                )
                return OUTCOME_SUCCEEDED
            except ValueError:
                # 参数/数据问题（空批次、未知 SKU、非法阈值）：重试结果相同，永久失败。
                db.rollback()
                raise
            except Exception:
                db.rollback()
                raise
    except RetryableRunError as exc:
        _safe_cleanup_on_failure(run_id)
        _mark_failed(run_id, str(exc) or exc.__class__.__name__)
        raise
    except ValueError as exc:
        _safe_cleanup_on_failure(run_id)
        _mark_failed(run_id, str(exc) or exc.__class__.__name__)
        raise
    except Exception as exc:  # 数据库/文件系统等基础设施故障：包装为可重试错误。
        _safe_cleanup_on_failure(run_id)
        _mark_failed(run_id, str(exc) or exc.__class__.__name__)
        raise RetryableRunError(str(exc) or exc.__class__.__name__) from exc
    finally:
        lock.release()
