"""测试环境：在导入应用代码前注入 SQLite/内存 broker 等环境变量。

生产使用 PostgreSQL + Redis；测试用文件型 SQLite（跨线程各自持有连接），
run_execution 的并发正确性由行级 CAS 与进程内锁共同保证，PG 咨询锁分支
在 test_run_execution.py 中用伪连接单独覆盖。
"""
import itertools
import os
import pathlib
import tempfile

import pytest

_TEST_ROOT = pathlib.Path(tempfile.mkdtemp(prefix="basket-tests-"))
(_TEST_ROOT / "artifacts").mkdir(parents=True, exist_ok=True)

os.environ["JWT_SECRET"] = "test-secret"
os.environ["SEED_DEMO"] = "false"
os.environ["REDIS_URL"] = "memory://"
os.environ["ARTIFACT_DIR"] = str(_TEST_ROOT / "artifacts")
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_ROOT / 'test.db'}?timeout=30"

from src.config.database import Base, SessionLocal, engine  # noqa: E402
from src.models import (  # noqa: E402
    Product,
    Receipt,
    ReceiptBatch,
    RuleRun,
    User,
)
from src.services import run_execution  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    Base.metadata.create_all(engine)
    yield


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def clean_db():
    """每个用例后清空所有表，避免自然键与自增 id 互相干扰。"""
    yield
    with engine.begin() as conn:
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(table.delete())


@pytest.fixture
def artifact_dir(monkeypatch, tmp_path):
    """把结果文件目录重定向到用例专属临时目录。"""
    monkeypatch.setattr(run_execution, "ARTIFACT_DIR", str(tmp_path))
    return tmp_path


_study_seq = itertools.count(1)
_sku_seq = itertools.count(1)


def make_study(db, *, receipts=None, status="queued", name="秋季早餐组合研究"):
    """构造 1 个用户、2 个商品、1 个批次、若干篮子与 1 条研究记录。"""
    seq = next(_study_seq)
    sku_seq = next(_sku_seq)
    user = User(
        email=f"analyst-{seq}@gucang.demo",
        name="研究员",
        role="analyst",
        password_hash="salt$digest",
    )
    db.add(user)
    db.flush()
    sku_oat, sku_bread = f"OAT{sku_seq}", f"BREAD{sku_seq}"
    db.add_all(
        [
            Product(sku=sku_oat, name="燕麦拿铁", category="饮品", price=15.0, cost=7.0),
            Product(sku=sku_bread, name="全麦餐包", category="烘焙", price=9.0, cost=3.5),
        ]
    )
    batch = ReceiptBatch(name=f"测试批次-{seq}", filename="baskets.jsonl", checksum="deadbeef")
    db.add(batch)
    db.flush()
    if receipts is None:
        # OAT 全量出现、BREAD 八成出现，保证能挖出 OAT→BREAD 规则。
        receipts = [[sku_oat, sku_bread]] * 8 + [[sku_oat]] * 2
    db.add_all(
        [
            Receipt(batch_id=batch.id, reference=f"R{i:03d}", items=items)
            for i, items in enumerate(receipts)
        ]
    )
    run = RuleRun(
        name=name,
        batch_id=batch.id,
        status=status,
        min_support=0.05,
        min_confidence=0.35,
        min_lift=1.0,
        max_size=3,
        creator_id=user.id,
    )
    db.add(run)
    db.commit()
    return user, batch, run, (sku_oat, sku_bread)
