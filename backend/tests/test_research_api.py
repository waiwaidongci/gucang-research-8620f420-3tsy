"""沿用现有研究入口 POST /runs 的端到端验证。"""
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.database import get_db
from src.constants.enums import RunStatus
from src.models import RuleRun
from src.routes.research import router
from tests.conftest import make_study


def _client(db):
    from src.controllers.dependencies import admins, current_user, writers
    from src.config.database import SessionLocal
    from src.models import User

    app = FastAPI()
    app.include_router(router)

    # 每个请求使用独立会话，与 worker 提交后的状态隔离，避免 identity map 过期数据。
    def _request_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _request_db
    user = db.query(User).first()
    app.dependency_overrides[current_user] = lambda: user
    app.dependency_overrides[writers] = lambda: user
    app.dependency_overrides[admins] = lambda: user
    return TestClient(app), user


def test_queue_endpoint_dispatches_worker_and_run_succeeds(db, artifact_dir, monkeypatch):
    _, batch, _, _ = make_study(db)
    client, _ = _client(db)

    # 不经过 Redis：让 delay 同步执行真实任务体。
    from src import worker

    monkeypatch.setattr(worker.analyze_run, "delay", worker.analyze_run.run)

    response = client.post(
        "/runs",
        json={
            "name": "入口研究",
            "batch_id": batch.id,
            "min_support": 0.05,
            "min_confidence": 0.35,
            "min_lift": 1.0,
            "max_size": 3,
        },
    )
    assert response.status_code == 201, response.text
    run_id = response.json()["id"]
    db.expire_all()
    run = db.get(RuleRun, run_id)
    assert run.status == RunStatus.SUCCEEDED
    assert run.result["rules"]
    from src.services.run_execution import artifact_path

    assert artifact_path(run_id).exists()


def test_queue_endpoint_marks_failed_when_broker_unavailable(db, monkeypatch):
    _, batch, _, _ = make_study(db)
    client, _ = _client(db)

    from src import worker

    def boom(run_id):
        raise ConnectionError("broker down")

    monkeypatch.setattr(worker.analyze_run, "delay", boom)
    response = client.post(
        "/runs",
        json={
            "name": "队列故障研究",
            "batch_id": batch.id,
            "min_support": 0.05,
            "min_confidence": 0.35,
            "min_lift": 1.0,
            "max_size": 3,
        },
    )
    assert response.status_code == 503
    db.expire_all()
    # make_study 预置 1 条 queued 研究，POST 新建的是另一条，验证新条目的失败状态。
    created = db.query(RuleRun).filter(RuleRun.name == "队列故障研究").one()
    assert created.status == RunStatus.FAILED
    assert "队列不可用" in created.error


def test_detail_and_list_reflect_succeeded_state(db, artifact_dir):
    _, _, run, _ = make_study(db)
    from src.services.run_execution import execute_run

    execute_run(run.id)
    client, _ = _client(db)

    detail = client.get(f"/runs/{run.id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == RunStatus.SUCCEEDED

    listed = client.get("/runs")
    assert listed.status_code == 200
    assert any(
        item["id"] == run.id and item["status"] == RunStatus.SUCCEEDED
        for item in listed.json()
    )
