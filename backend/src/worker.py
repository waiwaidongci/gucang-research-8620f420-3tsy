from celery import Celery

from src.config.settings import REDIS_URL
from src.services.run_execution import RetryableRunError, execute_run

app = Celery("basket_forge", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# 基础设施类故障有限重试 3 次（共 4 次尝试），指数退避；
# 参数/数据类永久失败（ValueError）不重试。execute_run 本身幂等，
# 重复投递或重试都不会改写已成功的研究状态、规则结果与结果文件。
MAX_RETRIES = 3


@app.task(
    name="basket.analyze",
    bind=True,
    soft_time_limit=120,
    time_limit=150,
    max_retries=MAX_RETRIES,
)
def analyze_run(self, id):
    try:
        return execute_run(id)
    except RetryableRunError as exc:
        if self.request.retries >= MAX_RETRIES:
            # 重试耗尽：execute_run 已将 run 置为 failed 并清理残留文件。
            return "failed"
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 5)
