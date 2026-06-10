"""Prometheus 文本格式指标:聚合 actor health + HTTP 请求计数 + 请求时延直方图 + GPU 显存。"""
import asyncio

from services.middleware import (
    HISTOGRAM_BUCKETS,
    request_counter_snapshot,
    request_hist_snapshot,
)


def _line(name: str, value, labels: dict) -> str:
    """渲染一行 Prometheus exposition 文本。"""
    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    return f"{name}{label_str} {value}"


def _render_histogram(lines: list):
    """渲染 http_request_duration_seconds 直方图(bucket / sum / count)。

    注意:middleware._observe_duration 已经对每个 le >= duration 的桶 +1,
    因此 buckets[i] 直接就是 "累计 <= le[i] 的样本数",这里直接输出,不再累加。
    """
    for path, h in request_hist_snapshot().items():
        for i, le in enumerate(HISTOGRAM_BUCKETS):
            lines.append(_line(
                "http_request_duration_seconds_bucket",
                h["buckets"][i], {"path": path, "le": le},
            ))
        # Prometheus 要求最后一个桶必为 +Inf,值等于总 count
        lines.append(_line(
            "http_request_duration_seconds_bucket",
            h["count"], {"path": path, "le": "+Inf"},
        ))
        lines.append(_line(
            "http_request_duration_seconds_sum",
            round(h["sum"], 6), {"path": path},
        ))
        lines.append(_line(
            "http_request_duration_seconds_count",
            h["count"], {"path": path},
        ))


async def render_metrics(actors: dict) -> str:
    """聚合 actor 健康 + GPU 显存 + HTTP 请求计数 + 时延直方图,返回 Prometheus 文本。

    Args:
        actors: {logical_name: ray_actor_handle}
    Returns:
        Prometheus exposition text(以换行结尾)。
    """
    lines = []

    # actor 健康 + 累计 + GPU 显存(每 actor 独立进程,显存反映该 actor 自身占用)
    for name, actor in actors.items():
        labels = {"model": name}
        try:
            h = await asyncio.wrap_future(actor.health_check.remote().future())
        except Exception:
            lines.append(_line("ray_actor_alive", 0, labels))
            continue
        lines.append(_line("ray_actor_alive", 1 if h["alive"] else 0, labels))
        lines.append(_line("ray_actor_requests_total", h["total_requests"], labels))
        lines.append(_line("ray_actor_errors_total", h["total_errors"], labels))
        lines.append(_line("ray_actor_avg_latency_ms", h["avg_latency_ms"], labels))
        if "gpu_memory_mb" in h:
            lines.append(_line("ray_actor_gpu_memory_mb", h["gpu_memory_mb"], labels))

    # HTTP 入口按 (path, status) 拆分的请求计数
    for (path, status), count in request_counter_snapshot().items():
        lines.append(_line("http_requests_total", count, {"path": path, "status": status}))

    # 请求时延直方图(每 path 一组 bucket/sum/count)
    _render_histogram(lines)

    return "\n".join(lines) + "\n"
