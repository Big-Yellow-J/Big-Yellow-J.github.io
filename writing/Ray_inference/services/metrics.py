"""Prometheus 文本格式指标:聚合 actor health + HTTP 请求计数。"""
import asyncio

from services.middleware import request_counter_snapshot


def _line(name: str, value, labels: dict) -> str:
    """渲染一行 Prometheus exposition 文本。"""
    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    return f"{name}{label_str} {value}"


async def render_metrics(actors: dict) -> str:
    """聚合 actor 健康指标 + HTTP 请求计数,返回 Prometheus 文本。

    Args:
        actors: {logical_name: ray_actor_handle}
    Returns:
        Prometheus exposition text(以换行结尾)。
    """
    lines = []

    # actor 健康/累计指标
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

    # HTTP 入口按 (path, status) 拆分的请求计数
    for (path, status), count in request_counter_snapshot().items():
        lines.append(_line("http_requests_total", count, {"path": path, "status": status}))

    return "\n".join(lines) + "\n"
