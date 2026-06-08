"""Prometheus 文本格式指标导出,从各 Actor 的 health_check 聚合。"""
import asyncio


def _line(name: str, value, labels: dict) -> str:
    """渲染单行 Prometheus exposition 文本。"""
    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
    return f"{name}{label_str} {value}"


async def render_metrics(actors: dict) -> str:
    """拉取所有 actor 的 health_check 并渲染为 Prometheus 文本。

    Args:
        actors: {logical_name: ray_actor_handle}
    Returns:
        Prometheus exposition text(以换行结尾)。
    """
    lines = []
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
    return "\n".join(lines) + "\n"
