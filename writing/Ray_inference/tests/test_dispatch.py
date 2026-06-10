"""dispatch 单元测试:句柄缓存 / RayActorError 自愈 / 熔断器。

不连 Ray,用 monkeypatch 替换 ray.get_actor + actor handle。
"""
import asyncio
import time
from unittest.mock import MagicMock

import pytest


class _FakeRef:
    """模拟 ray ObjectRef:.future() → asyncio Future,可控成功/失败/超时。"""
    def __init__(self, result=None, exc=None, delay=0.0):
        self._result, self._exc, self._delay = result, exc, delay

    def future(self):
        f = asyncio.get_event_loop().create_future()
        async def _set():
            await asyncio.sleep(self._delay)
            if self._exc:
                f.set_exception(self._exc)
            else:
                f.set_result(self._result)
        asyncio.ensure_future(_set())
        return f


def _make_actor(method_name="infer", side_effect=None):
    """造一个假 actor handle,getattr(actor, method).remote(...) → _FakeRef。"""
    actor = MagicMock()
    remote = MagicMock()
    if callable(side_effect):
        remote.side_effect = lambda *a, **kw: side_effect()
    else:
        remote.return_value = _FakeRef(result=side_effect)
    getattr(actor, method_name).remote = remote
    return actor


@pytest.mark.asyncio
async def test_actor_call_happy(monkeypatch):
    """正常调用,返回 actor 的结果。"""
    import services.dispatch as d
    actor = _make_actor("embed", side_effect={"success": True, "embedding": [1, 2]})
    monkeypatch.setattr(d, "_actors", {"clip": actor})
    monkeypatch.setattr(d, "_circuit", {})
    r = await d.actor_call("clip", "embed", "x.jpg", {}, rid="r1")
    assert r["success"] is True


@pytest.mark.asyncio
async def test_actor_call_refresh_on_actor_died(monkeypatch):
    """第一次 RayActorError,第二次 refresh 后成功。"""
    import services.dispatch as d
    from ray.exceptions import RayActorError

    calls = {"n": 0}

    def remote(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeRef(exc=RayActorError("died"))
        return _FakeRef(result={"success": True})

    actor = MagicMock()
    actor.embed.remote = remote
    monkeypatch.setattr(d, "_actors", {"clip": actor})
    monkeypatch.setattr(d, "_circuit", {})
    monkeypatch.setattr(d, "_refresh_actor", lambda key: d._actors.update({key: actor}))

    r = await d.actor_call("clip", "embed", "x.jpg", {}, rid="r2")
    assert r["success"] is True
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_circuit_opens_after_threshold(monkeypatch):
    """连续 CIRCUIT_FAIL_THRESHOLD 次失败 → 熔断打开 → 后续直接 503。"""
    import services.dispatch as d
    from fastapi import HTTPException
    from ray.exceptions import RayActorError

    monkeypatch.setattr(d, "CIRCUIT_FAIL_THRESHOLD", 3)
    monkeypatch.setattr(d, "CIRCUIT_OPEN_SEC", 60)
    monkeypatch.setattr(d, "CIRCUIT_WINDOW_SEC", 60)
    monkeypatch.setattr(d, "_circuit", {})

    actor = MagicMock()
    actor.embed.remote = lambda *a, **kw: _FakeRef(exc=RayActorError("dead"))
    monkeypatch.setattr(d, "_actors", {"clip": actor})
    monkeypatch.setattr(d, "_refresh_actor", lambda key: None)

    for _ in range(3):                  # 阈值次失败累计
        with pytest.raises(HTTPException):
            await d.actor_call("clip", "embed", "x.jpg", {}, rid="rid")

    # 第 N+1 次:熔断已开,应直接返回 503 而不进入 actor 调用
    with pytest.raises(HTTPException) as excinfo:
        await d.actor_call("clip", "embed", "x.jpg", {}, rid="rid")
    assert excinfo.value.status_code == 503
    assert "circuit open" in str(excinfo.value.detail)


def test_circuit_snapshot_returns_state(monkeypatch):
    import services.dispatch as d
    monkeypatch.setattr(d, "_circuit", {"clip": {"open_until": time.time() + 10, "fails": []}})
    snap = d.circuit_snapshot()
    assert snap["clip"]["open"] is True
