"""图像 / 文本 Embedding + 向量检索路由。

端点:
    POST /v1/embed          图像 → 向量,可选写库
    POST /v1/embed_text     文本 → 向量(CLIP 跨模态)
    POST /v1/search         以图搜图
    POST /v1/search_text    以文搜图(CLIP 文本→在图像 collection 检索)

请求级缓存:同 (source_hash, model) 的 embedding 在 LRU 内命中即返回,跳过 actor。
"""
from collections import OrderedDict
from typing import Optional

from fastapi import APIRouter, HTTPException, Request

from config import EMBED_CACHE_SIZE
from services.db.milvus import (
    ensure_collection,
    search_vectors,
    source_hash,
    upsert_embedding,
)
from services.dispatch import actor_call
from services.schemas import (
    EmbedBody,
    EmbedTextBody,
    SearchBody,
    SearchTextBody,
)

router = APIRouter(prefix="/v1", tags=["embed"])

# 业务模型名 → dispatch._RAY_NAMES 中的 actor key
_MODEL_TO_ACTOR = {"clip": "clip", "qwen_vl": "qwen_embed"}

# 简单 LRU(进程内):key = (kind, model, content_hash) → {"embedding","dim","model"}
_cache: "OrderedDict[tuple, dict]" = OrderedDict()


def _cache_get(k: tuple) -> Optional[dict]:
    if k in _cache:
        _cache.move_to_end(k)
        return _cache[k]
    return None


def _cache_set(k: tuple, v: dict) -> None:
    _cache[k] = v
    _cache.move_to_end(k)
    while len(_cache) > EMBED_CACHE_SIZE:
        _cache.popitem(last=False)


def cache_stats() -> dict:
    """监控用:当前缓存大小。"""
    return {"size": len(_cache), "capacity": EMBED_CACHE_SIZE}


async def _do_embed(model: str, source: str, rid: str) -> dict:
    """图像 embed:走缓存 → 走 actor。"""
    actor_key = _MODEL_TO_ACTOR.get(model)
    if actor_key is None:
        raise HTTPException(400, f"unknown embedder model: {model}")

    ckey = ("img", model, source_hash(source))
    cached = _cache_get(ckey)
    if cached is not None:
        return {**cached, "cached": True}

    result = await actor_call(actor_key, "embed", source, {}, rid)
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "embed failed"))
    _cache_set(ckey, {k: result[k] for k in ("embedding", "dim", "model")})
    return {**result, "cached": False}


async def _do_embed_text(model: str, text: str, rid: str) -> dict:
    """文本 embed:目前仅 CLIP,走缓存 → 走 actor.embed_text。"""
    if model != "clip":
        raise HTTPException(400, f"text embedding only supports 'clip', got {model}")

    ckey = ("txt", model, source_hash(text))
    cached = _cache_get(ckey)
    if cached is not None:
        return {**cached, "cached": True}

    result = await actor_call("clip", "embed_text", text, {}, rid)
    if not result.get("success"):
        raise HTTPException(500, result.get("error", "embed_text failed"))
    _cache_set(ckey, {k: result[k] for k in ("embedding", "dim", "model")})
    return {**result, "cached": False}


@router.post("/embed")
async def embed(body: EmbedBody, request: Request):
    """source → 图像 embedding;默认顺手写入 milvus(write_db=false 跳过)。"""
    rid = request.state.request_id
    r = await _do_embed(body.model, body.source, rid)
    if not body.write_db:
        return {**r, "written": False}
    ensure_collection(body.model, r["dim"])
    pid = upsert_embedding(body.model, body.source, r["embedding"], body.metadata)
    return {**r, "written": True, "milvus_id": pid}


@router.post("/embed_text")
async def embed_text(body: EmbedTextBody, request: Request):
    """text → 文本 embedding(CLIP 跨模态可与图向量同空间检索)。"""
    rid = request.state.request_id
    r = await _do_embed_text(body.model, body.text, rid)
    if not body.write_db:
        return {**r, "written": False}
    ensure_collection(body.model, r["dim"])
    pid = upsert_embedding(body.model, body.text, r["embedding"], body.metadata)
    return {**r, "written": True, "milvus_id": pid}


@router.post("/search")
async def search(body: SearchBody, request: Request):
    """以图搜图:source → embedding → milvus.search。"""
    rid = request.state.request_id
    r = await _do_embed(body.model, body.source, rid)
    ensure_collection(body.model, r["dim"])
    hits = search_vectors(body.model, r["embedding"], body.top_k, body.filter)
    return {"model": body.model, "dim": r["dim"], "results": hits}


@router.post("/search_text")
async def search_text(body: SearchTextBody, request: Request):
    """以文搜图:text → embedding → 在 embeddings_clip 上检索(CLIP 文/图共享空间)。"""
    rid = request.state.request_id
    r = await _do_embed_text(body.model, body.text, rid)
    ensure_collection(body.model, r["dim"])
    hits = search_vectors(body.model, r["embedding"], body.top_k, body.filter)
    return {"model": body.model, "dim": r["dim"], "results": hits}
