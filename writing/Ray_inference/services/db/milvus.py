"""Milvus 客户端 + collection schema 管理。

设计:
- 默认走 Milvus Lite(data/milvus_lite.db 单文件,无需 server)。
- 每个 embedding 模型一个 collection,命名 embeddings_<model>;维度可不同。
- schema 固定字段:id(source md5)、vector、source(原始 URL/路径)、model、metadata(JSON)、created_at。
- upsert 用 source md5 做主键,同图重复调用幂等;切换模型对同 source 重 embed 后写到对应 collection,互不干扰。
"""
import hashlib
import time
from typing import Optional

from pymilvus import DataType, MilvusClient

from config import (
    DATA_DIR,
    MILVUS_COLLECTION_PREFIX,
    MILVUS_METRIC,
    MILVUS_TOKEN,
    MILVUS_URI,
)
from utils.logging_setup import setup_logger

log = setup_logger("milvus")

_client: Optional[MilvusClient] = None
# 进程内已 load 的 collection 缓存:重启服务 / pymilvus 2.4+ 收紧要求时,search/upsert 前必须确保 load
_loaded_collections: set = set()


def _resolved_uri() -> str:
    """Milvus Lite 用 .db 文件路径时确保父目录存在。"""
    if MILVUS_URI.endswith(".db"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    return MILVUS_URI


def get_milvus() -> MilvusClient:
    """单例:FastAPI Depends 用,首次调用建连接,后续复用。"""
    global _client
    if _client is None:
        _client = MilvusClient(uri=_resolved_uri(), token=MILVUS_TOKEN or None)
        log.info("milvus connected uri=%s", _resolved_uri())
    return _client


def close_milvus() -> None:
    """优雅关闭(`@app.on_event('shutdown')` 调用)。"""
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None
    _loaded_collections.clear()


def _ensure_loaded(client: MilvusClient, name: str) -> None:
    """确保 collection 已 load 到内存(pymilvus 2.4+ 要求 search 前必须 load)。

    幂等:已 load 的进程内缓存跳过;真实 load 失败(集合刚建尚未 flush 等)忽略,
    下次调用再试。
    """
    if name in _loaded_collections:
        return
    try:
        client.load_collection(name)
    except Exception as e:
        log.warning("load_collection(%s) failed (will retry on next call): %s", name, e)
        return
    _loaded_collections.add(name)


def collection_for(model: str) -> str:
    """模型名 → collection 名(每模型独立)。"""
    return f"{MILVUS_COLLECTION_PREFIX}_{model}"


def ensure_collection(model: str, dim: int) -> str:
    """指定 (model, dim) 对应的 collection 不存在则建好,确保 load 后返回 collection 名。"""
    client = get_milvus()
    name = collection_for(model)
    if name in client.list_collections():
        _ensure_loaded(client, name)
        return name

    schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("source", DataType.VARCHAR, max_length=2048)
    schema.add_field("model", DataType.VARCHAR, max_length=64)
    schema.add_field("metadata", DataType.JSON)
    schema.add_field("created_at", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="HNSW",
        metric_type=MILVUS_METRIC,
        params={"M": 16, "efConstruction": 200},
    )
    client.create_collection(collection_name=name, schema=schema, index_params=index_params)
    log.info("milvus collection created name=%s dim=%d metric=%s", name, dim, MILVUS_METRIC)
    _ensure_loaded(client, name)
    return name


def source_hash(source: str) -> str:
    """source(path/URL/base64)→ 16 字符 md5,作为 milvus 主键(同图幂等)。"""
    s = source[:2048] if len(source) > 2048 else source
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]


def upsert_embedding(
    model: str,
    source: str,
    vector: list,
    metadata: Optional[dict] = None,
) -> str:
    """写入或更新一条向量,返回主键 id。"""
    client = get_milvus()
    name = collection_for(model)
    _ensure_loaded(client, name)
    pid = source_hash(source)
    client.upsert(
        collection_name=name,
        data=[{
            "id": pid,
            "vector": vector,
            "source": source,
            "model": model,
            "metadata": metadata or {},
            "created_at": int(time.time()),
        }],
    )
    return pid


def search_vectors(
    model: str,
    vector: list,
    top_k: int = 10,
    filter_expr: Optional[str] = None,
) -> list:
    """top_k 相似检索,返回命中条目(含 source / metadata / distance)。

    注意 pymilvus 2.4+ 不接受空字符串作为 filter,会报 "expression must evaluate to bool"。
    None / 空串 / 纯空白都视为"无过滤",此时不向 search() 传 filter 参数。
    """
    client = get_milvus()
    name = collection_for(model)
    _ensure_loaded(client, name)
    kwargs = {
        "collection_name": name,
        "data": [vector],
        "limit": top_k,
        "output_fields": ["source", "model", "metadata", "created_at"],
    }
    if filter_expr and filter_expr.strip():
        kwargs["filter"] = filter_expr
    hits = client.search(**kwargs)
    return hits[0]


def ping() -> bool:
    """轻量探活:列 collection 即可触发连接。失败抛异常,由调用方转 503。"""
    get_milvus().list_collections()
    return True
