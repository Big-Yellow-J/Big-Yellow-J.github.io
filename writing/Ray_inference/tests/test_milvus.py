"""services.db.milvus 集成测试:用真实 Milvus Lite(临时文件)。

这些测试不需要 Ray / 模型,只验证 collection 管理、upsert、search、ping。
"""
import pytest


def test_source_hash_stable():
    from services.db.milvus import source_hash
    a = source_hash("https://example.com/cat.jpg")
    b = source_hash("https://example.com/cat.jpg")
    c = source_hash("https://example.com/dog.jpg")
    assert a == b
    assert a != c
    assert len(a) == 16


def test_collection_for():
    from services.db.milvus import collection_for
    assert collection_for("clip") == "embeddings_clip"
    assert collection_for("qwen_vl") == "embeddings_qwen_vl"


def test_ensure_collection_creates(tmp_milvus_uri):
    from services.db.milvus import collection_for, ensure_collection, get_milvus
    client = get_milvus()
    name = ensure_collection("clip", dim=4)
    assert name == collection_for("clip")
    assert name in client.list_collections()


def test_ensure_collection_idempotent(tmp_milvus_uri):
    from services.db.milvus import ensure_collection
    n1 = ensure_collection("clip", dim=4)
    n2 = ensure_collection("clip", dim=4)
    assert n1 == n2                     # 重复调用安全


def test_upsert_then_search(tmp_milvus_uri):
    """两条向量入库后,用最相似的向量检索,top1 必须命中自己。"""
    from services.db.milvus import (
        ensure_collection,
        search_vectors,
        upsert_embedding,
    )
    ensure_collection("clip", dim=4)
    v_cat = [1.0, 0.0, 0.0, 0.0]
    v_dog = [0.0, 1.0, 0.0, 0.0]
    pid_cat = upsert_embedding("clip", "/img/cat.jpg", v_cat, {"label": "cat"})
    pid_dog = upsert_embedding("clip", "/img/dog.jpg", v_dog, {"label": "dog"})
    assert pid_cat != pid_dog

    hits = search_vectors("clip", v_cat, top_k=2)
    assert len(hits) == 2
    # 第一名应该是 cat(余弦相似度最高)
    assert hits[0]["entity"]["source"] == "/img/cat.jpg"


def test_upsert_idempotent(tmp_milvus_uri):
    """同 source 重复 upsert 应该只产生一条记录(主键 = source md5)。"""
    from services.db.milvus import (
        ensure_collection,
        get_milvus,
        upsert_embedding,
    )
    ensure_collection("clip", dim=4)
    upsert_embedding("clip", "/img/cat.jpg", [1, 0, 0, 0])
    upsert_embedding("clip", "/img/cat.jpg", [1, 0, 0, 0])
    upsert_embedding("clip", "/img/cat.jpg", [1, 0, 0, 0])
    # Lite 也支持 query;通过 list_collections + get_collection_stats 看行数
    client = get_milvus()
    stats = client.get_collection_stats("embeddings_clip")
    # pymilvus 2.4 返回 dict-like,row_count 或 count 字段
    rc = stats.get("row_count", stats.get("count", None))
    if rc is not None:
        assert int(rc) == 1


def test_ping(tmp_milvus_uri):
    from services.db.milvus import ping
    assert ping() is True
