import os
import time
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from pymilvus import MilvusClient, DataType, FieldSchema
from sentence_transformers import SentenceTransformer

def read_txt(path, max_length=2048):
    def split_by_sentences(text, max_bytes):
        sentences = []
        current_sentence = ""
        delimiters = ['。', '！', '？', '；', '\n', '.', '!', '?', ';']
        for char in text:
            # 计算当前字节长度
            if len((current_sentence + char).encode("utf-8")) >= max_bytes:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = char  # 当前字符放下一段
            else:
                current_sentence += char

            # 遇到分隔符则切分
            if char in delimiters:
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""

        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        return sentences

    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if len(line.encode("utf-8")) > max_length:
                chunks = split_by_sentences(line, max_length)
                data.extend(chunks)
            else:
                data.append(line)
    return data

def embedding_model(model_name, cache_dir= '/mnt/d/TransformersCache'):
    model = SentenceTransformer(model_name,
                                # model_kwargs={"attn_implementation": "flash_attention_2"},
                                cache_folder= cache_dir)
    return model

def connect_milvus(host= "127.0.0.1", port= "19530", token= "root:Milvus", max_connect= 5):
    connect_steps = 1
    while connect_steps<= max_connect:
        try:
            client = MilvusClient(uri= f"http://{host}:{port}",
                                token= token)
            print("连接成功！")
            return client
        except:
            connect_steps +=1
    raise ConnectionError("连接失败！！！")

def create_collections(client, collection_name, fields=None, params_idnexs=None,
                       max_try_steps=3):
    # 具体参数：https://milvus.io/docs/zh/create-collection.md
    for attempt in range(1, max_try_steps + 1):
        try:
            if fields is None:
                schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
                schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
                schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
            else:
                schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
                for field in fields:
                    schema.add_field(**field)
            
            # 创建索引
            if params_idnexs:
                index_params = client.prepare_index_params()
                for params_idnex in params_idnexs:
                    index_params.add_index(**params_idnex)
                    index_params.add_index(**params_idnex)
            else:
                index_params= None
            
            # 创建集合
            client.create_collection(collection_name=collection_name,
                                     schema=schema,
                                     shards_num=1,
                                     index_params= index_params)
            print(f"集合 '{collection_name}' 创建成功！")
            return
        except Exception as e:
            print(f"第 {attempt} 次创建集合失败: {e}")
            if attempt == max_try_steps:
                raise RuntimeError(f"创建集合失败，已重试 {max_try_steps} 次")

def embedding_txt(embedding_fn, docs_dict, batch_size=16, embedding_dim=1024, 
                  cache_dir= './cache/'):
    result = {}
    for filename, texts in docs_dict.items():
        print(f"\n正在处理文件: {filename} ({len(texts)} 条文本)")
        # Step 1: 去重 + 建立映射
        text_to_indices = {}
        unique_texts = []
        for i, text in enumerate(texts):
            if text not in text_to_indices:
                text_to_indices[text] = []
                unique_texts.append(text)
            text_to_indices[text].append(i)

        total_unique = len(unique_texts)
        print(f"  去重后: {total_unique} 条")

        # Step 2: 预分配结果数组
        vectors_unique = np.zeros((total_unique, embedding_dim), dtype=np.float32)

        # Step 3: 批量编码，直接写入预分配数组
        idx = 0
        for i in tqdm(range(0, total_unique, batch_size), desc="  Embedding", leave=False):
            batch = unique_texts[i:i + batch_size]
            batch_vec = embedding_fn.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True
            )  # (b, dim)

            b = batch_vec.shape[0]
            vectors_unique[idx:idx + b] = batch_vec
            idx += b

            # 立即释放临时变量
            del batch_vec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 4: 构建结果（一一对应，重复文本复用 vector）
        file_result = []
        vector_idx = 0
        seen_texts = set()

        for text in texts:
            if text not in seen_texts:
                # 第一次遇到，分配新 vector
                vector = vectors_unique[vector_idx].tolist()
                vector_idx += 1
                seen_texts.add(text)
            # 重复文本直接复用
            file_result.append({
                "text": text,
                "vector": vector
            })
        result[filename] = file_result
        print(f"  完成！共 {len(file_result)} 条记录")
    for filename, file_result in result.items():
        # 提取文本与向量
        texts = [item["text"] for item in file_result]
        vectors = np.array([item["vector"] for item in file_result], dtype=np.float32)
        # 保存 JSONL
        json_path = os.path.join(cache_dir, f"{filename}.jsonl")
        with open(json_path, "w", encoding="utf-8") as f:
            for text in texts:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        # 保存向量文件
        np.save(os.path.join(cache_dir, f"{filename}.npy"), vectors)
        print(f"✅ 已保存 {filename}: {len(texts)} 条文本, 向量 shape={vectors.shape}")
    return result

def load_embedding(cache_dir, embedding_dim=768):
    data = []
    global_id = 0
    for file in sorted(os.listdir(cache_dir)):
        if not file.endswith(".jsonl"):
            continue

        prefix = os.path.splitext(file)[0]
        json_path = os.path.join(cache_dir, f"{prefix}.jsonl")
        npy_path = os.path.join(cache_dir, f"{prefix}.npy")

        if not os.path.exists(npy_path):
            print(f"⚠️ 向量文件缺失: {npy_path}")
            continue

        # 读取文本
        texts = []
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                texts.append(json.loads(line)["text"])

        # 读取向量
        vectors = np.load(npy_path)
        if vectors.shape[1] != embedding_dim:
            raise ValueError(f"维度不匹配: {vectors.shape[1]} vs {embedding_dim}")

        # 构造 Milvus 数据
        for text, vec in zip(texts, vectors):
            data.append({
                "id": global_id,
                "vector": vec.tolist(),
                "text": text,
                "tag": prefix
            })
            global_id += 1

        print(f"✅ 已加载 {prefix}: {len(texts)} 条记录")

    print(f"📦 数据总量: {len(data)} 条")
    return data

def insert_batches(client, collection_name, data, max_bytes=60 * 1024 * 1024):
    def estimate_size(obj):
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    batch, current_size = [], 0
    total = len(data)
    count = 0
    for i, item in enumerate(data):
        item_size = estimate_size(item)
        if current_size + item_size > max_bytes:
            client.insert(collection_name=collection_name, data=batch)
            count += 1
            print(f"✅ 已插入第 {count} 批，共 {len(batch)}/{len(data)} 条，约 {current_size/1024/1024:.2f}MB")
            batch, current_size = [], 0
        batch.append(item)
        current_size += item_size

    if batch:
        client.insert(collection_name=collection_name, data=batch)
        count += 1
        print(f"✅ 已插入最后一批，共 {len(batch)} 条")

    print(f"🎉 插入完成，共 {count} 批，总计 {total} 条")

if __name__ == '__main__':
    import pprint
    from dotenv import load_dotenv
    load_dotenv('./HOST.env')
    host = os.getenv('host')
    collection_name = 'FourGreatClassics'
    user_name_password = 'root:Milvus'
    model_name = 'moka-ai/m3e-base' #1024:'Qwen/Qwen3-Embedding-0.6B' 768:'sentence-transformers/paraphrase-albert-small-v2'

    max_length = 2048
    fields = [
        {"field_name": "id", "datatype": DataType.INT64, "is_primary": True},
        {"field_name": "vector", "datatype": DataType.FLOAT_VECTOR, "dim": 768},
        {"field_name": "text", "datatype": DataType.VARCHAR, "max_length": max_length},
        {"field_name": "tag", "datatype": DataType.VARCHAR, "max_length": 1024},
    ]
    params_idnex = [
        {'field_name': 'id', 'index_type': 'AUTOINDEX'},
        {'field_name': 'vector', 'index_type': 'AUTOINDEX', 'metric_type': 'COSINE'}, # 对于QwenEmbedding模型计算相似度方式“内积” IP
    ]

    client = connect_milvus(token= user_name_password, host= host)
    # if client.has_collection(collection_name):
    #     client.drop_collection(collection_name)
    #     print(f"已删除现有集合: {collection_name}")

    # 创建
    create_collections(client= client, collection_name= collection_name, fields= fields,
                       params_idnexs= params_idnex)

    # 插入数据
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    embedding_fn = embedding_model(model_name)
    embedding_fn.to(device)

    # 直接编码
    # paths = './data/'
    # docs = {}
    # data = []
    # global_id = 0
    # for path in sorted(os.listdir(paths)):
    #     docs[os.path.basename(path).split('.')[0]] = read_txt(os.path.join(paths, path), max_length)
    # embedding_result = embedding_txt(embedding_fn, docs, 16, 768)
    # for filename, items in embedding_result.items():
    #     for item in items:
    #         data.append({
    #             "id": global_id,
    #             "vector": item["vector"],
    #             "text": item["text"],
    #             "tag": filename
    #         })
    #         global_id += 1
    # 加载处理过的
    # data = load_embedding(cache_dir= './cache/')
    # res = client.insert(collection_name= collection_name, data= data)
    # insert_batches(client, collection_name, data)

    # 搜索数据
    client.load_collection(collection_name)
    res = client.search(
        collection_name= collection_name,
        data= embedding_fn.encode(['武松']),
        limit=2,
        filter= "tag == 'AllMenAreBrothers'",
        output_fields=["text", "tag"],
    )
    pprint.pprint(res)

    # 删除实体
    # res = client.delete(collection_name= collection_name, ids=[0, 2])
    # res = client.delete(collection_name= collection_name, filter="subject == 'People'",)