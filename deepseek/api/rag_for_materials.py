# -*- coding: utf-8 -*-
# coding: utf-8
# @Time        : 2025/6/10
# @Author      : liuboyuan
import json
import os
import time
from openai import OpenAI
from pymilvus import MilvusClient
from typing import List, Tuple
import re
import pickle

from tqdm import tqdm

# 导入公共配置
from config import (
    EMBEDDING_MODEL_NAME,
    SEARCH_ENHANCE_MODEL_NAME,
    gemini_base_url,
    gemini_api_key
)

# 连接到Milvus
def connect_to_milvus(uri="./milvus_demo.db"):
    # connections.connect(alias="default", uri=uri)
    # print("已连接到本地Milvus数据库")
    # windows下本地库不可用。
    milvus = MilvusClient(uri=uri)
    return milvus

# 创建Milvus集合
def create_collection(client, collection_name='article_collection', embedding_dim=1024):
    client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",  # 内积距离
        consistency_level="Strong",
        # 支持的值为 (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`)。更多详情请参见 https://milvus.io/docs/consistency.md#Consistency-Level。
    )
    return client

# 使用Gemini Embedding模型生成向量
def get_embedding(text: str, model=EMBEDDING_MODEL_NAME) -> List[float]:
    client = OpenAI(
        api_key=gemini_api_key,
        base_url=gemini_base_url
    )
    
    response = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    
    # 确保所有的值都是浮点数
    embedding = response.data[0].embedding
    return [float(val) for val in embedding]

# 生成多个查询向量
def generate_multiple_queries(question: str, num_queries=3) -> List[str]:
    """生成多个相关查询以获得更全面的结果"""
    # 在实际应用中，可以使用LLM生成多种表述的查询
    variations = [
        question,  # 原始问题
        f"关于{question}的内容",  # 变体1
        f"{question}相关的规定",  # 变体2
    ]
    return variations[:num_queries]  # 返回指定数量的查询变体

# 合并多个查询结果
def merge_search_results(all_results, top_k=10):
    """合并多个查询的结果并按相似度排序"""
    # 使用字典合并相同文本的结果，取最高分数
    merged = {}
    for results in all_results:
        for text, score in results:
            if text in merged:
                merged[text] = max(merged[text], score)
            else:
                merged[text] = score
    
    # 排序并返回前top_k个结果
    sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# 将文本分块
def split_text_into_chunks(text: str, chunk_size=500, overlap=50) -> List[str]:
    # 通过标题分割文本
    sections = re.split(r'(?=^#+\s+)', text, flags=re.MULTILINE)
    
    chunks = []
    for section in sections:
        if not section.strip():
            continue
            
        words = section.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
    
    return chunks

# 保存嵌入向量到文件
def save_embeddings(embeddings, texts, file_path="embedding.txt"):
    data_to_save = {
        "embeddings": embeddings,
        "texts": texts
    }
    with open(file_path, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"嵌入向量已保存到 {file_path}")

# 从文件加载嵌入向量
def load_embeddings(file_path="embedding.txt"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    # 确保所有嵌入向量的值都是浮点数
    embeddings = []
    for embedding in data["embeddings"]:
        embeddings.append([float(val) for val in embedding])
    
    return embeddings, data["texts"]

# 语义重排序，使用LLM重新排序搜索结果
def semantic_rerank(query: str, results: List[Tuple[str, float]], top_k=5) -> List[Tuple[str, float]]:
    """使用LLM对搜索结果进行语义重排序"""
    if not results:
        return []
    
    # 如果结果数量少于2，无需重排
    if len(results) < 2:
        return results
    
    client = OpenAI(
        api_key=gemini_api_key,
        base_url=gemini_base_url,
    )
    
    # 构建提示词，要求模型给每个结果打分
    texts = [text for text, _ in results[:top_k]]
    prompt = f"""请根据查询"{query}"，对以下{len(texts)}个文本片段按照与查询的相关性从高到低进行排序。
给每个文本片段打分（0-10分），其中10分表示最相关，0分表示完全不相关。
只返回每个文本的分数，不要解释，每行输出一个分数。

文本片段:
""" + "\n\n".join([f"{i+1}. {text}" for i, text in enumerate(texts)])
    
    try:
        response = client.chat.completions.create(
            model=SEARCH_ENHANCE_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        # 解析分数
        scores_text = response.choices[0].message.content.strip()
        scores = []
        for line in scores_text.split('\n'):
            try:
                # 尝试提取数字
                score = float(re.search(r'(\d+(\.\d+)?)', line).group(1))
                scores.append(score)
            except (AttributeError, ValueError):
                # 如果行中没有找到数字，或无法转换为浮点数，则分配默认分数
                scores.append(5.0)
        
        # 确保分数列表长度与文本片段数量一致
        if len(scores) < len(texts):
            scores.extend([5.0] * (len(texts) - len(scores)))
        
        # 合并原始结果和新分数
        reranked_results = []
        for i, ((text, orig_score), new_score) in enumerate(zip(results[:top_k], scores)):
            # 结合原始向量相似度和LLM评分的加权分数
            combined_score = 0.3 * orig_score + 0.7 * (new_score / 10.0)
            reranked_results.append((text, combined_score))
        
        # 对结果重新排序
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # 添加未被重排的结果
        if len(results) > top_k:
            reranked_results.extend(results[top_k:])
        
        return reranked_results
    except Exception as e:
        print(f"语义重排序出错: {e}")
        # 出错时返回原始结果
        return results

# 主函数
def search(question, need_recreate_db = False):
    print("进入搜索主函数")
    # Milvus lite不支持windows, 直接采用Docker...
    # 连接本地Milvus数据库
    client = connect_to_milvus(uri="http://localhost:19530")
    collection_name = "article_collection_x3"
    # 估算维度
    # embedding_text = get_embedding(
    #     "衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"
    # )
    # print(len(embedding_text))
    ## 离线部分
    if need_recreate_db:
        # 创建集合
        create_collection(client, collection_name=collection_name, embedding_dim=768)
        # 读取文件内容
        with open("./article.md", 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = split_text_into_chunks(content, chunk_size=500, overlap=50)
            print(chunks)
            for c in chunks:
                print(c)
            print(len(chunks))

        # 检查嵌入向量文件是否存在
        embedding_file = "embedding.txt"
        if os.path.exists(embedding_file):
            print(f"从文件 {embedding_file} 加载嵌入向量")
            doc_embeddings, saved_chunks = load_embeddings(embedding_file)
        else:
            print("生成并保存嵌入向量")
            doc_embeddings = []
            for w in chunks:
                p = get_embedding(w)
                doc_embeddings.append(p)
                time.sleep(0.5)
            save_embeddings(doc_embeddings, chunks, embedding_file)

        data = []
        for i, line in enumerate(tqdm(chunks, desc="Creating embeddings")):
            # 确保向量中的所有元素都是浮点数
            vector = doc_embeddings[i]
            if not all(isinstance(val, float) for val in vector):
                vector = [float(val) for val in vector]

            data.append(
                {
                    "id": i,
                    "vector": vector,
                    "text": line
                }
            )
        try:
            client.insert(collection_name=collection_name, data=data)
            print(f"成功插入 {len(data)} 条数据到集合 {collection_name}")
        except Exception as e:
            print(f"插入数据出错: {e}")
            print(f"错误类型: {type(e)}")


    try:
        # 生成多个查询变体
        query_variations = generate_multiple_queries(question)
        all_results = []
        print("生成查询变体结束。")
        # 对每个查询变体执行搜索
        for query in query_variations:
            print(f"执行查询: {query}")
            # 获取查询的嵌入向量
            query_vector = get_embedding(query)
            print(f"已获取向量。")
            search_res = client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=10,
                search_params={
                    "metric_type": "IP", 
                    "params": {
                        "level": 3,
                        "nprobe": 20,
                        "ef": 100
                    }
                },
                output_fields=["text"],
            )
            
            # 提取结果
            results = [(res["entity"]["text"], float(res["distance"])) for res in search_res[0]]
            all_results.append(results)
        
        # 合并结果
        merged_results = merge_search_results(all_results)
        
        # 对搜索结果进行后处理
        threshold = 0.4
        filtered_results = [
            (text, distance) for text, distance in merged_results 
            if distance > threshold
        ]
        
        # 如果过滤后结果太少，则使用原始结果
        if len(filtered_results) < 3:
            filtered_results = merged_results
            
        # 对结果进行语义重排序
        reranked_results = semantic_rerank(question, filtered_results)
        
        print("最终搜索结果:")
        print(json.dumps(reranked_results, indent=4, ensure_ascii=False))
        
        return reranked_results
    except Exception as e:
        print(f"搜索出错: {e}")
        print(f"错误类型: {type(e)}")
        return []


if __name__ == "__main__":
    # 离线导入任务, 完成为True
    question = "过程管理 总则"
    need_recreate_db_flag = False
    search(question, need_recreate_db_flag)
