import os
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai  # 可替换为任何兼容 OpenAI 接口的 API (如小米 MiLM)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MindFlow")

# ==================== 配置 ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 本地嵌入模型，轻量高效
LLM_MODEL = "gpt-3.5-turbo"                # 可替换为小米 MiLM
VECTOR_DIM = 384                            # all-MiniLM-L6-v2 输出维度
INDEX_PATH = "mindflow_index.faiss"        # 向量索引文件
META_PATH = "mindflow_meta.json"            # 元数据(文本、标签、图谱)
CONFLICT_THRESHOLD = 0.85                   # 相似度冲突阈值
# ==============================================

openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_API_BASE

# 全局嵌入模型 (本地运行，不消耗 API Token)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# 全局向量索引
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatIP(VECTOR_DIM)  # 内积相似度

# 全局元数据列表，与向量索引中位置一一对应
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = []

def save_index_and_meta():
    """持久化索引与元数据"""
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

# ==================== Agent 定义 ====================

class CollectionAgent:
    """
    采集 Agent：将碎片信息（文本）转化为语义标签，嵌入并存入知识库。
    实际应用时可扩展为浏览器插件、API 抓取等入口。
    """
    @staticmethod
    def add_knowledge(text: str, source: str = "manual") -> str:
        # 1. 产成语义标签 (利用 LLM 提取关键词)
        prompt = f"请从以下文本中提取 3-5 个核心关键词（标签），用逗号分隔。只输出关键词，不要其他内容。\n文本：{text}"
        tags = chat_with_llm(prompt).strip().split(",")
        tags = [t.strip() for t in tags if t.strip()]

        # 2. 向量化
        vector = embedder.encode([text])[0].astype(np.float32)
        # 3. 存入索引
        idx = index.ntotal
        index.add(np.array([vector], dtype=np.float32))
        # 4. 记录元数据
        meta_entry = {
            "id": idx,
            "text": text,
            "tags": tags,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "relations": []  # 将由整理 Agent 填充
        }
        metadata.append(meta_entry)
        save_index_and_meta()
        logger.info(f"已采集知识 #{idx}: {text[:50]}... 标签: {tags}")
        return f"成功添加知识，ID={idx}，标签：{tags}"

    @staticmethod
    def import_from_file(filepath: str):
        """从文本文件批量导入，每行一条知识"""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    CollectionAgent.add_knowledge(line.strip(), source=filepath)


class OrganizationAgent:
    """
    整理 Agent：定期运行，进行冲突检测、去重，并构建知识图谱（反向链接）。
    核心逻辑包含长链推理：通过比较知识对，推断矛盾或一致性，生成关系图谱。
    """
    @staticmethod
    def detect_conflicts_and_build_graph():
        if len(metadata) < 2:
            return "知识条目过少，无需整理。"

        texts = [m["text"] for m in metadata]
        vectors = np.array([embedder.encode([t])[0] for t in texts], dtype=np.float32)

        # 相似度矩阵 (内积搜索近似余弦)
        faiss.normalize_L2(vectors)
        index_temp = faiss.IndexFlatIP(VECTOR_DIM)
        index_temp.add(vectors)
        # 查询每个条目的最近邻 (自身除外)
        k = min(5, len(metadata))  # 最多取5个最相似
        distances, indices = index_temp.search(vectors, k)

        conflict_count = 0
        new_relations = 0

        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            for d, j in zip(dist_row, idx_row):
                if i == j:  # 跳过自身
                    continue
                if d > CONFLICT_THRESHOLD:
                    # 相似度过高，可能冲突或重复，启用长链推理判断
                    type_ = OrganizationAgent._analyze_pair(i, j, d)
                    if type_ in ("conflict", "duplicate"):
                        conflict_count += 1
                    # 建立双向关系
                    if j not in metadata[i]["relations"]:
                        metadata[i]["relations"].append(j)
                        new_relations += 1
                    if i not in metadata[j]["relations"]:
                        metadata[j]["relations"].append(i)
                        new_relations += 1
        save_index_and_meta()
        logger.info(f"整理完成：发现 {conflict_count} 处冲突/重复，新建 {new_relations} 条关联。")
        return f"整理完成，冲突/重复: {conflict_count}，新建图谱关联: {new_relations}。"

    @staticmethod
    def _analyze_pair(i: int, j: int, similarity: float) -> str:
        """利用长链推理判断两条高度相似知识的关系：重复、矛盾或互补"""
        a = metadata[i]["text"]
        b = metadata[j]["text"]
        prompt = (
            f"请判断以下两段知识的关系，只回答一个词：duplicate（完全重复）、conflict（互相矛盾）或 complementary（互补）。\n"
            f"知识A：{a}\n知识B：{b}\n两者相似度：{similarity:.2f}"
        )
        answer = chat_with_llm(prompt).strip().lower()
        if "duplicate" in answer:
            return "duplicate"
        elif "conflict" in answer:
            return "conflict"
        else:
            return "complementary"


class DialogueAgent:
    """
    对话 Agent：接收用户问题，执行多跳检索与长链推理，生成带引用的答案。
    """
    @staticmethod
    def answer(query: str) -> str:
        # 1. 向量检索 + 知识图谱扩展（多跳）
        primary_hits = DialogueAgent._vector_search(query, top_k=5)
        # 扩展至它们关联的节点（一跳邻居）
        expanded_ids = set()
        for hit in primary_hits:
            expanded_ids.add(hit["id"])
            for rel in metadata[hit["id"]]["relations"]:
                expanded_ids.add(rel)
        # 收集扩展后的上下文
        context_entries = [metadata[i] for i in expanded_ids if i < len(metadata)]
        # 2. 长链推理: 利用上下文进行假设验证、归纳
        context_text = "\n---\n".join(
            f"[ID:{m['id']}] {m['text']}" for m in context_entries[:8]  # 控制长度
        )
        prompt = f"""你是一个拥有个人知识库的AI助手。请根据以下知识片段回答问题。
要求：
- 基于提供的知识，不要编造事实。
- 回答中包含推理过程（链式思考），并给出最终答案。
- 引用的地方标注知识ID，例如 [ID:3]

知识库内容：
{context_text}

问题：{query}

请逐步推理并回答。"""
        answer = chat_with_llm(prompt)
        return answer

    @staticmethod
    def _vector_search(query: str, top_k=5) -> List[Dict]:
        q_vec = embedder.encode([query])[0].astype(np.float32)
        q_vec = q_vec / np.linalg.norm(q_vec)
        distances, indices = index.search(np.array([q_vec], dtype=np.float32), top_k)
        hits = []
        for d, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(metadata):
                hits.append({"id": int(idx), "score": float(d)})
        return hits


# ==================== 工具函数 ====================
def chat_with_llm(prompt: str) -> str:
    """调用 LLM，可无痛替换为小米 MiLM 或其他本地模型"""
    try:
        resp = openai.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM 调用失败: {e}")
        return ""


def schedule_organization(interval_hours=24):
    """模拟定时任务，循环执行整理 Agent"""
    import threading
    def job():
        while True:
            time.sleep(interval_hours * 3600)
            logger.info("启动定时整理任务...")
            OrganizationAgent.detect_conflicts_and_build_graph()
    t = threading.Thread(target=job, daemon=True)
    t.start()
    logger.info(f"已启动整理 Agent 定时器，每 {interval_hours} 小时运行一次。")


# ==================== 演示主流程 ====================
if __name__ == "__main__":
    # 启动后台整理 Agent (可选)
    # schedule_organization(interval_hours=1)

    # 模拟采集阶段
    print("=== 采集多条知识 ===")
    CollectionAgent.add_knowledge("小米汽车使用澎湃OS作为车机系统，支持手机无缝互联。")
    CollectionAgent.add_knowledge("澎湃OS是小米自研的操作系统，基于Android深度定制。")
    CollectionAgent.add_knowledge("小米汽车的车机系统是基于澎湃OS开发的，拥有强大的生态互联能力。")
    CollectionAgent.add_knowledge("苹果CarPlay不属于小米生态，与澎湃OS不兼容。")

    # 执行整理 (冲突检测与图谱构建)
    print("\n=== 执行整理 Agent ===")
    result = OrganizationAgent.detect_conflicts_and_build_graph()
    print(result)

    # 对话查询，展示多跳检索与长链推理
    print("\n=== 对话查询 ===")
    q1 = "小米汽车用什么系统？和苹果CarPlay兼容吗？"
    ans1 = DialogueAgent.answer(q1)
    print(f"Q: {q1}\nA: {ans1}")

    q2 = "澎湃OS的生态互联体现在哪些方面？"
    ans2 = DialogueAgent.answer(q2)
    print(f"\nQ: {q2}\nA: {ans2}")