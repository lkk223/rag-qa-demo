# 第 5 课参考实现：检索 + 生成（RAG 问答）
# 依赖：pip install chromadb openai httpx
# 运行：python src/query.py   （在项目根目录下）

import httpx
import chromadb
from openai import OpenAI

# 关键：禁止 httpx 读取系统代理（你电脑开着 127.0.0.1:7897 代理）
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    http_client=http_client,
)
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b"


def embed_one(text):
    """把单个文本变成向量"""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def ask(question, top_k=3):
    """检索 + 生成：返回 (答案, 参考片段列表)"""
    # 打开向量库
    chroma = chromadb.PersistentClient(path="chroma_db")
    coll = chroma.get_or_create_collection("docs")

    # ① 问题变向量
    qvec = embed_one(question)

    # ② 检索最相似的 top_k 段
    res = coll.query(query_embeddings=[qvec], n_results=top_k)
    docs = res["documents"][0]

    # ③ 拼成提示词，交给大模型
    context = "\n\n---\n\n".join(docs)
    prompt = f"""请根据下面提供的资料回答问题。
要求：只依据资料内容作答；资料里没有的信息就回答"资料中未提及"。

资料：
{context}

问题：{question}"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content, docs


if __name__ == "__main__":
    question = input("请输入你的问题：")
    answer, docs = ask(question)

    print("\n=== 回答 ===\n")
    print(answer)
    print("\n=== 参考片段（前 3 段检索结果）===")
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] {d[:180]}...")
