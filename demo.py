# -*- coding: utf-8 -*-
"""RAG 文档问答 Demo —— 交互式命令行演示

前置：先在项目根目录跑 python src/ingest_full.py 建好向量库
运行：python demo.py
"""

import httpx
import chromadb
from openai import OpenAI

# 不走系统代理（本机开着 127.0.0.1:7897 代理会干扰本地请求）
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(trust_env=False),
)
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:0.5b"   # 显存大时可换成 qwen3:4b，效果更好


def main():
    print("=" * 58)
    print("  RAG 文档问答 Demo")
    print("  知识库：Qwen 技术报告（371 块）")
    print("  模型：", CHAT_MODEL)
    print("=" * 58)

    chroma = chromadb.PersistentClient(path="chroma_db")
    coll = chroma.get_or_create_collection("docs")
    count = coll.count()
    if count == 0:
        print("向量库为空，请先运行：python src/ingest_full.py")
        return
    print(f"向量库已加载：{count} 块\n")

    while True:
        try:
            q = input("请输入问题（直接回车退出）：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break

        # ① 问题向量化
        qvec = client.embeddings.create(model=EMBED_MODEL, input=[q]).data[0].embedding

        # ② 检索最相近的 2 段
        res = coll.query(query_embeddings=[qvec], n_results=2)
        docs = res["documents"][0]

        # ③ 生成回答
        context = "\n\n---\n\n".join(d[:400] for d in docs)
        prompt = (
            "请根据下面提供的资料回答问题。只依据资料内容作答；"
            "资料里没有的信息就回答“资料中未提及”。\n\n"
            f"资料：\n{context}\n\n问题：{q}"
        )
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )

        print("\n【回答】")
        print(resp.choices[0].message.content)
        print("\n【参考来源】")
        for i, d in enumerate(docs, 1):
            print(f"  [{i}] {d[:150]}...")
        print()


if __name__ == "__main__":
    main()
