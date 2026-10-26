# 第 6 课（2/2）：RAG 评测脚本
# 依赖：chromadb openai httpx
# 运行：python src/eval.py   （在项目根目录下，先跑 make_testset.py）
# 输出：results/eval_report.json

import os
import json
import httpx
import chromadb
from openai import OpenAI

http_client = httpx.Client(trust_env=False)
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    http_client=http_client,
)
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:0.5b"


def embed_one(text):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def rag_answer(question, top_k=2):
    """跑一遍 RAG，返回 (回答, 检索到的片段)。top_k 和截断都调小以省显存。"""
    chroma = chromadb.PersistentClient(path="chroma_db")
    coll = chroma.get_or_create_collection("docs")
    qvec = embed_one(question)
    res = coll.query(query_embeddings=[qvec], n_results=top_k)
    docs = res["documents"][0]
    context = "\n\n---\n\n".join(d[:400] for d in docs)
    prompt = f"""请根据下面提供的资料回答问题。
只依据资料内容作答；资料里没有的信息就回答"资料中未提及"。

资料：
{context}

问题：{question}"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return resp.choices[0].message.content, docs


def judge(question, gold, pred):
    """用 LLM 当裁判：判断 AI 回答是否与标准答案一致"""
    prompt = f"""你是一个严格的评分员。判断下面 AI 的回答是否与标准答案一致（意思相同即可，不要求字面完全一致）。

标准答案：{gold}

AI 回答：{pred}

只输出一个词：一致 或 不一致"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return "一致" in resp.choices[0].message.content and "不一致" not in resp.choices[0].message.content


def main():
    os.makedirs("results", exist_ok=True)
    qa = json.load(open("data/testset.json", encoding="utf-8"))
    if not qa:
        print("测试集为空，请先跑 make_testset.py")
        return

    correct = 0
    results = []
    for i, item in enumerate(qa, 1):
        q = item["question"]
        gold = item["answer"]
        pred, docs = rag_answer(q)
        ok = judge(q, gold, pred)
        correct += 1 if ok else 0
        results.append({
            "question": q,
            "gold": gold,
            "pred": pred,
            "correct": bool(ok),
        })
        print(f"[{i}/{len(qa)}] {'PASS' if ok else 'FAIL'} | {q[:70]}")

    acc = correct / len(qa)
    print(f"\n=== 评测结果 ===")
    print(f"正确: {correct}/{len(qa)}")
    print(f"准确率: {acc:.1%}")

    with open("results/eval_report.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "correct": correct, "total": len(qa), "details": results},
                  f, ensure_ascii=False, indent=2)
    print("报告已保存到 results/eval_report.json")


if __name__ == "__main__":
    main()
