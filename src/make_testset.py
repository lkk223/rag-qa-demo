# 第 6 课（1/2）：从文档自动生成问答测试集
# 依赖：chromadb openai httpx
# 运行：python src/make_testset.py   （在项目根目录下）
# 输出：data/testset.json

import httpx
import chromadb
import json
import random
import re
from openai import OpenAI

http_client = httpx.Client(trust_env=False)
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    http_client=http_client,
)
CHAT_MODEL = "qwen3:4b"
N_QUESTIONS = 10          # 生成多少条测试题


def ask_llm(prompt, max_tokens=2000):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def extract_json(text):
    """从模型输出里抠出 JSON（去掉思考过程/多余文字）"""
    if not text:
        return None
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # 找所有 {...} 片段，取第一个能解析的
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            return json.loads(m.group())
        except Exception:
            continue
    return None


def main():
    chroma = chromadb.PersistentClient(path="chroma_db")
    coll = chroma.get_or_create_collection("docs")
    data = coll.get()
    docs = data["documents"]

    random.seed(42)
    sample = random.sample(docs, min(N_QUESTIONS * 2, len(docs)))

    qa = []
    for chunk in sample:
        if len(qa) >= N_QUESTIONS:
            break
        prompt = f"""请根据下面这段资料，出一道能被该资料回答的问题，并给出标准答案。
直接输出下面的 JSON，不要解释、不要输出思考过程：
{{"question": "问题", "answer": "标准答案"}}

资料：
{chunk[:1200]}"""
        raw = ask_llm(prompt)
        item = extract_json(raw)
        if item and "question" in item and "answer" in item:
            qa.append({
                "question": str(item["question"]).strip(),
                "answer": str(item["answer"]).strip(),
            })
            print(f"  已生成 {len(qa)}/{N_QUESTIONS}")

    with open("data/testset.json", "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)
    print(f"完成：生成 {len(qa)} 条测试题")


if __name__ == "__main__":
    main()
