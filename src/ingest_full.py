# 第 3+4 课合并参考实现：读 PDF → 切分 → 向量化 → 存入 ChromaDB
# 依赖：pip install pypdf chromadb openai
# 运行：python src/ingest_full.py   （在项目根目录下）

import os
import httpx
import chromadb
from pypdf import PdfReader
from openai import OpenAI

# 关键：禁止 httpx 读取系统代理（你电脑开着 127.0.0.1:7897 代理，会干扰本地请求）
http_client = httpx.Client(trust_env=False)
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    http_client=http_client,
)
EMBED_MODEL = "nomic-embed-text"


def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def embed(texts, batch_size=50):
    """分批把文本转成向量（一次性传太多会让 Ollama 内部崩溃）"""
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_emb.extend([d.embedding for d in resp.data])
        print(f"  embedding 进度：{min(i + batch_size, len(texts))}/{len(texts)}")
    return all_emb


def main():
    pdf_path = os.path.join("data", "document.pdf")
    if not os.path.exists(pdf_path):
        print("请先把测试 PDF 放到 data/document.pdf")
        return

    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    print("切分块数：", len(chunks))

    embeddings = embed(chunks)
    print("向量维度：", len(embeddings[0]))

    # 向量库持久化到本地 chroma_db 目录
    chroma = chromadb.PersistentClient(path="chroma_db")
    coll = chroma.get_or_create_collection("docs")

    # 清空旧数据，避免重复添加
    try:
        coll.delete(where={})
    except Exception:
        pass

    coll.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": "document.pdf"} for _ in chunks],
    )
    print("已入库块数：", coll.count())


if __name__ == "__main__":
    main()
