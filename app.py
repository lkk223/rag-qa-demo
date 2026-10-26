# -*- coding: utf-8 -*-
"""RAG 文档问答 —— Gradio 多 Tab 演示应用

三个功能面板：
  1. 问答    —— 提问、查看回答与参考来源、调节检索数量
  2. 评测    —— 一键跑测试集，得到准确率与逐条结果
  3. 知识库  —— 查看向量库状态、重建索引

运行：python app.py
"""

import os

# 必须在 import gradio/requests 之前清除代理，
# 否则 Gradio 启动时的本地健康检查会走系统代理导致 502
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import json
import time
import httpx
import chromadb
from openai import OpenAI
import gradio as gr

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    http_client=httpx.Client(trust_env=False),
)
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:0.5b"
DB_PATH = "chroma_db"
COLL_NAME = "docs"


def get_coll():
    chroma = chromadb.PersistentClient(path=DB_PATH)
    return chroma.get_or_create_collection(COLL_NAME)


def embed(text):
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


def chat(prompt, max_tokens=600):
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


# ---------------- Tab 1: 问答 ----------------
def ask(question, top_k):
    if not question or not question.strip():
        return "请输入问题", "", ""
    coll = get_coll()
    if coll.count() == 0:
        return "向量库为空，请先到「知识库」页重建索引", "", ""

    t0 = time.time()
    qvec = embed(question)
    res = coll.query(query_embeddings=[qvec], n_results=int(top_k))
    docs = res["documents"][0]
    dists = res["distances"][0]

    context = "\n\n---\n\n".join(d[:400] for d in docs)
    prompt = (
        "请根据下面提供的资料回答问题。只依据资料内容作答；"
        "资料里没有的信息就回答“资料中未提及”。\n\n"
        f"资料：\n{context}\n\n问题：{question}"
    )
    answer = chat(prompt)
    elapsed = time.time() - t0

    src_lines = []
    for i, (d, dist) in enumerate(zip(docs, dists), 1):
        sim = max(0.0, 1 - dist)
        src_lines.append(f"[{i}] 相似度 {sim:.3f}\n{d[:220]}...")
    return answer, "\n\n".join(src_lines), f"耗时 {elapsed:.1f}s | top_k={int(top_k)}"


# ---------------- Tab 2: 评测 ----------------
def run_eval():
    path = os.path.join("data", "testset.json")
    if not os.path.exists(path):
        return "未找到测试集 data/testset.json", []
    qa = json.load(open(path, encoding="utf-8"))
    if not qa:
        return "测试集为空", []

    rows = []
    correct = 0
    for item in qa:
        q, gold = item["question"], item["answer"]
        pred, _, _ = ask(q, 2)
        judge_prompt = (
            "你是一个严格的评分员。判断 AI 回答是否与标准答案一致（意思相同即可）。\n\n"
            f"标准答案：{gold}\n\nAI 回答：{pred}\n\n只输出一个词：一致 或 不一致"
        )
        verdict = chat(judge_prompt, 400)
        ok = "一致" in verdict and "不一致" not in verdict
        correct += 1 if ok else 0
        rows.append([q[:50], "PASS" if ok else "FAIL"])

    acc = correct / len(qa)
    summary = f"准确率：{correct}/{len(qa)} = {acc:.1%}"
    return summary, rows


# ---------------- Tab 3: 知识库 ----------------
def kb_status():
    coll = get_coll()
    return (
        f"文档：data/document.pdf\n"
        f"向量块数：{coll.count()}\n"
        f"Embedding 模型：{EMBED_MODEL}（768 维）\n"
        f"生成模型：{CHAT_MODEL}\n"
        f"向量库路径：{os.path.abspath(DB_PATH)}"
    )


def rebuild():
    import subprocess
    import sys
    try:
        out = subprocess.run(
            [sys.executable, "src/ingest_full.py"],
            capture_output=True, text=True, timeout=600, encoding="utf-8", errors="replace",
        )
        return (out.stdout or out.stderr)[-1500:] or "执行完成（无输出）"
    except Exception as e:
        return f"重建失败：{e}"


# ---------------- 界面 ----------------
with gr.Blocks(title="RAG 文档问答系统") as demo:
    gr.Markdown(
        "# RAG 文档问答系统\n"
        "知识库：Qwen 技术报告（59 页）｜检索增强生成 + 自动评测"
    )

    with gr.Tabs():
        # Tab 1
        with gr.Tab("💬 问答"):
            with gr.Row():
                q_in = gr.Textbox(label="你的问题", placeholder="例如：Qwen 有哪些模型规模？", lines=2, scale=4)
                top_k = gr.Slider(1, 5, value=2, step=1, label="检索数量 top_k", scale=1)
            ask_btn = gr.Button("提问", variant="primary")
            a_out = gr.Textbox(label="回答", lines=6)
            s_out = gr.Textbox(label="参考来源（含相似度）", lines=8)
            t_out = gr.Textbox(label="信息", lines=1)
            ask_btn.click(ask, [q_in, top_k], [a_out, s_out, t_out])
            q_in.submit(ask, [q_in, top_k], [a_out, s_out, t_out])

        # Tab 2
        with gr.Tab("📊 评测"):
            gr.Markdown("对测试集跑一遍 RAG，用 LLM 当裁判判分，输出准确率。")
            eval_btn = gr.Button("运行评测", variant="primary")
            eval_sum = gr.Textbox(label="结果", lines=2)
            eval_table = gr.Dataframe(headers=["问题", "判定"], label="逐条结果", wrap=True)
            eval_btn.click(run_eval, None, [eval_sum, eval_table])

        # Tab 3
        with gr.Tab("🗂 知识库"):
            gr.Markdown("查看向量库状态，或从 PDF 重新构建索引。")
            kb_btn = gr.Button("刷新状态")
            kb_out = gr.Textbox(label="知识库状态", lines=7)
            rebuild_btn = gr.Button("重建索引（读 PDF → 切块 → 向量化 → 入库）")
            rebuild_out = gr.Textbox(label="重建日志", lines=10)
            kb_btn.click(kb_status, None, kb_out)
            rebuild_btn.click(rebuild, None, rebuild_out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
