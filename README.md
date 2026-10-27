# RAG 文档问答系统（Qwen 技术报告）

一个基于 **RAG（检索增强生成）** 的文档问答系统：把 PDF 文档切块、向量化后存入向量库，
提问时先检索相关段落，再交给本地大模型作答，并提供**自动评测**。

## 功能

- PDF 文档加载与切分（chunk）
- 文本向量化（Ollama `nomic-embed-text`）
- 向量存储与相似检索（ChromaDB）
- 检索 + 生成问答
- 自动评测（LLM-as-judge，输出准确率）

## 技术栈

| 组件 | 选型 |
|---|---|
| 大模型 | Ollama（`qwen3:4b` / `qwen2.5:0.5b`） |
| Embedding | `nomic-embed-text`（768 维） |
| 向量库 | ChromaDB |
| 语言/库 | Python 3.14 + `openai` + `chromadb` + `pypdf` |

> ⚠️ 本机 GPU 显存 6GB，4B 模型长上下文会 OOM；评测时使用小模型或小上下文版本。

## 快速开始

```bash
# 1. 安装 Ollama 并拉模型
ollama pull qwen3:4b
ollama pull nomic-embed-text

# 2. 安装依赖
pip install -r requirements.txt

# 3. 把 PDF 放到 data/document.pdf

# 4. 切分 + 向量化入库
python src/ingest_full.py

# 5. 提问
python src/query.py

# 6. 生成测试集 + 评测
python src/make_testset.py
python src/eval.py
```

## 项目结构

```
rag-qa-project/
├── data/
│   ├── document.pdf      # 测试文档（Qwen 技术报告，59 页）
│   └── testset.json      # 自动生成的测试集
├── src/
│   ├── ingest.py         # 第3课：读 PDF + 切分
│   ├── ingest_full.py    # 第3+4课：读 + 切 + 向量化 + 入库
│   ├── query.py          # 第5课：检索 + 生成
│   ├── make_testset.py   # 第6课：自动生成测试集
│   └── eval.py           # 第6课：评测
├── results/
│   └── eval_report.json  # 评测报告
├── chroma_db/            # 向量库（自动生成）
├── test_api.py           # 环境验证
├── requirements.txt
└── README.md
```

## 评测结果

- 测试文档：Qwen Technical Report（59 页，166,662 字符，切分 371 块）
- 测试集：自动生成 4 条问答
- 模型：`qwen2.5:0.5b`（受显存限制）
- **准确率：25%**

## 已知限制

1. 显存 6GB 时 4B 模型长上下文会 OOM → 已提供 `qwen3-4b-2k`（2K 上下文版本）
2. 测试集由小模型自动生成，质量问题（如混入无关内容）会影响准确率
3. 中文文档可用 `bge-m3` 提升检索效果

## 后续优化方向

- [ ] 更换 embedding 模型（nomic → bge-m3）对比效果
- [ ] 调整 chunk_size / top_k 做对比实验
- [ ] 扩充测试集（人工校验 + 50 条以上）
- [ ] 加检索命中率（recall@k）指标
