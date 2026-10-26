# RAG 文档问答 Demo（Qwen 技术报告）

一个完整的 **RAG（检索增强生成）** 项目：把一份 59 页的英文技术报告变成可问答的知识库，
提问时先检索相关段落、再交给本地大模型作答，并附带自动评测。

## 一、这是什么

给一本 PDF 文档，问它问题，它**先从文档里找相关资料，再照着资料回答**。

与直接问大模型的区别：回答有依据、可溯源、能回答模型没学过的私有文档。

## 二、环境要求

| 项 | 要求 |
|---|---|
| 系统 | Windows |
| Python | 3.10+ |
| Ollama | 已安装并运行（localhost:11434） |
| 显存 | 建议 8GB+；6GB 请用 `qwen2.5:0.5b` |

## 三、安装

```bash
# 1. 拉 Ollama 模型
ollama pull qwen2.5:0.5b        # 小模型，显存友好
ollama pull nomic-embed-text    # embedding 模型

# 2. 安装 Python 依赖
pip install -r requirements.txt
```

## 四、复现步骤（按顺序执行）

```bash
# ① 建库：读 PDF → 切块 → 向量化 → 存入 ChromaDB
python src/ingest_full.py
# 预期输出：
#   切分块数： 371
#   向量维度： 768
#   已入库块数： 371

# ② 演示：交互式问答
python demo.py
# 输入问题，得到回答 + 参考来源

# ③ 评测：自动生成测试集并打分
python src/eval.py
# 预期输出：准确率报告，并存到 results/eval_report.json
```

## 五、项目结构

```
rag-qa-demo/
├── data/
│   ├── document.pdf      # 测试文档：Qwen 技术报告（59 页）
│   └── testset.json      # 测试集（问题+标准答案）
├── src/
│   ├── ingest_full.py    # 建库：读+切+向量化+入库
│   ├── query.py          # 单次问答（命令行版）
│   ├── make_testset.py   # 自动生成测试集
│   └── eval.py           # 评测：LLM-as-judge
├── demo.py               # 交互式演示（推荐展示用）
├── requirements.txt
└── README.md
```

> `chroma_db/` 是运行后自动生成的向量库，可用 `ingest_full.py` 重建，不需要打包。

## 六、关键设计说明（面试可讲）

1. **为什么要切块？**
   - 大模型上下文有限，整本书塞不下
   - 检索要精准到“小段”，块太大噪音多、太小信息碎
   - 本项目的 `chunk_size=500`、`overlap=50`

2. **为什么要向量化？**
   - 计算机不懂文字语义，但能算数字距离
   - 意思相近 → 向量相近 → 检索变成“找最近的点”

3. **为什么 top_k=2？**
   - 取太多上下文会超显存（实测 3 段 + 长上下文会 OOM）
   - 取太少可能漏掉答案，2 是显存和召回的折中

4. **为什么要 trust_env=False？**
   - 本机开着系统代理（127.0.0.1:7897），httpx 会自动走代理
   - 导致连本地 Ollama 报 502，必须禁用代理读取

## 七、已知限制

- 6GB 显存跑不动 4B 模型的长上下文 → 已提供 `qwen3-4b-2k`（2K 上下文）方案
- 测试集由小模型自动生成，含噪音（如无关的“木塔”题），拉低准确率
- 中文文档可换 `bge-m3` 提升检索

## 八、可扩展方向

- [ ] 换 `bge-m3` embedding 做 A/B 对比
- [ ] 人工校验扩充测试集到 50 条
- [ ] 增加 `recall@k` 检索命中率指标
- [ ] 用 Gradio 做成网页版 demo
