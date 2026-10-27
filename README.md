RAG 文档问答系统（Qwen 技术报告）
基于本地大模型 + 向量检索，把 59 页 PDF 文档变成「会翻书回答」的问答助手，并配套自动评测。

网页演示

一、项目背景
直接问大模型，容易出现两个问题：答错（幻觉）、答不了私有文档。

本项目用 RAG（Retrieval-Augmented Generation，检索增强生成） 解决：先在你的文档里找到相关段落，再让大模型「照着资料回答」，做到有据可查、可溯源。

二、核心功能
功能	说明
💬 文档问答	提问 → 向量检索相关段落 → 大模型生成回答
🔍 来源溯源	每个回答附带参考片段及相似度分数
📊 自动评测	测试集 + LLM-as-judge，量化准确率
🗂 知识库管理	查看向量块数、模型信息、一键重建索引
🎛 参数调节	可调检索数量 top_k，观察召回与显存的权衡
三、技术栈
组件	选型
生成模型	Ollama 本地部署（qwen3:4b / qwen2.5:0.5b）
Embedding	nomic-embed-text（768 维）
向量库	ChromaDB
网页界面	Gradio
语言/库	Python 3.10+、openai、chromadb、pypdf、httpx
四、快速开始
bash


# 1. 安装 Ollama 并拉取模型
ollama pull qwen3:4b
ollama pull nomic-embed-text

# 2. 安装依赖
pip install -r requirements.txt

# 3. 建库：读 PDF → 切块 → 向量化 → 入库
python src/ingest_full.py

# 4. 启动网页演示
python app.py
# 浏览器打开 http://127.0.0.1:7860

# 5.（可选）命令行问答 / 评测
python demo.py
python src/eval.py
提示：本机显存 6GB，4B 模型长上下文会 OOM，可用 qwen2.5:0.5b 或 qwen3-4b-2k（2K 上下文版本）。

五、项目结构
代码块


rag-qa-demo/
├── data/
│   ├── document.pdf      # 测试文档：Qwen 技术报告（59 页，16.7 万字符）
│   └── testset.json      # 测试集
├── src/
│   ├── ingest_full.py    # 建库：读 + 切 + 向量化 + 入库
│   ├── query.py          # 命令行问答
│   ├── make_testset.py   # 自动生成测试集
│   └── eval.py           # 评测：LLM-as-judge
├── app.py                # Gradio 三 Tab 网页应用
├── demo.py               # 命令行交互演示
├── requirements.txt
└── README.md
六、评测结果
指标	数值
测试文档	Qwen 技术报告（59 页）
切分块数	371 块（chunk_size=500，overlap=50）
向量维度	768
测试集	4 条（自动生成）
模型	qwen2.5:0.5b（受显存限制）
准确率	25%（1/4）
结果分析：准确率偏低有两层原因——① 受 6GB 显存限制只能用 0.5B 小模型，推理能力有限；② 测试集由小模型自动生成，混入噪音（如与文档无关的题目）。两者共同拉低分数，是后续优化的重点。

七、设计说明（面试可讲）
为什么切块？ 大模型上下文有限，整本书塞不下；检索需要精准到“小段”，块太大噪音多、太小信息碎。本项目 chunk_size=500、overlap=50。

为什么要向量化？ 计算机不懂文字语义，但能算数字距离。把文本映射为向量后，“语义相近”变成“距离相近”，检索简化成最近邻搜索。

为什么 top_k=2？ 取太多上下文会超显存（实测 3 段 + 长上下文会 OOM），取太少可能漏掉答案。2 是显存与召回的折中。

为什么 trust_env=False？ 本机开着系统代理（127.0.0.1:7897），httpx 会自动读取代理导致连本地 Ollama 返回 502，必须禁用代理。

八、已知限制
6GB 显存跑不动 4B 模型长上下文（已提供 2K 上下文版本）
测试集为自动生成，含噪音，准确率偏保守
单文档、单轮问答，暂未支持多文档与对话历史
九、后续优化方向
 换 bge-m3 embedding 做 A/B 对比
 人工校验扩充测试集到 50 条
 增加 recall@k 检索命中率指标
 支持多文档上传与多轮对话
 部署到 Hugging Face Spaces（需改用在线 API）
