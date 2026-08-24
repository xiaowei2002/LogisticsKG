# LogisticsKG

## 环境准备

使用 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
git clone https://github.com/xiaowei2002/LogisticsKG.git
cd LogisticsKG
uv sync
```

复制 `.env` 并填入配置：

```ini
OPENAI_API_KEY=your_key
OPENAI_MODEL_NAME=qwen3.8-max        # 文本模型（实体/关系抽取、去重）
OPENAI_VL_MODEL_NAME=qwen3-vl-plus   # 视觉模型（PDF 结构化解析）
OPENAI_BASE_URL=https://.../compatible-mode/v1
```
## 数据准备
把 PDF 放到以下目录（按书籍 / 标准分类）：
```
pdfs/
├── 书籍/
│   └── 汽车物流基础.pdf
└── 标准/
    └── 物流术语_GBT+18354-2021.pdf
```

## 运行

```bash
python -m src.main                      # 全流程：PDF → JSON → 图谱
python -m src.main --skip-pdf           # 跳过 PDF 解析，直接用已有 JSON 建图谱
python -m src.main --skip-pdf           # 命中缓存时零 LLM 调用，直接可视化
python -m src.main --force              # 忽略缓存，重新生成图谱
python -m src.main --estimate-only      # 只估算 PDF→JSON 阶段的花费
```

## 输出
```
output/
├── 物流术语_GBT+18354-2021.json           # Stage 1：结构化 JSON
├── 物流术语_GBT+18354-2021_graph.json     # Stage 2：该文档的知识图谱
├── 物流术语_GBT+18354-2021_graph.html     # 该图谱的可视化
└── merged_graph.json / merged_graph.html  # 
```

## RAG

RAG 问答系统（Gradio Web UI）支持「文档 RAG + GraphRAG」融合检索

### 1. 准备语料

- **文档**：将 PDF 放入 `pdfs/`（默认读取 `RAG/config.yaml` 中的 `rag.docs_dir`，与图谱构建共用）
- **知识图谱**：GraphRAG 依赖 `output/merged_graph.json`，先运行上文「运行」的 `python -m src.main` 构建

### 2. 配置 LLM

编辑 `RAG/config.yaml` 的 `llm` 节（默认本地 Ollama）：

```yaml
llm:
  ollama:
    model: qwen3:1.7b
    api_key: ollama
    base_url: http://localhost:11434/v1
```

- 用 Ollama：先 `ollama pull qwen3:1.7b` 并启动 `ollama serve`
- 用云端 OpenAI 兼容接口：把 `base_url` / `api_key` / `model` 改成对应服务（与 `.env` 一致）

### 3. 启动

```bash
python -m RAG.app               # 启动 Web UI（默认 http://127.0.0.1:7860）
python -m RAG.app --build       # 启动前预构建 RAG/GraphRAG 索引（首次较慢，之后秒级加载）
python -m RAG.app --no-browser  # 不自动打开浏览器
```

浏览器打开后，右侧切换「使用 RAG / 不使用 RAG」，即可对比同一问题的两种回答。

### 4. 可选参数

在 `RAG/config.yaml` 中调整：`rag.chunk_size` / `rag.top_k` / `rag.bm25_weight`、`graphrag.entity_top_k` / `graphrag.context_depth` / `graphrag.max_triples` 等。

