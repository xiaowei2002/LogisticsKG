# LogisticsKG

物流领域知识图谱构建：把 PDF 格式的物流书籍与国家标准，自动转换为结构化知识图谱。

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
```

## 输出
```
output/
├── 物流术语_GBT+18354-2021.json           # Stage 1：结构化 JSON
├── 物流术语_GBT+18354-2021_graph.json     # Stage 2：该文档的知识图谱
├── 物流术语_GBT+18354-2021_graph.html     # 该图谱的可视化
└── merged_graph.json / merged_graph.html  # 