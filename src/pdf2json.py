"""
PDF结构化解析JSON
"""
import json
import sys
from pathlib import Path
import pymupdf
from client import Client

client = Client()


# ---------- 1. PDF 文本提取 ----------
def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """逐页提取 PDF 文本，返回 [{page, text}, ...]"""
    doc = pymupdf.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


# ---------- 2. 大模型结构化解析 ----------
STRUCT_PROMPT = """你是一个文档结构化解析专家。请分析以下从 PDF 提取的文本，识别文档的逻辑结构。

要求：
1. 只输出 JSON，不要任何解释文字
2. 格式：{{"title": "文档标题", "sections": [{{"heading": "章节标题", "level": 1, "content": "正文内容", "subsections": [...]}}]}}
3. level 从 1 开始：1=章, 2=节, 3=条, 4=款
4. 保留原始关键信息（编号、定义、数据等），不要改写
5. 避免将页码标记（如 [第X页]）当作正文标题

文本内容:
{text}

请输出 JSON:"""


def parse_structure(text: str) -> dict:
    """调用大模型识别文档结构"""
    prompt = STRUCT_PROMPT.format(text=text[:12000])
    raw = client.chat(
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3]
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "JSON解析失败", "raw": raw}


# ---------- 3. 主流程 ----------
def process_pdf(pdf_path: str, output_dir: str = "output"):
    """处理单个 PDF，输出结构化 JSON"""
    print(f"处理: {pdf_path}")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print("未提取到文本")
        return

    full_text = "\n\n".join(f"[第{p['page']}页] {p['text']}" for p in pages)
    result = parse_structure(full_text)

    out_path = Path(output_dir) / (Path(pdf_path).stem + ".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  已保存至 {out_path}")


def process_folder(pdf_folder: str = "pdfs", output_dir: str = "output"):
    pdf_dir = Path(pdf_folder)
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    print(f"共发现 {len(pdf_files)} 个 PDF 文件")
    for pdf_path in sorted(pdf_files):
        process_pdf(str(pdf_path), output_dir)


if __name__ == "__main__":
    process_folder(sys.argv[1] if len(sys.argv) > 1 else "pdfs")
