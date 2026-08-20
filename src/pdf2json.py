"""PDF 结构化解析（Qwen-VL 视觉大模型版本）。

将 PDF 逐页渲染为图片，交给视觉大模型识别版式、OCR 文字、解析表格并输出
结构化 JSON。相比 pymupdf/pdfplumber 的纯文本提取，能正确处理：

- 扫描版书籍（无文本层，靠 OCR）
- 表格（输出为 markdown 表格）
- 封面 / 目录 / 页眉页脚 / 页码噪声（prompt 中直接跳过）

用法（在项目根目录）：
    python -m src.pdf2json --input pdfs --output output
    python -m src.pdf2json --input pdfs --estimate-only   # 只估算花费，不调用模型
"""
import argparse
import base64
import json
import os
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from loguru import logger

try:
    from src.client import Client
except ImportError:  # 兼容 `python src/pdf2json.py` 直接运行
    from client import Client

load_dotenv()

# 视觉模型名：由 .env 中的 OPENAI_VL_MODEL_NAME 控制，缺省 qwen3-vl-plus
DEFAULT_VL_MODEL = os.getenv("OPENAI_VL_MODEL_NAME", "qwen3-vl-plus")

# ============ 成本估算常量（单位：元 / 百万 token） ============
# 以下是 qwen-vl-max 的粗略价格，请按你实际使用的模型与账单调整。
# 注意：图像 token 化数量随渲染分辨率（dpi）与版面密度变化，此处为经验估算，
# 仅供参考，不构成精确报价。
PRICE_INPUT_PER_MTOKEN = 3.0     # 输入（含图像 token 化）约 ¥3 / 百万 token
PRICE_OUTPUT_PER_MTOKEN = 9.0    # 输出约 ¥9 / 百万 token
TOKENS_PER_PAGE_IMAGE = 1500     # 单页图像（约 150 dpi）token 化后的估算输入 token
PROMPT_TOKENS_PER_PAGE = 300     # 每页 system/user 文本 prompt 的固定开销
TOKENS_PER_PAGE_OUTPUT = 800     # 单页结构化 JSON 的估算输出 token


def estimate_cost(page_count: int) -> dict:
    """按页数估算 token 用量与花费（元）。"""
    input_tokens = page_count * (TOKENS_PER_PAGE_IMAGE + PROMPT_TOKENS_PER_PAGE)
    output_tokens = page_count * TOKENS_PER_PAGE_OUTPUT
    cost = (
        input_tokens / 1_000_000 * PRICE_INPUT_PER_MTOKEN
        + output_tokens / 1_000_000 * PRICE_OUTPUT_PER_MTOKEN
    )
    return {
        "pages": page_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_yuan": round(cost, 2),
    }


STRUCT_PROMPT = """你是一个文档结构化解析专家。这是某 PDF 文档的第 {page_no} 页（图片）。请识别页面内容并输出结构化 JSON。

先判断本页类型：
- cover：封面、书名页、版权页
- toc：目录页
- frontmatter：前置说明页，如前言、序、引言、编委/编写委员会名单、编制说明等
- backmatter：后置说明页，如参考文献、索引、致谢、后记、封底等
- empty：空白页，或仅含页眉、页脚、页码等版式元素
- content：含实质正文

规则：
1. 若类型为 cover / toc / frontmatter / backmatter / empty，输出 {{"skip": true, "type": "cover|toc|frontmatter|backmatter|empty", "title": "", "blocks": []}}，不要提取任何正文。
2. 若为 content：跳过页眉、页脚、页码；把正文拆分为若干 block：
   - heading：章节标题，附 level（1=章, 2=节, 3=条, 4=款）
   - text：正文段落，保留编号、定义、数据等关键信息，不要改写
   - table：表格，用 markdown 表格表示（| 表头 | ... | 与 |---| 分隔行）
   注意：含规范性/资料性技术内容的附录（appendix）应视为 content 并正常提取，不要当作 backmatter 跳过。
3. title 字段：仅当本页明确出现文档主标题（封面或首页大标题）时才填写，否则留空字符串 ""。
4. 只输出 JSON，不要任何解释文字。

输出格式示例：
{{"skip": false, "type": "content", "title": "", "blocks": [{{"type": "heading", "level": 2, "text": "3.1 物流"}}, {{"type": "text", "content": "根据实际需要……"}}]}}"""


def _parse_json(text: str) -> dict:
    """剥离可能的 markdown 代码围栏后解析 JSON，失败时降级为整段文本。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "skip": False,
            "type": "content",
            "title": "",
            "blocks": [{"type": "text", "content": text}],
        }


def _parse_page(client: Client, png_bytes: bytes, page_no: int) -> dict:
    """调用视觉模型解析单页图片。"""
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    prompt = STRUCT_PROMPT.format(page_no=page_no)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    raw = client.chat(messages=messages, stream=False, temperature=0.0)
    return _parse_json(raw)


def _assemble(title: str, page_records: list[dict]) -> dict:
    """把逐页 block 合并成扁平的 section 列表（heading + content）。"""
    sections: list[dict] = []
    current: dict | None = None

    for record in page_records:
        for block in record.get("blocks", []):
            btype = block.get("type")
            if btype == "heading":
                if current and current["content"].strip():
                    sections.append(current)
                current = {
                    "heading": block.get("text", ""),
                    "level": block.get("level", 1),
                    "content": "",
                }
            elif btype in ("text", "table"):
                if current is None:
                    current = {"heading": "", "level": 0, "content": ""}
                current["content"] += block.get("content", "") + "\n"

    if current and current["content"].strip():
        sections.append(current)

    return {"title": title, "sections": sections}


def flatten_to_text(doc: dict) -> str:
    """把结构化文档展平为纯文本，供知识图谱抽取使用。"""
    parts = [doc.get("title", "")]
    for section in doc.get("sections", []):
        heading = section.get("heading", "")
        content = (section.get("content") or "").strip()
        if heading:
            parts.append(heading)
        if content:
            parts.append(content)
    return "\n".join(p for p in parts if p)


def process_pdf(
    pdf_path: str,
    output_dir: str = "output",
    dpi: int = 150,
    vl_model: str | None = None,
    estimate_only: bool = False,
    force: bool = False,
) -> dict | None:
    """处理单个 PDF：渲染→逐页 VLM 解析→组装→写 JSON。

    支持断点续跑：每页完成后追加写入 <stem>.pages.jsonl 检查点，并增量写出
    最终 JSON；中断后重启会跳过已完成的页继续，全部完成后清理检查点。
    """
    doc = pymupdf.open(pdf_path)
    page_count = len(doc)
    est = estimate_cost(page_count)
    stem = Path(pdf_path).stem
    out_path = Path(output_dir) / f"{stem}.json"
    checkpoint = Path(output_dir) / f"{stem}.pages.jsonl"

    logger.info(
        "文件: {}（{} 页）｜预估：输入 {:,} tok / 输出 {:,} tok，约 ¥{}",
        Path(pdf_path).name,
        page_count,
        est["input_tokens"],
        est["output_tokens"],
        est["cost_yuan"],
    )

    # 最终 JSON 已存在、无残留检查点、且非 force → 视为已完成，直接跳过
    if out_path.exists() and not checkpoint.exists() and not force:
        logger.info("已存在结构化 JSON，跳过解析: {}", out_path.name)
        doc.close()
        return json.loads(out_path.read_text(encoding="utf-8"))

    if estimate_only:
        doc.close()
        return est

    # 从检查点恢复已完成的页；force 时丢弃旧检查点重新开始
    records: dict[int, dict] = {}
    if checkpoint.exists():
        if force:
            checkpoint.unlink()
        else:
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                records[rec.get("page")] = rec
            logger.info("从检查点恢复 {} 页", len(records))

    client = Client(model_name=vl_model or DEFAULT_VL_MODEL)

    def _write_result() -> dict:
        """用当前已完成的页组装并写出最终 JSON（增量保存，中断后可继续）。"""
        ordered = [records[i + 1] for i in range(page_count) if (i + 1) in records]
        title = next((r.get("title") or "" for r in ordered if r.get("title")), "")
        result = _assemble(title, ordered)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return result

    for page_no in range(page_count):
        pageno = page_no + 1
        if pageno in records:
            continue
        pix = doc[page_no].get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
        record = _parse_page(client, png, pageno)
        record["page"] = pageno
        records[pageno] = record
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        _write_result()
        status = "跳过" if record.get("skip") else "完成"
        logger.info("  第 {}/{} 页 {}", pageno, page_count, status)
    doc.close()

    result = _write_result()
    checkpoint.unlink(missing_ok=True)
    logger.info("  已保存至 {}", out_path)
    return result


def process_folder(
    pdf_folder: str = "pdfs",
    output_dir: str = "output",
    dpi: int = 150,
    vl_model: str | None = None,
    estimate_only: bool = False,
):
    pdf_dir = Path(pdf_folder)
    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    total_pages = 0
    for pdf_path in pdf_files:
        total_pages += len(pymupdf.open(str(pdf_path)))

    est = estimate_cost(total_pages)
    logger.info(
        "共发现 {} 个 PDF、{} 页，预估总花费约 ¥{}（输入 {:,} tok / 输出 {:,} tok）",
        len(pdf_files),
        total_pages,
        est["cost_yuan"],
        est["input_tokens"],
        est["output_tokens"],
    )
    if estimate_only:
        return

    for pdf_path in pdf_files:
        process_pdf(str(pdf_path), output_dir, dpi, vl_model)


def main():
    parser = argparse.ArgumentParser(description="用 Qwen-VL 将 PDF 结构化解析为 JSON")
    parser.add_argument("--input", type=str, default="pdfs", help="PDF 文件或目录（默认 pdfs）")
    parser.add_argument("--output", type=str, default="output", help="输出目录（默认 output）")
    parser.add_argument("--dpi", type=int, default=150, help="页面渲染分辨率（默认 150）")
    parser.add_argument("--model", type=str, default=None, help="视觉模型名，缺省读取 OPENAI_VL_MODEL_NAME")
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="只统计页数并估算花费，不调用模型",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        process_folder(str(input_path), args.output, args.dpi, args.model, args.estimate_only)
    else:
        process_pdf(str(input_path), args.output, args.dpi, args.model, args.estimate_only)


if __name__ == "__main__":
    main()
