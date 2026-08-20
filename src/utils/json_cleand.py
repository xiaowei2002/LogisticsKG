"""
清洗结构化 JSON，删除与知识图谱无关的章节。
"""
import argparse
import json
import re
from pathlib import Path

from loguru import logger

# 需要整节删除的标题模式（对去除空白后的标题做正则匹配）。
SKIP_PATTERNS: list[str] = [
    # 前置页 / 编写说明
    r"前言",
    r"序言",
    r"撰写目的",
    r"编写目的",
    r"本书的",
    r"读者群",
    r"新颖之处",
    r"编制说明",
    r"编委",
    r"编写委员会",
    # 标准固定章节
    r"^(?:\d+)?(?:适用)?范围$",
    r"规范[性型]?引用文件",
    r"引用标准",
    r"引用文件",
    # 章节末尾教学辅助内容
    r"本章小结",
    r"^小结$",
    r"复习思考题",
    r"思考题",
    r"思题",
    r"复习题",
    r"习题",
    r"学习目标",
    r"学习要求",
    r"学习重点",
    r"实训",
    r"仪器和设备",
    r"场地与设备",
    r"任务材料",
    r"任务分解",
    # 后置页
    r"参考文献",
    r"致谢",
    r"后记",
]

# 命中这些模式后，连同其后（直到下一个带编号的正文标题）的子项一起删除。
BLOCK_PATTERNS: list[str] = [
    r"实训项目",
]

# 正文开始标志：形如「2.1」「2.1.1」的编号标题。
_CONTENT_START = re.compile(r"^\s*\d+(?:\.\d+)+\s*")
# 形如「第X章」「第X节」的标题。
_CHAPTER_START = re.compile(r"^\s*第[一二三四五六七八九十百]+[章节]")
# 参考文献条目：行首编号 + 出版标记（出版社/学报/期刊）。
_REFERENCE_ENTRY = re.compile(r"^\s*\[?\d+\]?\s*[、.．)）:：]?\s*[^\n]*(?:出版社|学报|期刊)")


def _normalize(heading: str) -> str:
    """去除标题中所有空白，避免 OCR 把「思考题」切成「思 考 题」。"""
    return re.sub(r"\s+", "", heading or "")


def _matches_any(normalized: str, patterns: list[str]) -> bool:
    return any(re.search(p, normalized) for p in patterns)


def _is_content_start(normalized: str) -> bool:
    return bool(_CONTENT_START.match(normalized) or _CHAPTER_START.match(normalized))


def _strip_references(text: str) -> str:
    """截断正文末尾嵌入的参考文献列表（从首个参考条目或「参考文献」行开始）。"""
    lines = (text or "").split("\n")
    for i, line in enumerate(lines):
        if _REFERENCE_ENTRY.search(line) or _normalize(line) == "参考文献":
            return "\n".join(lines[:i])
    return text


def clean_doc(doc: dict) -> dict:
    """清洗结构化文档，返回只含正文的副本（不修改入参）。

    Args:
        doc: pdf2json 产出的 {"title": ..., "sections": [...]} 结构

    Returns:
        清洗后的文档
    """
    sections = doc.get("sections", [])
    kept: list[dict] = []
    in_block = False
    removed = 0

    for sec in sections:
        norm = _normalize(sec.get("heading", ""))

        if in_block:
            # 在实训等块内：遇到真正的正文标题（带编号）且不属于待删关键词才退出块
            if _is_content_start(norm) and not _matches_any(norm, SKIP_PATTERNS):
                in_block = False
                kept.append(sec)
            else:
                removed += 1
            continue

        if _matches_any(norm, SKIP_PATTERNS):
            removed += 1
            if _matches_any(norm, BLOCK_PATTERNS):
                in_block = True
        else:
            kept.append(sec)

    # 裁剪各 section 正文末尾嵌入的参考文献，剔除因此变为空的 section。
    final: list[dict] = []
    for sec in kept:
        content = _strip_references(sec.get("content", ""))
        if content.strip():
            final.append({**sec, "content": content})
        else:
            removed += 1

    result = {"title": doc.get("title", ""), "sections": final}
    if removed:
        logger.info("清洗删除 {} 个 section（保留 {}）", removed, len(final))
    return result


def clean_file(path: str | Path, output_path: str | Path | None = None) -> dict:
    """读取单个 JSON，清洗后写回（缺省覆盖原文件），返回清洗后的文档。"""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    cleaned = clean_doc(doc)
    target = Path(output_path) if output_path else path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="清洗结构化 JSON，删除与知识图谱无关的章节")
    parser.add_argument("--input", type=str, default="output", help="JSON 文件或目录（默认 output）")
    parser.add_argument("--output", type=str, default=None, help="输出目录/文件，缺省覆盖原文件")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        files = [
            f for f in sorted(input_path.glob("*.json"))
            if not f.stem.endswith("_graph") and f.name != "merged_graph.json"
        ]
        for f in files:
            out = Path(args.output) / f.name if args.output else None
            logger.info("清洗: {}", f.name)
            clean_file(f, out)
    else:
        clean_file(input_path, args.output)


if __name__ == "__main__":
    main()
