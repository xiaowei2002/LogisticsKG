"""按句子边界对中英文文本分块，单句超长时回退到字符/词切分。"""
import argparse
import re
import sys

# 中文与英文的句末标点，切分后保留标点本身
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；!?;])\s*|\n+")


def _split_sentences(text: str) -> list[str]:
    """将文本按中英文句末标点与换行切分为句子列表。"""
    parts = _SENTENCE_BOUNDARY.split(text)
    return [part.strip() for part in parts if part and part.strip()]


def _split_long_sentence(sentence: str, max_chunk_size: int) -> list[str]:
    """单句超过上限时，按词（含空格）或字符（纯中文）进一步切分。"""
    if not sentence:
        return []
    if len(sentence) <= max_chunk_size:
        return [sentence]

    # 含空格（英文等）按词切，纯中文按字符切
    if " " in sentence:
        units = sentence.split()
        sep = " "
    else:
        units = list(sentence)
        sep = ""

    chunks = []
    current = ""
    for unit in units:
        candidate = unit if not current else current + sep + unit
        if len(candidate) <= max_chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # 极端情况下单个 unit 本身仍超长，硬切
            if len(unit) > max_chunk_size:
                for i in range(0, len(unit), max_chunk_size):
                    chunks.append(unit[i : i + max_chunk_size])
                current = ""
            else:
                current = unit
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """将文本按句子边界分块，并保证每块不超过 max_chunk_size 字符。

    Args:
        text: 待分块的文本
        max_chunk_size: 每块的最大字符数

    Returns:
        分块后的文本列表
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chunk_size:
        return [text]

    sentences = _split_sentences(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 单句超长：先推出当前块，再按字符/词硬切该句
        if len(sentence) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.extend(_split_long_sentence(sentence, max_chunk_size))
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="按句子边界将长文本切分为小块，单句超长时按字符/词回退切分。"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="输入文本文件路径；缺省时从标准输入读取。",
    )
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        default=500,
        help="每块的最大字符数（默认 500）。",
    )
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    result_chunks = chunk_text(text, max_chunk_size=args.max_chunk_size)

    for i, chunk in enumerate(result_chunks, start=1):
        print(f"--- Chunk {i} (length {len(chunk)}): ---")
        print(chunk)
        print()


if __name__ == "__main__":
    main()
