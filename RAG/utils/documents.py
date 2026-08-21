"""文档加载与切块：PDF → langchain Document 列表。

- load_pdfs: 用 pymupdf 逐页抽取文本，每页生成一个 Document（带 source/page 元数据）
- split_documents: 按句子边界分块，支持块间重叠（无需 langchain-text-splitters 依赖）
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

# 中英文句末标点 + 换行，切分后保留标点
_SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；!?;])|(?<=\n)")


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_BOUNDARY.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """按句子边界切块，相邻块重叠约 overlap 字符（以句子对齐）。"""
    text = (text or "").strip()
    if not text:
        return []
    sentences = _split_sentences(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        if current_len + len(sentence) + 1 <= chunk_size:
            current.append(sentence)
            current_len += len(sentence) + 1
            continue
        # 当前块已满：落块，并取尾部句子作为下一块的 overlap 前缀
        if current:
            chunks.append(" ".join(current))
        tail: List[str] = []
        tail_len = 0
        for s in reversed(current):
            if tail_len + len(s) + 1 > overlap:
                break
            tail.insert(0, s)
            tail_len += len(s) + 1
        current = tail
        current_len = tail_len
        # 单句超长：直接硬切
        if len(sentence) > chunk_size:
            for i in range(0, len(sentence), chunk_size):
                chunks.append(sentence[i : i + chunk_size])
            current, current_len = [], 0
        else:
            current.append(sentence)
            current_len += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def load_pdfs(docs_dir: str | Path) -> List[Document]:
    """递归加载目录下所有 PDF，每页一个 Document。"""
    import pymupdf  # fitz

    docs: List[Document] = []
    root = Path(docs_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"语料目录不存在: {root}")
    for pdf_path in sorted(root.rglob("*.pdf")):
        try:
            with pymupdf.open(pdf_path) as pdf:
                for page_no in range(len(pdf)):
                    text = pdf[page_no].get_text().strip()
                    if not text:
                        continue
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={"source": pdf_path.name, "page": page_no + 1},
                        )
                    )
        except Exception as exc:  # 单个 PDF 损坏不影响整体
            print(f"[warn] 跳过无法解析的 PDF: {pdf_path} ({exc})")
    return docs


def split_documents(
    docs: Iterable[Document], chunk_size: int = 800, overlap: int = 120
) -> List[Document]:
    """把每页文档按句子边界切块，元数据保留并追加 chunk 序号。"""
    out: List[Document] = []
    for doc in docs:
        for idx, chunk in enumerate(split_text(doc.page_content, chunk_size, overlap)):
            out.append(
                Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": idx},
                )
            )
    return out
