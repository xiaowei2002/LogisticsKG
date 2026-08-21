"""RAG 引擎：基于 langchain 的文档检索增强问答。

流程：PDF 语料 → 句子切块 → 本地嵌入 → 混合检索（语义 + BM25）
      → LCEL 链（prompt | chat_model | StrOutputParser）流式回答。
索引（切块 + 向量）落盘到 index_dir，二次启动秒级加载。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from RAG.utils.chat_model import ChatOpenAICompat
from RAG.utils.documents import load_pdfs, split_documents
from RAG.utils.embeddings import get_embedder

# 本包根目录（RAG/），用于把配置里的相对路径解析为绝对路径
RAG_DIR = Path(__file__).resolve().parents[1]

_SYSTEM_PROMPT = """你是物流领域知识问答助手，请严格依据【参考资料】回答用户问题。
要求：
1. 只使用参考资料中的信息作答，不要引入外部知识，不要编造；
2. 若参考资料不足以回答问题，请明确说明"参考资料中未找到相关信息"；
3. 用中文回答，条理清晰、简洁准确。"""


def _char_bigrams(text: str) -> List[str]:
    """中文友好的词法切分：字符二元组（无分词依赖，离线可用）。"""
    text = (text or "").lower().strip()
    if len(text) < 2:
        return [text] if text else []
    return [text[i : i + 2] for i in range(len(text) - 1)]


class HybridRetriever(BaseRetriever):
    """语义向量 + BM25 混合检索器（langchain BaseRetriever 子类）。

    最终分数 = (1 - bm25_weight) * 语义余弦相似度 + bm25_weight * 归一化 BM25 分数。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embedding: Any = Field(description="langchain Embeddings 实例")
    documents: List[Document] = Field(description="全部切块文档")
    vectors: Any = Field(exclude=True, description="文档向量矩阵 (n, dim)")
    bm25: Any = Field(exclude=True, description="BM25Okapi 索引")
    top_k: int = 4
    bm25_weight: float = 0.3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> List[Document]:
        return [doc for doc, _ in self.search(query, k=self.top_k)]

    def search(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """返回 [(文档, 融合分数)]，按分数降序，供问答与来源展示共用。"""
        k = k or self.top_k
        query_vec = np.asarray(self.embedding.embed_query(query), dtype=np.float32)
        semantic = self.vectors @ query_vec  # 向量已归一化，点积即余弦相似度

        bm25_raw = np.asarray(self.bm25.get_scores(_char_bigrams(query)), dtype=np.float64)
        bm25_max = float(bm25_raw.max()) if bm25_raw.size else 0.0
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else np.zeros_like(bm25_raw)

        fused = (1 - self.bm25_weight) * semantic + self.bm25_weight * bm25_norm
        order = np.argsort(fused)[::-1][:k]
        return [(self.documents[i], float(fused[i])) for i in order]


class RAGEngine:
    """文档 RAG 引擎：构建/加载索引并流式回答。"""

    def __init__(self, cfg: Any, chat_model: Optional[ChatOpenAICompat] = None):
        """
        Args:
            cfg: config.yaml 中的 rag 配置节（ConfigNode）
            chat_model: langchain 聊天模型；缺省由 config 构建
        """
        self.cfg = cfg
        self.docs_dir = self._path(getattr(cfg, "docs_dir", "../pdfs"))
        self.chunk_size = int(getattr(cfg, "chunk_size", 800))
        self.chunk_overlap = int(getattr(cfg, "chunk_overlap", 120))
        self.top_k = int(getattr(cfg, "top_k", 4))
        self.bm25_weight = float(getattr(cfg, "bm25_weight", 0.3))
        emb_cfg = getattr(cfg, "embedding", None) or {}
        self.embedding_model = getattr(emb_cfg, "model", "Qwen/Qwen3-Embedding-0.6B")
        self.embedding_device = getattr(emb_cfg, "device", "cpu")
        self.embedding_batch_size = int(getattr(emb_cfg, "batch_size", 16))
        self.embedding_dtype = getattr(emb_cfg, "dtype", "fp16")
        self.index_dir = self._path(getattr(cfg, "index_dir", "../output/.rag_index"))

        self.chat_model = chat_model or ChatOpenAICompat()
        self._embedder = None
        self._retriever: Optional[HybridRetriever] = None
        self._documents: List[Document] = []
        self._vectors: Optional[np.ndarray] = None
        self._chain = None

    @staticmethod
    def _path(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else RAG_DIR / p

    # ---------- 索引构建 / 缓存 ----------

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder(
                self.embedding_model,
                self.embedding_device,
                self.embedding_batch_size,
                self.embedding_dtype,
            )
        return self._embedder

    def _meta(self) -> Dict[str, Any]:
        return {
            "model": self.embedding_model,
            "device": self.embedding_device,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }

    def ensure_index(self, force: bool = False) -> Dict[str, Any]:
        """构建或加载索引，返回统计信息。"""
        if not force and self._retriever is not None:
            return self.stats()
        chunks_file, vectors_file, meta_file = self._cache_files()
        if not force and meta_file.is_file() and vectors_file.is_file() and chunks_file.is_file():
            try:
                saved_meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if saved_meta == self._meta():
                    self._load_cache(chunks_file, vectors_file)
                    return self.stats()
            except Exception:
                pass  # 缓存损坏则重建
        return self._build_index()

    def _cache_files(self):
        return (
            self.index_dir / "chunks.json",
            self.index_dir / "vectors.npy",
            self.index_dir / "meta.json",
        )

    def _build_index(self) -> Dict[str, Any]:
        print(f"[RAG] 加载语料目录: {self.docs_dir}")
        pages = load_pdfs(self.docs_dir)
        documents = split_documents(pages, self.chunk_size, self.chunk_overlap)
        if not documents:
            raise RuntimeError(f"语料目录中没有可用文本: {self.docs_dir}")
        print(f"[RAG] 切块完成: {len(documents)} 块，正在生成向量（首次较慢，结果已缓存）...")
        vectors = np.asarray(
            self.embedder.embed_documents([d.page_content for d in documents]),
            dtype=np.float32,
        )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_file, vectors_file, meta_file = self._cache_files()
        chunks_file.write_text(
            json.dumps(
                [{"content": d.page_content, "metadata": d.metadata} for d in documents],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        np.save(vectors_file, vectors)
        meta_file.write_text(json.dumps(self._meta(), ensure_ascii=False), encoding="utf-8")
        self._documents, self._vectors = documents, vectors
        return self.stats()

    def _load_cache(self, chunks_file: Path, vectors_file: Path) -> None:
        data = json.loads(chunks_file.read_text(encoding="utf-8"))
        self._documents = [Document(page_content=c["content"], metadata=c["metadata"]) for c in data]
        self._vectors = np.load(vectors_file)
        print(f"[RAG] 已加载索引缓存: {len(self._documents)} 块")

    def stats(self) -> Dict[str, Any]:
        return {
            "engine": "rag",
            "docs_dir": str(self.docs_dir),
            "chunks": len(self._documents),
            "vector_dim": int(self._vectors.shape[1]) if self._vectors is not None else 0,
            "index_ready": self._vectors is not None and len(self._documents) > 0,
        }

    # ---------- 检索 / 问答 ----------

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self.ensure_index()
            self._retriever = HybridRetriever(
                embedding=self.embedder,
                documents=self._documents,
                vectors=self._vectors,
                bm25=BM25Okapi([_char_bigrams(d.page_content) for d in self._documents]),
                top_k=self.top_k,
                bm25_weight=self.bm25_weight,
            )
        return self._retriever

    def _chain_builder(self):
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("system", "【参考资料】\n{context}"),
                    MessagesPlaceholder(variable_name="history", optional=True),
                    ("human", "{question}"),
                ]
            )
            self._chain = prompt | self.chat_model | StrOutputParser()
        return self._chain

    def retrieve(self, question: str) -> List[Tuple[Document, float]]:
        """检索 top_k 片段，返回 [(文档, 分数)]。"""
        return self.retriever.search(question, k=self.top_k)

    @staticmethod
    def _context(docs: List[Tuple[Document, float]]) -> str:
        parts = []
        for i, (doc, _) in enumerate(docs, 1):
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            parts.append(f"[片段{i} | 来源: {src} 第{page}页]\n{doc.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _sources(docs: List[Tuple[Document, float]], max_len: int = 160) -> List[Dict[str, Any]]:
        out = []
        for doc, score in docs:
            excerpt = doc.page_content.strip()
            if len(excerpt) > max_len:
                excerpt = excerpt[:max_len] + "…"
            out.append(
                {
                    "type": "chunk",
                    "score": round(score, 4),
                    "source": doc.metadata.get("source", "?"),
                    "page": doc.metadata.get("page", None),
                    "text": excerpt,
                }
            )
        return out

    def _inputs(self, question: str, history: Optional[List[Tuple[str, str]]]) -> Dict[str, Any]:
        docs = self.retrieve(question)
        return {
            "context": self._context(docs),
            "history": list(history or []),
            "question": question,
        }

    def answer(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        return self._chain_builder().invoke(self._inputs(question, history))

    def stream(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Iterator[str]:
        yield from self._chain_builder().stream(self._inputs(question, history))
