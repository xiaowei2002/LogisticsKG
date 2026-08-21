"""langchain Embeddings 封装：本地 Qwen3-Embedding-0.6B（sentence-transformers）。

模型缓存到 src/pretrain，离线可用；通过 functools.lru_cache 全局共享，
RAG 与 GraphRAG 复用同一份嵌入模型，避免重复加载。
"""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# sentence-transformers / HF 缓存目录（与 src 管线共用，离线加载）
_PRETRAIN_DIR = Path(__file__).resolve().parent.parent.parent / "src" / "pretrain"


def _setup_env() -> None:
    """指向本地模型缓存并禁用在线检查，保证离线可用。"""
    os.environ.setdefault("HF_HOME", str(_PRETRAIN_DIR))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_PRETRAIN_DIR))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@functools.lru_cache(maxsize=8)
def get_embedder(
    model: str, device: str = "cpu", batch_size: int = 16, dtype: str = "fp16"
) -> "SentenceTransformerEmbeddings":
    """按 (model, device, batch_size, dtype) 缓存并共享嵌入器实例。"""
    _setup_env()
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA 不可用")
        except RuntimeError:
            device = "cpu"
    return SentenceTransformerEmbeddings(
        model_name=model, device=device, batch_size=batch_size, dtype=dtype
    )


class SentenceTransformerEmbeddings(Embeddings):
    """把 sentence-transformers 模型包装为 langchain Embeddings 接口。

    embed_documents / embed_query 供 langchain 检索器（BaseRetriever）直接调用。
    dtype 支持 fp16 / fp32：fp16 内存减半（内存紧张时推荐），fp32 精度略高。
    """

    model_name: str
    device: str = "cpu"
    batch_size: int = 16
    dtype: str = "fp16"
    _model: Any = None

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        dtype: str = "fp16",
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.dtype = dtype
        _setup_env()
        model_kwargs: Dict[str, Any] = {}
        if dtype in ("fp16", "float16"):
            import torch

            model_kwargs["torch_dtype"] = torch.float16
        elif dtype in ("fp32", "float32"):
            import torch

            model_kwargs["torch_dtype"] = torch.float32
        self._model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]
