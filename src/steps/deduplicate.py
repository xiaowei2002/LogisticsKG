"""
在获取实体和关系后进行去重
"""
import enum

import dspy
from sentence_transformers import SentenceTransformer

from src.models import Graph
from src.utils.deduplicate import run_semhash_deduplication
from src.utils.llm_deduplicate import LLMDeduplicate


class DeduplicateMethod(enum.Enum):
    SEMHASH = "semhash"  # 基于确定性规则 + 语义哈希去重
    LM_BASED = "lm_based"  # KNN 聚类 + 簇内 LLM 去重
    FULL = "full"  # 先语义哈希，再做聚类 + 簇内 LLM 去重


def run_deduplication(
    lm: dspy.LM,
    graph: Graph,
    method: DeduplicateMethod = DeduplicateMethod.FULL,
    retrieval_model: SentenceTransformer | None = None,
    semhash_similarity_threshold: float = 0.95,
) -> Graph:
    """按指定方式对图谱去重。

    Args:
        lm: 用于 LLM 去重的 dspy 语言模型
        graph: 待去重的图谱
        method: 去重方式（SEMHASH / LM_BASED / FULL）
        retrieval_model: 检索嵌入模型，非 SEMHASH 方式时必填
        semhash_similarity_threshold: 语义哈希去重的相似度阈值

    Returns:
        去重后的图谱
    """
    if method != DeduplicateMethod.SEMHASH and retrieval_model is None:
        raise ValueError("No retrieval model provided")

    if method == DeduplicateMethod.SEMHASH:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
    elif method == DeduplicateMethod.LM_BASED:
        llm_deduplicate = LLMDeduplicate(retrieval_model, lm, graph)
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()
    elif method == DeduplicateMethod.FULL:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold
        )
        llm_deduplicate = LLMDeduplicate(retrieval_model, lm, deduplicated_graph)
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()

    return deduplicated_graph


