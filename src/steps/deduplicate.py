"""
在获取实体和关系后进行去重
"""
import enum
import os
from dotenv import load_dotenv
import dspy
from sentence_transformers import SentenceTransformer

from src.models import Graph
from src.utils.deduplicate import get_default_embedding_model, run_semhash_deduplication
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
        retrieval_model: 检索嵌入模型，默认自动加载中文嵌入模型（缓存到 src/pretrain）
        semhash_similarity_threshold: 语义哈希去重的相似度阈值

    Returns:
        去重后的图谱
    """
    if retrieval_model is None:
        retrieval_model = get_default_embedding_model()

    if method == DeduplicateMethod.SEMHASH:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold, retrieval_model
        )
    elif method == DeduplicateMethod.LM_BASED:
        llm_deduplicate = LLMDeduplicate(retrieval_model, lm, graph)
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()
    elif method == DeduplicateMethod.FULL:
        deduplicated_graph = run_semhash_deduplication(
            graph, semhash_similarity_threshold, retrieval_model
        )
        llm_deduplicate = LLMDeduplicate(retrieval_model, lm, deduplicated_graph)
        llm_deduplicate.cluster()
        deduplicated_graph = llm_deduplicate.deduplicate()

    return deduplicated_graph


if __name__ == "__main__":
    load_dotenv()

    # 1. 构建带近似重复实体的示例图谱
    graph = Graph(
        entities={"整车物流", "整车运输", "汽车物流", "仓储管理"},
        edges={"属于"},
        relations={
            ("整车物流", "属于", "汽车物流"),
            ("仓储管理", "属于", "汽车物流"),
        },
    )

    # 2. 检索模型（Qwen 嵌入，缓存到 src/pretrain）
    retrieval_model = get_default_embedding_model()

    # 3. LLM（复用 .env 中的 OpenAI 兼容配置）
    lm = dspy.LM(
        "openai/" + os.getenv("OPENAI_MODEL_NAME", "qwen3.8-max"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    # 4. 去重（FULL：先语义哈希，再做聚类 + LLM）
    #    仅做语义哈希可改用 method=DeduplicateMethod.SEMHASH，无需 retrieval_model
    deduplicated_graph = run_deduplication(
        lm,
        graph,
        method=DeduplicateMethod.FULL,
        retrieval_model=retrieval_model,
    )

    print("去重前:")
    graph.stats()
    print("\n去重后:")
    deduplicated_graph.stats()
    print("实体:", deduplicated_graph.entities)
    print("关系:", deduplicated_graph.relations)


