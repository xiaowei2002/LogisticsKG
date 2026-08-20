"""基于语义哈希的知识图谱去重。"""
import os
import unicodedata
from pathlib import Path

import inflect
from semhash import SemHash
from sentence_transformers import SentenceTransformer

from src.models import Graph

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def get_default_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """加载中文嵌入模型，模型缓存到 src/pretrain。"""
    pretrain_dir = Path(__file__).resolve().parent.parent / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(pretrain_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(pretrain_dir))
    return SentenceTransformer(model_name)


class DeduplicateList:
    """对字符串列表进行语义去重。去重前先NFKC规范化与单数化。"""

    def __init__(
        self, threshold: float = 0.95, model: SentenceTransformer | None = None
    ):
        self.threshold = threshold
        self.model = model
        self.inflect_engine = inflect.engine()
        self.original_map: dict[str, str] = {}
        self.items_map: dict[str, str] = {}
        self.duplicates: dict[str, str] = {}
        self.deduplicated: list[str] = []

        # 统计值
        self.total_items: int = 0
        self.deduplicated_items: int = 0
        self.duplicate_items: int = 0
        self.reduction: float = 0.0

    def normalize(self, text: str) -> str:
        """规范化文本。"""
        return unicodedata.normalize("NFKC", text)

    def singularize(self, text: str) -> str:
        """将文本单数化。"""
        tokens = []
        for tok in text.split():
            sing = self.inflect_engine.singular_noun(tok)
            tokens.append(sing if isinstance(sing, str) and sing else tok)
        return " ".join(tokens).strip()

    def deduplicate(self, items: list[str]) -> list[str]:
        """使用语义哈希对条目列表去重。

        Args:
            items: 待去重的条目列表

        Returns:
            去重后的条目列表
        """
        self.total_items = len(items)

        # 先规范化并单数化每个字符串
        normalized_items = set()
        for item in items:
            normalized = self.normalize(item)
            singular = self.singularize(normalized)
            self.original_map[item] = singular
            self.items_map[singular] = item
            normalized_items.add(singular)

        # 对规范化后的字符串去重
        semhash = SemHash.from_records(records=list(normalized_items), model=self.model)
        deduplication_result = semhash.self_deduplicate(threshold=self.threshold)

        self.deduplicated_items = len(deduplication_result.selected)
        self.duplicate_items = len(deduplication_result.filtered)
        self.reduction = (
            (self.duplicate_items / self.total_items) * 100 if self.total_items else 0.0
        )

        # 将每个被去重的条目映射到其代表条目的原始字符串
        for selected in deduplication_result.selected_with_duplicates:
            representative = selected.record
            for duplicate_value, _score in selected.duplicates:
                self.items_map[duplicate_value] = self.items_map[representative]
                if duplicate_value not in self.duplicates:
                    self.duplicates[duplicate_value] = representative

        self.deduplicated = deduplication_result.selected
        return self.deduplicated

    def stats(self) -> str:
        return (
            f"Total items: {self.total_items}; "
            f"Deduplicated items: {self.deduplicated_items}; "
            f"Duplicate items: {self.duplicate_items}; "
            f"Reduction: {self.reduction:.1f}"
        )


def run_semhash_deduplication(
    graph: Graph,
    similarity_threshold: float = 0.95,
    retrieval_model: SentenceTransformer | None = None,
) -> Graph:
    """对图谱进行去重。"""
    if retrieval_model is None:
        retrieval_model = get_default_embedding_model()

    # 对图谱各组成部分分别去重
    entities_dedup = DeduplicateList(similarity_threshold, model=retrieval_model)
    entities_dedup.deduplicate(list(graph.entities))
    edges_dedup = DeduplicateList(similarity_threshold, model=retrieval_model)
    edges_dedup.deduplicate(list(graph.edges))

    def _canonical_entity(name: str) -> str:
        if name in entities_dedup.original_map:
            return entities_dedup.items_map[entities_dedup.original_map[name]]
        return name

    def _canonical_edge(name: str) -> str:
        if name in edges_dedup.original_map:
            return edges_dedup.items_map[edges_dedup.original_map[name]]
        return name

    def _get_relation(relation: tuple[str, str, str]) -> tuple[str, str, str]:
        """获取转换后的关系三元组。"""
        return (
            _canonical_entity(relation[0]),
            _canonical_edge(relation[1]),
            _canonical_entity(relation[2]),
        )

    # 去重图谱
    new_entities = [
        entities_dedup.items_map[item] for item in entities_dedup.deduplicated
    ]
    new_edges = [edges_dedup.items_map[item] for item in edges_dedup.deduplicated]
    new_relations = {_get_relation(relation) for relation in graph.relations}

    # 更新实体元数据的键，使其与去重后的实体名一致
    new_entity_metadata: dict[str, set[str]] | None = None
    if graph.entity_metadata:
        new_entity_metadata = {}
        for original_entity, metadata_set in graph.entity_metadata.items():
            deduped_entity = _canonical_entity(original_entity)
            if deduped_entity in new_entity_metadata:
                new_entity_metadata[deduped_entity].update(metadata_set)
            else:
                new_entity_metadata[deduped_entity] = metadata_set.copy()

    # 更新实体簇与边簇的键/成员，与去重结果保持一致
    new_entity_clusters: dict[str, set[str]] | None = None
    if graph.entity_clusters:
        new_entity_clusters = {}
        for representative, members in graph.entity_clusters.items():
            deduped_rep = _canonical_entity(representative)
            deduped_members = {_canonical_entity(m) for m in members}
            if deduped_rep in new_entity_clusters:
                new_entity_clusters[deduped_rep].update(deduped_members)
            else:
                new_entity_clusters[deduped_rep] = deduped_members

    new_edge_clusters: dict[str, set[str]] | None = None
    if graph.edge_clusters:
        new_edge_clusters = {}
        for representative, members in graph.edge_clusters.items():
            deduped_rep = _canonical_edge(representative)
            deduped_members = {_canonical_edge(m) for m in members}
            if deduped_rep in new_edge_clusters:
                new_edge_clusters[deduped_rep].update(deduped_members)
            else:
                new_edge_clusters[deduped_rep] = deduped_members

    return Graph(
        entities=new_entities,
        edges=new_edges,
        relations=new_relations,
        entity_clusters=new_entity_clusters,
        edge_clusters=new_edge_clusters,
        entity_metadata=new_entity_metadata,
    )


if __name__ == "__main__":
    graph = Graph(
        entities={"整车物流", "汽车物流", "物流", "冷链运输", "冷链物流"},
        edges={"属于"},
        relations={
            ("整车物流", "属于", "物流"),
            ("汽车物流", "属于", "物流"),
            ("冷链运输", "属于", "冷链物流"),
        },
    )

    print("去重前:")
    graph.stats()

    deduplicated_graph = run_semhash_deduplication(graph, similarity_threshold=0.95)

    print("\n去重后:")
    deduplicated_graph.stats()
    print("实体:", deduplicated_graph.entities)
    print("关系:", deduplicated_graph.relations)
