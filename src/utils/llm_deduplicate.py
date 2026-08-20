"""基于 LLM 的知识图谱去重。"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

import dspy
import numpy as np
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from src.models import Graph


def _dict_clusters_to_lists(clusters) -> list[list[str]]:
    """将 {代表项: 成员集合} 形式的簇转换为列表的列表。

    本项目 Graph 的 entity_clusters/edge_clusters 为 dict[str, set[str]]，
    而 LLMDeduplicate 内部以 list[list[str]] 表示簇。
    """
    if not clusters:
        return []
    if isinstance(clusters, dict):
        return [
            [rep] + [m for m in members if m != rep]
            for rep, members in clusters.items()
        ]
    return [list(cluster) for cluster in clusters]


class LLMDeduplicate:
    """使用检索增强 + LLM 对知识图谱实体/边进行去重。
    先调用 cluster() 将实体/边按嵌入分簇，再调用 deduplicate()
    在每个簇内使用 LLM 识别重复项并选取代表名称。
    """

    graph: Graph
    nodes: list[str]
    edges: list[str]
    node_clusters: list[list[str]]
    edge_clusters: list[list[str]]
    retrieval_model: SentenceTransformer
    lm: dspy.LM

    def __init__(self, retrieval_model: SentenceTransformer, lm: dspy.LM, graph: Graph):
        """初始化并缓存节点/边的嵌入与 BM25 索引。"""
        self.graph = graph
        self.nodes = list(graph.entities)
        self.edges = list(graph.edges)
        self.node_clusters = _dict_clusters_to_lists(graph.entity_clusters)
        self.edge_clusters = _dict_clusters_to_lists(graph.edge_clusters)
        self.retrieval_model = retrieval_model
        self.lm = lm

        self.node_embeddings = retrieval_model.encode(
            self.nodes, show_progress_bar=True
        )
        self.node_bm25 = BM25Okapi([text.lower().split() for text in self.nodes])

        self.edge_embeddings = retrieval_model.encode(
            self.edges, show_progress_bar=True
        )
        self.edge_bm25 = BM25Okapi([text.lower().split() for text in self.edges])

        dspy.configure(lm=lm)

    def get_relevant_items(
        self, query: str, top_k: int = 50, item_type: str = "node"
    ) -> list[str]:
        """使用 BM25 与嵌入的排序融合，检索 top-k 相关项。"""
        query_tokens = query.lower().split()

        if item_type == "node":
            bm25_scores = self.node_bm25.get_scores(query_tokens)
            embeddings = self.node_embeddings
            items = self.nodes
        else:
            bm25_scores = self.edge_bm25.get_scores(query_tokens)
            embeddings = self.edge_embeddings
            items = self.edges

        query_embedding = self.retrieval_model.encode([query], show_progress_bar=False)
        embedding_scores = cosine_similarity(query_embedding, embeddings).flatten()

        combined_scores = 0.5 * bm25_scores + 0.5 * embedding_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        return [items[i] for i in top_indices]

    def cluster(self):
        """使用 KMeans 将节点/边嵌入分簇，并保存为 list[list[str]]。"""
        cluster_size = 128

        embedding_sets = {"node": self.node_embeddings, "edge": self.edge_embeddings}

        for embedding_type, embeddings in embedding_sets.items():
            n_samples = len(embeddings)
            num_clusters = max(1, n_samples // cluster_size)

            # Step 1: 聚类中心
            kmeans = KMeans(
                n_clusters=num_clusters,
                init="random",
                n_init=1,
                max_iter=20,
                tol=0.0,
                algorithm="lloyd",
                verbose=True,
            )
            kmeans.fit(embeddings.astype(np.float32))
            centroids = kmeans.cluster_centers_

            # Step 2: 按最近中心分配每个点（每簇最多 cluster_size 个）
            distances = cdist(embeddings, centroids)
            assignments = np.argsort(distances, axis=1)

            clusters: list[list[int]] = [[] for _ in range(num_clusters)]
            assigned = np.zeros(n_samples, dtype=bool)

            for rank in range(num_clusters):
                for i in range(n_samples):
                    if assigned[i]:
                        continue
                    cluster_id = assignments[i, rank]
                    if len(clusters[cluster_id]) < cluster_size:
                        clusters[cluster_id].append(i)
                        assigned[i] = True

            unassigned = np.where(~assigned)[0]

            # 若有未分配项，将其作为独立簇加入
            if len(unassigned) > 0:
                logger.debug(
                    "Adding {} unassigned items as a separate cluster", len(unassigned)
                )
                clusters.append(unassigned.tolist())
            else:
                logger.debug("No unassigned items to add as a cluster")

            items = self.nodes if embedding_type == "node" else self.edges
            clusters_data = [[items[idx] for idx in cluster] for cluster in clusters]

            if embedding_type == "node":
                self.node_clusters = clusters_data
            else:
                self.edge_clusters = clusters_data

            logger.debug("Number of {} clusters: {}", embedding_type, len(clusters))
            logger.debug("Distribution of cluster sizes: {}...", [len(c) for c in clusters[:5]])

    def deduplicate_cluster(
        self, cluster: list[str], item_type: str = "node"
    ) -> tuple[set[str], dict[str, set[str]]]:
        cluster = cluster.copy()

        items = set()
        item_clusters = {}
        plural_type = "entities" if item_type == "node" else "edges"
        singular_type = "entity" if item_type == "node" else "edge"

        logger.info(
            "Starting deduplication of {} {} in cluster", len(cluster), plural_type
        )

        processed_count = 0
        while len(cluster) > 0:
            processed_count += 1
            item = cluster.pop()

            logger.debug(
                "[{}/{}] Processing {}: '{}'",
                processed_count,
                len(cluster),
                singular_type,
                item,
            )

            relevant_items = self.get_relevant_items(item, 16, item_type)

            class Deduplicate(dspy.Signature):
                __doc__ = f"""Find duplicate {plural_type} for the item and an alias that best represents the duplicates. Duplicates are those that are the same in meaning, such as with variation in tense, plural form, stem form, case, abbreviation, shorthand. Return an empty list if there are none.
                """

                item: str = dspy.InputField()
                set: list[str] = dspy.InputField()
                duplicates: list[str] = dspy.OutputField(
                    description="Exact matches to items in {plural_type} set"
                )
                alias: str = dspy.OutputField(
                    description=f"Best {singular_type} name to represent the duplicates, ideally from the {plural_type} set"
                )

            deduplicate = dspy.Predict(Deduplicate)
            result = deduplicate(item=item, set=relevant_items)
            items.add(result.alias)

            # 只保留出现在当前簇中的重复项
            duplicates = [dup for dup in result.duplicates if dup in cluster]

            if len(duplicates) > 0:
                logger.info(
                    "  → Using alias '{}' to represent: '{}' and {}",
                    result.alias,
                    item,
                    duplicates,
                )
                item_clusters[result.alias] = {item}
                for duplicate in duplicates:
                    cluster.remove(duplicate)
                    item_clusters[result.alias].add(duplicate)
            else:
                logger.debug(
                    "  ✗ No duplicates found for '{}', keeping as is", item
                )
                item_clusters[item] = {item}

        logger.debug(
            "Deduplication complete: {} unique {} from original {}",
            len(items),
            plural_type,
            processed_count,
        )

        return items, item_clusters

    def deduplicate(self) -> Graph:
        """并行处理各簇，重建去重后的图谱。"""
        entities = set()
        edges = set()
        entity_clusters = {}
        edge_clusters = {}

        pool = ThreadPoolExecutor(max_workers=64)

        node_futures = []
        cnt_nodes = 0
        for cluster in self.node_clusters:
            cnt_nodes += len(cluster)
            node_futures.append(pool.submit(self.deduplicate_cluster, cluster, "node"))

        edge_futures = []
        cnt_edges = 0
        for cluster in self.edge_clusters:
            cnt_edges += len(cluster)
            edge_futures.append(pool.submit(self.deduplicate_cluster, cluster, "edge"))

        for i, future in enumerate(node_futures):
            try:
                cluster_entities, cluster_entity_map = future.result()
                entities.update(cluster_entities)
                entity_clusters.update(cluster_entity_map)
            except Exception as e:
                logger.error("Error processing node cluster {}: {}", i, e)

        for i, future in enumerate(edge_futures):
            try:
                cluster_edges, cluster_edge_map = future.result()
                edges.update(cluster_edges)
                edge_clusters.update(cluster_edge_map)
            except Exception as e:
                logger.error("Error processing edge cluster {}: {}", i, e)

        logger.info(
            "Finished processing all clusters with {} nodes and {} edges LLM calls",
            cnt_nodes,
            cnt_edges,
        )

        # 根据簇更新关系三元组
        relations: set[tuple[str, str, str]] = set()

        for s, p, o in self.graph.relations:
            if s not in entities:
                for rep, cluster in entity_clusters.items():
                    if s in cluster:
                        s = rep
                        break

            if p not in edges:
                for rep, cluster in edge_clusters.items():
                    if p in cluster:
                        p = rep
                        break

            if o not in entities:
                for rep, cluster in entity_clusters.items():
                    if o in cluster:
                        o = rep
                        break

            relations.add((s, p, o))

        # 更新实体元数据键，使其与去重后的实体名一致
        new_entity_metadata: dict[str, set[str]] | None = None
        if self.graph.entity_metadata:
            new_entity_metadata = {}
            for original_entity, metadata_set in self.graph.entity_metadata.items():
                deduped_entity = original_entity
                for rep, cluster in entity_clusters.items():
                    if original_entity in cluster:
                        deduped_entity = rep
                        break
                if deduped_entity in new_entity_metadata:
                    new_entity_metadata[deduped_entity].update(metadata_set)
                else:
                    new_entity_metadata[deduped_entity] = metadata_set.copy()

        return Graph(
            entities=entities,
            edges=edges,
            relations=relations,
            entity_clusters=entity_clusters,
            edge_clusters=edge_clusters,
            entity_metadata=new_entity_metadata,
        )


if __name__ == "__main__":
    load_dotenv()
    #  构建一个带近似重复实体的示例图谱
    graph = Graph(
        entities={"整车物流", "整车运输", "汽车物流", "仓储管理"},
        edges={"属于"},
        relations={
            ("整车物流", "属于", "汽车物流"),
            ("仓储管理", "属于", "汽车物流"),
        },
    )
    # 2. 检索模型（Qwen 嵌入，模型缓存到 src/pretrain）
    pretrain_dir = Path(__file__).resolve().parent.parent / "pretrain"
    pretrain_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(pretrain_dir))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(pretrain_dir))

    retrieval_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    # 3. LLM
    lm = dspy.LM(
        "openai/" + os.getenv("OPENAI_MODEL_NAME", "qwen3.8-max"),
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("OPENAI_BASE_URL"),
    )

    # 4. 先分簇，再在每个簇内用 LLM 去重
    dedup = LLMDeduplicate(retrieval_model, lm, graph)
    dedup.cluster()
    deduped_graph = dedup.deduplicate()

    print("去重前:")
    graph.stats()
    print("\n去重后:")
    deduped_graph.stats()
    print("实体:", deduped_graph.entities)
    print("关系:", deduped_graph.relations)
