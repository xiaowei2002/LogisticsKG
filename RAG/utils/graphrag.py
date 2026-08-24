"""GraphRAG 引擎：基于 merge 知识图谱（output/merged_graph.json）的检索增强问答。

流程：加载 merged_graph.json → networkx 有向图 → 实体链接（语义 + BM25 混合）
      → 命中实体 k 跳邻域子图 → 三元组上下文 → LCEL 链流式回答。
实体向量缓存到 index_dir，二次启动秒级加载。
"""
import json
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import networkx as nx
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ConfigDict, Field
from rank_bm25 import BM25Okapi

from RAG.utils.chat_model import ChatOpenAICompat
from RAG.utils.embeddings import get_embedder
from RAG.utils.rag import RAG_DIR, _char_bigrams

_SYSTEM_PROMPT = """你是物流领域知识问答助手，将基于给定的【知识图谱三元组】回答用户问题。
每个三元组形如 "实体A -[关系]-> 实体B"，描述物流领域实体之间的关系。
要求：
1. 只依据给出的三元组进行推理作答，不要编造三元组中不存在的事实；
2. 若三元组不足以回答问题，请明确说明"知识图谱中未找到相关信息"；
3. 用中文回答，条理清晰、简洁准确。"""


class GraphRAGEngine:
    """知识图谱 RAG 引擎：实体链接 + 子图检索 + LLM 问答。"""

    def __init__(self, cfg: Any, chat_model: Optional[ChatOpenAICompat] = None):
        """
        Args:
            cfg: config.yaml 中的 graphrag 配置节（ConfigNode）
            chat_model: langchain 聊天模型；缺省由 config 构建
        """
        self.cfg = cfg
        self.graph_path = self._path(getattr(cfg, "graph_path", "../output/merged_graph.json"))
        self.entity_top_k = int(getattr(cfg, "entity_top_k", 8))
        self.context_depth = int(getattr(cfg, "context_depth", 2))
        self.max_triples = int(getattr(cfg, "max_triples", 40))
        emb_cfg = getattr(cfg, "embedding", None) or {}
        self.embedding_model = getattr(emb_cfg, "model", "Qwen/Qwen3-Embedding-0.6B")
        self.embedding_device = getattr(emb_cfg, "device", "cpu")
        self.embedding_batch_size = int(getattr(emb_cfg, "batch_size", 16))
        self.embedding_dtype = getattr(emb_cfg, "dtype", "fp16")
        self.index_dir = self._path(getattr(cfg, "index_dir", "../output/.graphrag_index"))

        self.chat_model = chat_model or ChatOpenAICompat()
        self._embedder = None
        self._graph: Optional[nx.DiGraph] = None
        self._nodes: List[str] = []
        self._node_vectors: Optional[np.ndarray] = None
        self._bm25: Optional[BM25Okapi] = None
        self._chain = None

    @staticmethod
    def _path(value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else RAG_DIR / p

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

    # ---------- 图谱与实体索引 ----------

    def load_graph(self) -> nx.DiGraph:
        """加载 merged_graph.json 为 networkx 有向图（边属性 relation）。"""
        if not self.graph_path.is_file():
            raise FileNotFoundError(
                f"merge 知识图谱不存在: {self.graph_path}（请先运行图谱构建管线，或改用 RAG 模式）"
            )
        with self.graph_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        graph = nx.DiGraph()
        graph.add_nodes_from(data.get("entities", []))
        for subject, relation, target in data.get("relations", []):
            graph.add_node(subject)
            graph.add_node(target)
            graph.add_edge(subject, target, relation=relation)
        return graph

    def ensure_index(self, force: bool = False) -> Dict[str, Any]:
        """加载图谱 + 构建/加载实体向量索引，返回统计信息。"""
        if not force and self._graph is not None:
            return self.stats()
        self._graph = self.load_graph()
        self._nodes = sorted(self._graph.nodes)
        if not self._nodes:
            raise RuntimeError(f"知识图谱为空: {self.graph_path}")

        nodes_file = self.index_dir / "entities.json"
        vectors_file = self.index_dir / "entity_vectors.npy"
        meta_file = self.index_dir / "meta.json"
        meta = {"model": self.embedding_model, "device": self.embedding_device}

        cached = (
            nodes_file.is_file()
            and vectors_file.is_file()
            and meta_file.is_file()
            and json.loads(meta_file.read_text(encoding="utf-8")) == meta
        )
        if cached and not force:
            saved_nodes = json.loads(nodes_file.read_text(encoding="utf-8"))
            if saved_nodes == self._nodes:
                self._node_vectors = np.load(vectors_file)
                print(f"[GraphRAG] 已加载实体索引缓存: {len(self._nodes)} 个实体")
            else:
                cached = False
        if not cached:
            print(f"[GraphRAG] 正在为 {len(self._nodes)} 个实体生成向量（首次较慢，结果已缓存）...")
            self._node_vectors = np.asarray(
                self.embedder.embed_documents(self._nodes), dtype=np.float32
            )
            self.index_dir.mkdir(parents=True, exist_ok=True)
            nodes_file.write_text(
                json.dumps(self._nodes, ensure_ascii=False), encoding="utf-8"
            )
            np.save(vectors_file, self._node_vectors)
            meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

        self._bm25 = BM25Okapi([_char_bigrams(n) for n in self._nodes])
        return self.stats()

    def stats(self) -> Dict[str, Any]:
        return {
            "engine": "graphrag",
            "graph_path": str(self.graph_path),
            "nodes": int(self._graph.number_of_nodes()) if self._graph is not None else 0,
            "edges": int(self._graph.number_of_edges()) if self._graph is not None else 0,
            "index_ready": self._graph is not None and self._node_vectors is not None,
        }

    # ---------- 实体链接与子图检索 ----------

    def link_entities(self, query: str) -> List[Tuple[str, float]]:
        """混合检索命中实体：[(实体名, 融合分数)]，按分数降序。"""
        query_vec = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
        semantic = self._node_vectors @ query_vec
        bm25_raw = np.asarray(self._bm25.get_scores(_char_bigrams(query)), dtype=np.float64)
        bm25_max = float(bm25_raw.max()) if bm25_raw.size else 0.0
        bm25_norm = bm25_raw / bm25_max if bm25_max > 0 else np.zeros_like(bm25_raw)
        fused = 0.7 * semantic + 0.3 * bm25_norm
        order = np.argsort(fused)[::-1][: self.entity_top_k]
        return [(self._nodes[i], float(fused[i])) for i in order]

    def retrieve_subgraph(
        self, matched: List[Tuple[str, float]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """从命中实体出发做 k 跳扩展，收集三元组上下文。

        Returns:
            (triples, entities)：三元组列表（含分数）与参与实体列表，供 prompt 与来源展示。
        """
        if not matched:
            return [], []
        entity_scores = {name: score for name, score in matched}
        visited = {name for name, _ in matched}
        frontier = deque([(name, 1) for name, _ in matched])
        triples: dict[Tuple[str, str, str], float] = {}

        while frontier:
            node, depth = frontier.popleft()
            if depth > self.context_depth:
                continue
            neighbors: List[Tuple[str, bool]] = [(s, True) for s in self._graph.successors(node)]
            neighbors += [(p, False) for p in self._graph.predecessors(node)]
            for neighbor, is_successor in neighbors:
                if is_successor:
                    rel = self._graph[node][neighbor]["relation"]
                    triple = (node, rel, neighbor)
                else:
                    rel = self._graph[neighbor][node]["relation"]
                    triple = (neighbor, rel, node)
                score = entity_scores.get(node, 0.0)
                triples[triple] = max(triples.get(triple, 0.0), score)
                if neighbor not in visited:
                    visited.add(neighbor)
                    entity_scores.setdefault(neighbor, score * 0.5)
                    frontier.append((neighbor, depth + 1))

        ranked = sorted(triples.items(), key=lambda kv: kv[1], reverse=True)[: self.max_triples]
        triple_dicts = [
            {"subject": s, "relation": p, "object": o, "score": round(score, 4)}
            for (s, p, o), score in ranked
        ]
        entity_dicts = [
            {"name": name, "score": round(score, 4)}
            for name, score in sorted(
                entity_scores.items(), key=lambda kv: kv[1], reverse=True
            )[: self.entity_top_k]
        ]
        return triple_dicts, entity_dicts

    @staticmethod
    def _context(triples: List[Dict[str, Any]]) -> str:
        lines = []
        for i, t in enumerate(triples, 1):
            lines.append(f"{i}. {t['subject']} -[{t['relation']}]-> {t['object']}")
        return "\n".join(lines)

    # ---------- 问答 ----------

    def _chain_builder(self):
        if self._chain is None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _SYSTEM_PROMPT),
                    ("system", "【知识图谱三元组】\n{context}"),
                    MessagesPlaceholder(variable_name="history", optional=True),
                    ("human", "{question}"),
                ]
            )
            self._chain = prompt | self.chat_model | StrOutputParser()
        return self._chain

    def retrieve(self, question: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """实体链接 + 子图检索，返回 (triples, entities)。"""
        matched = self.link_entities(question)
        return self.retrieve_subgraph(matched)

    def _inputs(self, question: str, history: Optional[List[Tuple[str, str]]]) -> Dict[str, Any]:
        triples, _ = self.retrieve(question)
        return {
            "context": self._context(triples),
            "history": list(history or []),
            "question": question,
        }

    def answer(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> str:
        return self._chain_builder().invoke(self._inputs(question, history))

    def stream(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Iterator[str]:
        yield from self._chain_builder().stream(self._inputs(question, history))
