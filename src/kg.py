"""知识图谱生成与检索的编排类。"""
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Union

import dspy
import networkx as nx
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from src.models import Graph
from src.steps.deduplicate import DeduplicateMethod, run_deduplication
from src.steps.get_entities import get_entities
from src.steps.get_relations import get_relations
from src.utils import visualize_kg
from src.utils.chunk_text import chunk_text
from src.utils.deduplicate import get_default_embedding_model

# 静默 dspy 内部的日志
logging.getLogger("dspy").setLevel(logging.CRITICAL)


def _normalize_model(model: Optional[str]) -> str:
    """补全模型名：缺省时从环境变量读取，并确保带 provider 前缀。"""
    model = model or os.getenv("OPENAI_MODEL_NAME") or "qwen3.8-max"
    if "/" not in model:
        model = "openai/" + model
    return model


# ============ KG 生成成本估算常量（元 / 百万 token） ============
# qwen 文本模型粗略价格，请按实际账单调整。
KG_PRICE_INPUT_PER_MTOKEN = 1.6     # 输入约 ¥1.6 / 百万 token
KG_PRICE_OUTPUT_PER_MTOKEN = 6.4    # 输出约 ¥6.4 / 百万 token
KG_PROMPT_TOKENS_PER_CALL = 2000    # 每次调用 system prompt + 实体/格式约束的固定开销
KG_OUTPUT_TOKENS_PER_CALL = 800     # 每次调用输出实体/关系列表的估算 token

# 单分块实体数硬上限：防止过度抽取导致关系输入与去重成本爆炸。
MAX_ENTITIES_PER_CHUNK = 150


def estimate_kg_cost(chars: int, n_chunks: int) -> dict:
    """按字符数与分块数估算 KG 生成阶段的 token 用量与花费（元）。"""
    calls = n_chunks * 2  # 每块实体抽取 + 关系抽取两次调用
    input_tokens = chars + calls * KG_PROMPT_TOKENS_PER_CALL
    output_tokens = calls * KG_OUTPUT_TOKENS_PER_CALL
    cost = (
        input_tokens / 1_000_000 * KG_PRICE_INPUT_PER_MTOKEN
        + output_tokens / 1_000_000 * KG_PRICE_OUTPUT_PER_MTOKEN
    )
    return {
        "chunks": n_chunks,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_yuan": round(cost, 2),
    }


def _cache_key(step: str, model: str, *parts: str) -> str:
    """按 step + model + 内容生成稳定哈希，作为 LLM 缓存键。"""
    h = hashlib.sha256()
    h.update(step.encode("utf-8"))
    for p in (model,) + parts:
        h.update(b"\x00")
        h.update(p.encode("utf-8"))
    return h.hexdigest()


class _LLMCache:
    """基于内容哈希的磁盘缓存，命中时跳过 LLM 调用。

    键为 sha256(step | model | content...)，值为 JSON 可序列化结果。
    """

    def __init__(self, path: Path | None):
        self.path = path
        self.hits = 0
        self.data: dict = {}
        self._lock = threading.Lock()
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("LLM 缓存文件损坏，已忽略: {}", path)
                self.data = {}

    def get(self, step: str, model: str, *parts: str):
        if self.path is None:
            return None
        key = _cache_key(step, model, *parts)
        if key in self.data:
            self.hits += 1
            return self.data[key]
        return None

    def set(self, step: str, model: str, value, *parts: str) -> None:
        if self.path is None:
            return
        with self._lock:
            self.data[_cache_key(step, model, *parts)] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps(self.data, ensure_ascii=False), encoding="utf-8"
            )


class KGGen:
    def __init__(
        self,
        model: str = None,
        max_tokens: int = 16000,
        temperature: float = 0.0,
        reasoning_effort: str = None,
        api_key: str = None,
        api_base: str = None,
        retrieval_model: Optional[str] = None,
        disable_cache: bool = False,
    ):
        """初始化 KGGen。

        Args:
            model: 模型名，需带 provider 前缀（如 'openai/qwen3.8-max'），缺省从环境变量读取
            max_tokens: 最大生成 token 数
            temperature: 采样温度
            reasoning_effort: 推理强度（部分模型支持）
            api_key: API key，缺省从环境变量 OPENAI_API_KEY 读取
            api_base: API base URL，缺省从环境变量 OPENAI_BASE_URL 读取
            retrieval_model: 检索嵌入模型名，缺省时去重阶段自动加载中文模型
            disable_cache: 是否禁用 dspy 缓存
        """
        self.model = _normalize_model(model)
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_BASE_URL")
        self.retrieval_model: Optional[SentenceTransformer] = None
        self.lm: Optional[dspy.LM] = None
        self.disable_cache = disable_cache

        self.init_model(
            model=self.model,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=self.api_key,
            api_base=self.api_base,
            retrieval_model=retrieval_model,
        )

    def validate_temperature(self, temperature: float):
        if "gpt-5" in self.model and temperature < 1.0:
            raise ValueError("Temperature must be 1.0 for gpt-5 family models")

    def validate_max_tokens(self, max_tokens: int):
        if "gpt-5" in self.model and max_tokens < 16000:
            raise ValueError("Max tokens must be 16000 for gpt-5 family models")

    def init_model(
        self,
        model: str = None,
        reasoning_effort: str = None,
        max_tokens: int = None,
        temperature: float = None,
        retrieval_model: str = None,
        api_key: str = None,
        api_base: str = None,
    ):
        """用新参数（重新）初始化 dspy LM 与检索模型。"""
        if model is not None:
            self.model = _normalize_model(model)
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if api_key is not None:
            self.api_key = api_key
        if api_base is not None:
            self.api_base = api_base
        if temperature is not None:
            self.temperature = temperature
        if reasoning_effort is not None:
            self.reasoning_effort = reasoning_effort
        if retrieval_model is not None:
            self.retrieval_model = get_default_embedding_model(retrieval_model)

        self.validate_temperature(self.temperature)
        self.validate_max_tokens(self.max_tokens)

        lm_kwargs: dict = {
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "cache": not self.disable_cache,
            "timeout": 300,
            "num_retries": 2,
        }
        if self.reasoning_effort:
            lm_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        self.lm = dspy.LM(**lm_kwargs)

    @staticmethod
    def from_file(file_path: str) -> Graph:
        return Graph.from_file(file_path)

    @staticmethod
    def from_dict(graph_dict: dict) -> Graph:
        return Graph.model_validate(graph_dict)

    def generate(
        self,
        input_data: Union[str, List[Dict]],
        model: str = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
        chunk_size: Optional[int] = None,
        reasoning_effort: str = None,
        deduplication_method: DeduplicateMethod | None = DeduplicateMethod.SEMHASH,
        temperature: float = None,
        output_folder: Optional[str] = None,
        no_dspy: bool = False,
        cache_dir: Optional[str] = None,
    ) -> Graph:
        """从文本或对话消息生成知识图谱。

        Args:
            input_data: 文本字符串或消息字典列表
            model: 覆盖模型名
            api_key: 覆盖 API key
            api_base: 覆盖 API base URL
            context: 数据上下文描述
            chunk_size: 文本分块大小（字符数）
            reasoning_effort: 推理强度
            deduplication_method: 去重方式，None 表示不去重
            temperature: 采样温度
            output_folder: 若提供，则将图谱导出到该目录下的 graph.json
            no_dspy: 使用 litellm 原生 prompt 而非 dspy 编排
            cache_dir: 若提供，则将分块抽取结果缓存到该目录下的 llm_cache.json

        Returns:
            生成的知识图谱
        """
        is_conversation = isinstance(input_data, list)
        if is_conversation:
            text_content = []
            for message in input_data:
                if (
                    not isinstance(message, dict)
                    or "role" not in message
                    or "content" not in message
                ):
                    raise ValueError(
                        "Messages must be dicts with 'role' and 'content' keys"
                    )
                if message["role"] in ["user", "assistant"]:
                    text_content.append(f"{message['role']}: {message['content']}")
            processed_input = "\n".join(text_content)
        else:
            processed_input = input_data

        if any(
            x is not None
            for x in [model, temperature, api_key, api_base, reasoning_effort]
        ):
            self.init_model(
                model=model if model is not None else self.model,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                api_key=api_key if api_key is not None else self.api_key,
                api_base=api_base if api_base is not None else self.api_base,
                reasoning_effort=reasoning_effort
                if reasoning_effort is not None
                else self.reasoning_effort,
            )

        cache = _LLMCache(Path(cache_dir) / "llm_cache.json" if cache_dir else None)
        effective_temp = temperature if temperature is not None else self.temperature
        usage_before = self.extract_token_usage_from_history()

        def _extract(content):
            """抽取单个分块的实体与关系，优先命中磁盘缓存。"""
            entities = cache.get("entities", self.model, content)
            if entities is None:
                entities = get_entities(
                    content,
                    is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=effective_temp,
                )
                entities = entities[:MAX_ENTITIES_PER_CHUNK]
                cache.set("entities", self.model, list(entities), content)
            else:
                entities = entities[:MAX_ENTITIES_PER_CHUNK]

            entities_sig = json.dumps(entities, ensure_ascii=False)
            relations = cache.get("relations", self.model, content, entities_sig)
            if relations is None:
                relations = get_relations(
                    content,
                    entities,
                    is_conversation=is_conversation,
                    use_litellm_prompt=no_dspy,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    temperature=effective_temp,
                )
                cache.set(
                    "relations", self.model, [list(r) for r in relations],
                    content, entities_sig,
                )
            else:
                relations = [tuple(r) for r in relations]
            return entities, relations

        def _process(content, lm):
            with dspy.context(lm=lm):
                return _extract(content)

        if not chunk_size:
            try:
                entities, relations = _process(processed_input, self.lm)
            except Exception as e:
                if "context length" in str(e).lower():
                    logger.warning(
                        "Context length error: {}. Chunking text with chunk size 16384.",
                        e,
                    )
                    chunk_size = 16384
                else:
                    raise

        if chunk_size:
            chunks = chunk_text(processed_input, chunk_size)
            entities = set()
            relations = set()

            est = estimate_kg_cost(len(processed_input), len(chunks))
            logger.info(
                "分块抽取：{} 块，预估输入 {:,} tok / 输出 {:,} tok，约 ¥{}",
                len(chunks),
                est["input_tokens"],
                est["output_tokens"],
                est["cost_yuan"],
            )

            with tqdm(total=len(chunks), desc="分块抽取", unit="块") as pbar:
                with ThreadPoolExecutor() as executor:
                    future_to_chunk = {
                        executor.submit(_process, chunk, self.lm): i
                        for i, chunk in enumerate(chunks)
                    }
                    for future in as_completed(future_to_chunk):
                        chunk_entities, chunk_relations = future.result()
                        entities.update(chunk_entities)
                        relations.update(chunk_relations)
                        pbar.set_postfix(实体=len(entities), 关系=len(relations))
                        pbar.update(1)

        logger.info("LLM 缓存命中 {} 次", cache.hits)

        graph = Graph(
            entities=entities,
            relations=relations,
            edges={relation[1] for relation in relations},
        )

        if deduplication_method:
            graph = self.deduplicate(
                graph, method=deduplication_method, context=context
            )

        if output_folder:
            self.export_graph(graph, os.path.join(output_folder, "graph.json"))

        usage_after = self.extract_token_usage_from_history()
        prompt_delta = usage_after["prompt_tokens"] - usage_before["prompt_tokens"]
        completion_delta = (
            usage_after["completion_tokens"] - usage_before["completion_tokens"]
        )
        if prompt_delta or completion_delta:
            cost = (
                prompt_delta / 1_000_000 * KG_PRICE_INPUT_PER_MTOKEN
                + completion_delta / 1_000_000 * KG_PRICE_OUTPUT_PER_MTOKEN
            )
            logger.info(
                "本次生成实际用量：{:,} prompt / {:,} completion tok，约 ¥{:.2f}",
                prompt_delta,
                completion_delta,
                cost,
            )
        return graph

    def deduplicate(
        self,
        graph: Graph,
        method: DeduplicateMethod = DeduplicateMethod.FULL,
        semhash_similarity_threshold: float = 0.95,
        model: str = None,
        temperature: float = None,
        api_key: str = None,
        api_base: str = None,
        context: str = "",
    ) -> Graph:
        """对图谱去重。"""
        if any(x is not None for x in [model, temperature, api_key, api_base]):
            self.init_model(
                model=model if model is not None else self.model,
                temperature=temperature
                if temperature is not None
                else self.temperature,
                api_key=api_key if api_key is not None else self.api_key,
                api_base=api_base if api_base is not None else self.api_base,
            )

        return run_deduplication(
            lm=self.lm,
            graph=graph,
            method=method,
            retrieval_model=self.retrieval_model,
            semhash_similarity_threshold=semhash_similarity_threshold,
        )

    def cluster(self, graph: Graph, **kwargs) -> Graph:
        """已废弃，等价于 deduplicate。"""
        return self.deduplicate(graph, **kwargs)

    def aggregate(self, graphs: list[Graph]) -> Graph:
        """合并多个图谱。"""
        all_entities = set()
        all_relations = set()
        all_edges = set()
        all_entity_metadata: dict[str, set[str]] = {}

        for graph in graphs:
            all_entities.update(graph.entities)
            all_relations.update(graph.relations)
            all_edges.update(graph.edges)
            if graph.entity_metadata:
                for entity, metadata_set in graph.entity_metadata.items():
                    if entity in all_entity_metadata:
                        all_entity_metadata[entity].update(metadata_set)
                    else:
                        all_entity_metadata[entity] = metadata_set.copy()

        return Graph(
            entities=all_entities,
            relations=all_relations,
            edges=all_edges,
            entity_metadata=all_entity_metadata if all_entity_metadata else None,
        )

    @staticmethod
    def visualize(
        graph: Graph,
        output_path: str | None = None,
        open_in_browser: bool = False,
    ) -> Path:
        return visualize_kg.visualize(
            graph, output_path, open_in_browser=open_in_browser
        )

    # ====== 检索方法 ======

    def _parse_embedding_model(
        self, model: Optional[SentenceTransformer] = None
    ) -> Optional[SentenceTransformer]:
        if model is None:
            model = self.retrieval_model
        if model is None:
            raise ValueError("No retrieval model provided")
        return model

    @staticmethod
    def to_nx(graph: Graph) -> nx.DiGraph:
        G = nx.DiGraph()
        for entity in graph.entities:
            G.add_node(entity)
        for relation in graph.relations:
            source, rel, target = relation
            G.add_edge(source, target, relation=rel)
        return G

    def generate_embeddings(
        self,
        graph: Union[Graph, nx.DiGraph],
        model: Optional[SentenceTransformer] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        model = self._parse_embedding_model(model)
        if isinstance(graph, Graph):
            graph = self.to_nx(graph)

        node_embeddings = {node: model.encode(node).tolist() for node in graph.nodes}
        relation_embeddings = {
            rel: model.encode(rel).tolist()
            for rel in {edge[2]["relation"] for edge in graph.edges(data=True)}
        }
        return node_embeddings, relation_embeddings

    def retrieve(
        self,
        query: str,
        node_embeddings: dict[str, np.ndarray],
        graph: nx.DiGraph,
        model: Optional[SentenceTransformer] = None,
        k: int = 8,
        verbose: bool = False,
    ) -> tuple[list[tuple[str, float]], set[str], str]:
        model = self._parse_embedding_model(model)
        top_nodes = self.retrieve_relevant_nodes(query, node_embeddings, model, k)
        context = set()
        for node, _ in top_nodes:
            node_context = self.retrieve_context(node, graph)
            if verbose:
                logger.info("Context for node {}: {}", node, node_context)
            context.update(node_context)
        context_text = " ".join(context)
        if verbose:
            logger.info("Combined context: '{}'\n---", context_text)
        return top_nodes, context, context_text

    @staticmethod
    def retrieve_relevant_nodes(
        query: str,
        node_embeddings: dict[str, np.ndarray],
        model: SentenceTransformer,
        k: int = 8,
    ) -> list[tuple[str, float]]:
        query_embedding = model.encode(query).reshape(1, -1)
        similarities = []
        for node, embed in node_embeddings.items():
            target_embedding = np.array(embed).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, target_embedding)[0][0]
            similarities.append((node, similarity))
        similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def retrieve_context(node: str, graph: nx.DiGraph, depth: int = 2) -> list[str]:
        context = set()

        def explore_neighbors(current_node, current_depth):
            if current_depth > depth:
                return
            for neighbor in graph.neighbors(current_node):
                rel = graph[current_node][neighbor]["relation"]
                context.add(f"{current_node} {rel} {neighbor}.")
                explore_neighbors(neighbor, current_depth + 1)
            for neighbor in graph.predecessors(current_node):
                rel = graph[neighbor][current_node]["relation"]
                context.add(f"{neighbor} {rel} {current_node}.")
                explore_neighbors(neighbor, current_depth + 1)

        explore_neighbors(node, 1)
        return list(context)

    @staticmethod
    def export_graph(graph: Graph, output_path: str):
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        graph_dict = {
            "entities": list(graph.entities),
            "relations": list(graph.relations),
            "edges": list(graph.edges),
            "entity_clusters": {k: list(v) for k, v in graph.entity_clusters.items()}
            if graph.entity_clusters
            else None,
            "edge_clusters": {k: list(v) for k, v in graph.edge_clusters.items()}
            if graph.edge_clusters
            else None,
            "entity_metadata": {
                k: list(v) for k, v in graph.entity_metadata.items()
            }
            if graph.entity_metadata
            else None,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f, ensure_ascii=False, indent=2)

    # ====== Token 用量 ======

    def reset_token_usage(self):
        self.lm.history = []

    def extract_token_usage_from_history(self) -> Dict[str, int]:
        """从 dspy LM 历史中提取 token 用量。"""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        for entry in self.lm.history:
            if isinstance(entry, dict):
                usage = entry.get("usage") or entry.get("response", {}).get("usage")
                if usage:
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                    total_tokens += usage.get("total_tokens", 0)

        return {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
        }
