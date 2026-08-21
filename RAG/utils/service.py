"""领域问答服务编排：统一封装 RAG、GraphRAG 与二者融合（hybrid）。

路由规则：
- mode = "hybrid":   文档 RAG + GraphRAG 同时检索、融合上下文作答（merge 图谱未构建时退化为纯文档 RAG）
- mode = "rag":      仅文档 RAG
- mode = "graphrag": 仅 GraphRAG（需要 merge 知识图谱存在）
- mode = "auto":     merge 知识图谱已构建 → GraphRAG；未构建 → 回退文档 RAG
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from RAG.utils.chat_model import ChatOpenAICompat
from RAG.utils.config import config
from RAG.utils.graphrag import GraphRAGEngine
from RAG.utils.llm import OpenAICompatibleLLM
from RAG.utils.rag import RAGEngine, RAG_DIR

MODE_AUTO = "auto"
MODE_RAG = "rag"
MODE_GRAPHRAG = "graphrag"
MODE_HYBRID = "hybrid"  # 文档 RAG + GraphRAG 融合

_HYBRID_SYSTEM = """你是物流领域知识问答助手，请依据【参考资料 · 文档片段】和【知识图谱三元组】回答用户问题。
要求：
1. 只依据给定资料作答，可结合文档片段与三元组中的实体关系进行推理，不要编造；
2. 若资料不足以回答问题，请明确说明"资料中未找到相关信息"；
3. 用中文回答，条理清晰、简洁准确。"""

_NO_GRAPH_CONTEXT = "（未构建 merge 知识图谱，本部分为空）"


@dataclass
class AnswerResult:
    mode: str
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DomainService:
    """领域大模型服务：RAG / GraphRAG / 融合 统一入口。"""

    def __init__(self, cfg: Any = None):
        self.cfg = cfg if cfg is not None else config
        self._rag: Optional[RAGEngine] = None
        self._graphrag: Optional[GraphRAGEngine] = None
        self._graph_path: Optional[Path] = None
        self._chat_model: Optional[ChatOpenAICompat] = None
        self._hybrid_chain = None

    # ---------- 引擎懒加载 ----------

    @property
    def chat_model(self) -> ChatOpenAICompat:
        """按本服务的 cfg 构建聊天模型（支持 --config 自定义配置文件）。"""
        if self._chat_model is None:
            llm_cfg = getattr(self.cfg, "llm", {}) or {}
            providers = {k: v for k, v in dict(llm_cfg).items() if isinstance(v, dict)}
            params = dict(next(iter(providers.values()))) if providers else dict(llm_cfg)
            self._chat_model = ChatOpenAICompat(llm=OpenAICompatibleLLM(**params))
        return self._chat_model

    @property
    def rag(self) -> RAGEngine:
        if self._rag is None:
            self._rag = RAGEngine(getattr(self.cfg, "rag", {}) or {}, chat_model=self.chat_model)
        return self._rag

    @property
    def graphrag(self) -> GraphRAGEngine:
        if self._graphrag is None:
            self._graphrag = GraphRAGEngine(
                getattr(self.cfg, "graphrag", {}) or {}, chat_model=self.chat_model
            )
        return self._graphrag

    @property
    def graph_path(self) -> Path:
        if self._graph_path is None:
            graphrag_cfg = getattr(self.cfg, "graphrag", {}) or {}
            p = Path(getattr(graphrag_cfg, "graph_path", "../output/merged_graph.json"))
            self._graph_path = p if p.is_absolute() else RAG_DIR / p
        return self._graph_path

    def graph_available(self) -> bool:
        """merge 知识图谱是否已构建。"""
        enabled = getattr(getattr(self.cfg, "graphrag", {}) or {}, "enabled", True)
        return bool(enabled) and self.graph_path.is_file()

    # ---------- 路由 ----------

    def resolve_mode(self, mode: str) -> str:
        mode = (mode or MODE_AUTO).strip().lower()
        if mode == MODE_AUTO:
            return MODE_GRAPHRAG if self.graph_available() else MODE_RAG
        if mode == MODE_HYBRID:
            return MODE_HYBRID
        if mode == MODE_GRAPHRAG:
            if not self.graph_available():
                raise RuntimeError(
                    f"merge 知识图谱不存在: {self.graph_path}，请改用 rag 模式"
                )
            return MODE_GRAPHRAG
        if mode == MODE_RAG:
            return MODE_RAG
        raise ValueError(f"未知模式: {mode}（可选 auto / rag / graphrag / hybrid）")

    def _engine(self, mode: str):
        engine = self.graphrag if mode == MODE_GRAPHRAG else self.rag
        engine.ensure_index()  # 首次调用自动构建/加载索引
        return engine

    # ---------- 融合（hybrid）：文档 RAG + GraphRAG ----------

    @property
    def hybrid_chain(self):
        if self._hybrid_chain is None:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _HYBRID_SYSTEM),
                    ("system", "【参考资料 · 文档片段】\n{context}"),
                    ("system", "【知识图谱三元组】\n{graph_context}"),
                    MessagesPlaceholder(variable_name="history", optional=True),
                    ("human", "{question}"),
                ]
            )
            self._hybrid_chain = prompt | self.chat_model | StrOutputParser()
        return self._hybrid_chain

    def _hybrid_retrieve(
        self, question: str, history: Optional[List[Tuple[str, str]]] = None
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """同时检索文档与知识图谱，融合为链的输入，并汇总两类来源。

        Returns:
            (chain_inputs, sources)：chain_inputs 含 context/graph_context/history/question。
        """
        docs = self.rag.retrieve(question)
        sources: List[Dict[str, Any]] = self.rag._sources(docs)
        doc_context = self.rag._context(docs)

        graph_context = _NO_GRAPH_CONTEXT
        if self.graph_available():
            triples, entities = self.graphrag.retrieve(question)
            graph_context = self.graphrag._context(triples)
            sources += [
                {"type": "entity", "name": e["name"], "score": e["score"]} for e in entities
            ] + [
                {
                    "type": "triple",
                    "text": f"{t['subject']} -[{t['relation']}]-> {t['object']}",
                    "score": t["score"],
                }
                for t in triples
            ]

        inputs = {
            "context": doc_context,
            "graph_context": graph_context,
            "history": list(history or []),
            "question": question,
        }
        return inputs, sources

    def _ensure_hybrid_ready(self) -> None:
        self.rag.ensure_index()
        if self.graph_available():
            self.graphrag.ensure_index()

    # ---------- 对外接口 ----------

    def status(self) -> Dict[str, Any]:
        """供 UI 展示的状态信息。"""
        default_mode = getattr(getattr(self.cfg, "app", {}) or {}, "default_mode", MODE_AUTO)
        return {
            "default_mode": default_mode,
            "graph_available": self.graph_available(),
            "graph_path": str(self.graph_path),
            "rag_stats": self.rag.stats(),
            "graphrag_stats": self.graphrag.stats(),
        }

    def answer(
        self,
        question: str,
        mode: str = MODE_AUTO,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> AnswerResult:
        resolved = self.resolve_mode(mode)

        if resolved == MODE_HYBRID:
            self._ensure_hybrid_ready()
            inputs, sources = self._hybrid_retrieve(question, history)
            answer = self.hybrid_chain.invoke(inputs)
            return AnswerResult(
                mode=resolved,
                answer=answer,
                sources=sources,
                stats={"engine": "hybrid", **self.rag.stats()},
            )

        engine = self._engine(resolved)
        if resolved == MODE_GRAPHRAG:
            triples, entities = engine.retrieve(question)
            sources = [
                {"type": "entity", "name": e["name"], "score": e["score"]} for e in entities
            ] + [
                {
                    "type": "triple",
                    "text": f"{t['subject']} -[{t['relation']}]-> {t['object']}",
                    "score": t["score"],
                }
                for t in triples
            ]
        else:
            docs = engine.retrieve(question)
            sources = engine._sources(docs)
        answer = engine.answer(question, history)
        return AnswerResult(mode=resolved, answer=answer, sources=sources, stats=engine.stats())

    def stream(
        self,
        question: str,
        mode: str = MODE_AUTO,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """流式问答：先发 status / meta（模式 + 来源），再逐 token 推送，最后 done。"""
        resolved = self.resolve_mode(mode)

        if resolved == MODE_HYBRID:
            yield {
                "type": "status",
                "message": "正在准备 RAG + GraphRAG 引擎（首次构建索引较慢，请稍候）...",
            }
            self._ensure_hybrid_ready()
            inputs, sources = self._hybrid_retrieve(question, history)
            yield {
                "type": "meta",
                "mode": resolved,
                "sources": sources,
                "stats": {"engine": "hybrid", **self.rag.stats()},
            }
            for token in self.hybrid_chain.stream(inputs):
                yield {"type": "token", "text": token}
            yield {"type": "done"}
            return

        yield {
            "type": "status",
            "message": f"正在准备 {resolved.upper()} 引擎（首次构建索引较慢，请稍候）...",
        }
        engine = self._engine(resolved)

        if resolved == MODE_GRAPHRAG:
            triples, entities = engine.retrieve(question)
            sources = [
                {"type": "entity", "name": e["name"], "score": e["score"]} for e in entities
            ] + [
                {
                    "type": "triple",
                    "text": f"{t['subject']} -[{t['relation']}]-> {t['object']}",
                    "score": t["score"],
                }
                for t in triples
            ]
        else:
            docs = engine.retrieve(question)
            sources = engine._sources(docs)

        yield {
            "type": "meta",
            "mode": resolved,
            "sources": sources,
            "stats": engine.stats(),
        }
        for token in engine.stream(question, history):
            yield {"type": "token", "text": token}
        yield {"type": "done"}
