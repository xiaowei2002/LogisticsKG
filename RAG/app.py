"""
物流领域大模型对话系统 Web UI（Gradio）。

两种问答模式一键切换，对比回答差异：
- 使用 RAG    ：检索增强回答（文档 RAG + GraphRAG 融合；merge 图谱未构建时自动退化为纯文档检索）
- 不使用 RAG  ：直接大模型回答（对照基线）
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from RAG.utils.config import load_config  # noqa: E402
from RAG.utils.service import (  # noqa: E402
    MODE_HYBRID,
    DomainService,
)

MODE_NONE = "none"  # 不使用 RAG 的对照模式

# 隐藏所有滚动条（保留鼠标滚轮/触摸滚动功能），并限制来源面板高度
CUSTOM_CSS = """
* { scrollbar-width: none !important; -ms-overflow-style: none !important; }
*::-webkit-scrollbar { width: 0; height: 0; display: none; }
.sources-box .wrap { max-height: 42vh; overflow: auto; }
"""

MODES: List[Tuple[str, str]] = [
    ("使用 RAG（检索增强）", MODE_HYBRID),
    ("不使用 RAG（直接大模型回答）", MODE_NONE),
]
MODE_LABELS: Dict[str, str] = {value: label for label, value in MODES}

service = DomainService()


def _msg_role(message: Any) -> str:
    return message.get("role") if isinstance(message, dict) else str(getattr(message, "role", "") or "")


def _msg_content(message: Any) -> str:
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(block["text"])
            elif hasattr(block, "text") and getattr(block, "text"):
                parts.append(getattr(block, "text"))
        return "\n".join(parts)
    return str(content or "")


def _bubble(text: str, label: str) -> Dict[str, Any]:
    return {"role": "assistant", "content": text, "metadata": {"title": label}}


def _no_rag_stream(question: str, prior: List[Tuple[str, str]]) -> Iterator[str]:
    """不使用 RAG：把对话历史 + 当前问题直接交给大模型。"""
    messages = []
    for role, content in prior:
        messages.append(AIMessage(content=content) if role == "assistant" else HumanMessage(content=content))
    messages.append(HumanMessage(content=question))
    for chunk in service.chat_model.stream(messages):
        if chunk.content:
            yield chunk.content


def respond(
    message: str, history: List[Any], mode: str
) -> Iterator[Tuple[List[Dict[str, Any]], Dict[str, Any] | None]]:
    """Gradio 事件回调：流式产出 (对话消息, 检索来源)。"""
    history = list(history or [])
    label = MODE_LABELS.get(mode, mode)
    history.append({"role": "user", "content": message, "metadata": {"title": f"提问 · {label}"}})

    # 除当前问题外的历史对话（供检索管线做多轮上下文）
    prior = [
        ("assistant" if _msg_role(m) == "assistant" else "human", _msg_content(m))
        for m in history[:-1]
        if _msg_content(m)
    ]

    partial = ""
    sources: Dict[str, Any] | None = (
        {"note": "未使用 RAG（直接大模型回答）", "sources": []} if mode == MODE_NONE else None
    )

    try:
        if mode == MODE_NONE:
            for token in _no_rag_stream(message, prior):
                partial += token
                yield history + [_bubble(partial, label)], sources
        else:
            # 使用 RAG：文档 RAG + GraphRAG 融合检索（merge 图谱未构建时自动退化为纯文档检索）
            for event in service.stream(message, MODE_HYBRID, prior):
                etype = event.get("type")
                if etype == "meta":
                    sources = {"mode": event["mode"], "sources": event["sources"]}
                elif etype == "token":
                    partial += event["text"]
                    yield history + [_bubble(partial, label)], sources
                elif etype == "error":
                    partial += f"\n❌ {event['message']}"
                    yield history + [_bubble(partial, label)], sources
    except Exception as exc:
        partial = (partial or "") + f"\n❌ {exc}"
    yield history + [_bubble(partial, label)], sources


def build_ui() -> gr.Blocks:
    app_cfg = getattr(service.cfg, "app", {}) or {}
    default_mode = getattr(app_cfg, "default_mode", MODE_HYBRID)
    if default_mode not in MODE_LABELS:
        default_mode = MODE_HYBRID

    with gr.Blocks(title="物流领域大模型问答（RAG）") as demo:
        gr.Markdown(
            "# 🚚 物流领域大模型问答\n"
            "**使用 RAG / 不使用 RAG 对比演示**：切换右侧「问答模式」，"
            "同一问题分别用检索增强与直接大模型回答，直观对比差异。\n"
            "*使用 RAG = 文档 RAG + GraphRAG 融合检索（未构建图谱时自动退化为纯文档检索）。*"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height="calc(100vh - 330px)", label="对话")
            with gr.Column(scale=2):
                mode = gr.Radio(
                    choices=MODES,
                    value=default_mode,
                    label="问答模式",
                    info="使用 RAG 时同时检索文档与知识图谱，融合资料作答",
                )
                sources_box = gr.JSON(
                    label="检索来源（文档片段 + 实体/三元组；不使用 RAG 为空）",
                    elem_classes=["sources-box"],
                )
        with gr.Row():
            question = gr.Textbox(
                placeholder="例如：什么是整车物流？仓储管理包括哪些环节？",
                label="问题",
                scale=4,
            )
            send = gr.Button("发送", variant="primary")
            clear = gr.ClearButton([question, chatbot, sources_box], value="清空对话")

        for handler in (question.submit, send.click):
            handler(respond, [question, chatbot, mode], [chatbot, sources_box]).then(
                lambda: "", None, [question]
            )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="物流领域大模型对话系统（RAG，Gradio UI）")
    parser.add_argument("--host", default=None, help="监听地址（缺省取 config app.host）")
    parser.add_argument("--port", type=int, default=None, help="监听端口（缺省取 config app.port）")
    parser.add_argument("--config", default=None, help="配置文件路径（缺省 RAG/config.yaml）")
    parser.add_argument("--build", action="store_true", help="启动前检查/构建 RAG、GraphRAG 索引")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()

    global service
    if args.config:
        service = DomainService(load_config(args.config))

    if args.build:
        print("[build] 检查/构建 RAG 索引（已缓存则秒级加载）...", flush=True)
        service.rag.ensure_index()
        if service.graph_available():
            print("[build] 检查/构建 GraphRAG 实体索引...", flush=True)
            service.graphrag.ensure_index()
        else:
            print("[build] 未找到 merge 知识图谱，跳过 GraphRAG 索引（将使用文档检索）", flush=True)

    app_cfg = getattr(service.cfg, "app", {}) or {}
    host = args.host or getattr(app_cfg, "host", "127.0.0.1")
    port = args.port or int(getattr(app_cfg, "port", 7860))

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)  # 串行处理，保证流式输出稳定
    print(f"🚚 物流领域大模型问答系统（Gradio）已启动: http://{host}:{port}", flush=True)
    print(
        f"   RAG 模式内部路由: {'GraphRAG（merge 图谱已构建）' if service.graph_available() else '文档检索（未构建图谱）'}",
        flush=True,
    )
    demo.launch(
        server_name=host,
        server_port=port,
        inbrowser=not args.no_browser,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
