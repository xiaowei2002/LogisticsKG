"""langchain BaseChatModel 封装：复用 RAG/utils/llm.py 的 OpenAI 兼容客户端。

把自研的 OpenAICompatibleLLM 包装为 langchain 聊天模型，使其能直接参与
LCEL 链（prompt | model | parser），并支持 invoke / stream / batch 等标准接口。
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

from RAG.utils.llm import OpenAICompatibleLLM, get_llm

_ROLE_MAP = {"human": "user", "ai": "assistant", "system": "system"}


def _content_to_str(content: Any) -> str:
    """把 langchain 消息 content（str / list[block]）转成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, tuple) and len(block) == 2 and isinstance(block[1], str):
                parts.append(block[1])
        return "\n".join(parts)
    return str(content)


def _to_openai_messages(messages: Sequence[BaseMessage]) -> List[dict]:
    return [
        {"role": _ROLE_MAP.get(msg.type, "user"), "content": _content_to_str(msg.content)}
        for msg in messages
    ]


class ChatOpenAICompat(BaseChatModel):
    """对接任意 OpenAI 兼容接口（Ollama / vLLM / 云端 API）的 langchain 聊天模型。

    Args:
        llm: 已实例化的 OpenAICompatibleLLM；缺省时通过 get_llm() 从 config.yaml 构建。
    """

    model_name: str = Field(default="", description="底层模型名")
    temperature: float = Field(default=0.7, description="采样温度")
    llm: Any = Field(default=None, exclude=True, description="OpenAICompatibleLLM 实例")

    def __init__(self, llm: Optional[OpenAICompatibleLLM] = None, **kwargs: Any):
        if llm is None:
            llm = get_llm()
        kwargs.setdefault("model_name", llm.model)
        kwargs.setdefault("temperature", getattr(llm, "temperature", 0.7))
        super().__init__(llm=llm, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "openai-compatible"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = _to_openai_messages(messages)
        text = self.llm.chat(payload, stream=False, **kwargs) or ""
        message = AIMessage(content=text)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        payload = _to_openai_messages(messages)
        for delta in self.llm.chat(payload, stream=True, **kwargs):
            if not delta:
                continue
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta, chunk=chunk)
            yield chunk


def build_chat_model(**overrides: Any) -> ChatOpenAICompat:
    """从 config.yaml 构建 langchain 聊天模型，可传参覆盖。"""
    llm = get_llm(**overrides)
    return ChatOpenAICompat(llm=llm)


if __name__ == "__main__":
    model = build_chat_model()
    print("invoke ->", model.invoke("用一句话介绍你自己").content)
    print("stream ->", "".join(c.content for c in model.stream("讲个冷笑话")))
