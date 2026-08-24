from abc import ABC, abstractmethod
from typing import Any, Union, Generator
from openai import OpenAI
from RAG.utils.config import config

Message = dict
Messages = list


class BaseLLM(ABC):
    """大模型适配器统一基类"""

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.extra = kwargs

    @abstractmethod
    def chat(self, messages: Union[str, Message], *, stream: bool = True, **kwargs: Any) -> Union[
        str, Generator[str, None, None]]:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model})"


def _normalize_messages(messages: Messages) -> Messages:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    return list(messages)


class OpenAICompatibleLLM(BaseLLM):
    """适配所有兼容OpenAI接口的模型和本地部署模型"""

    def __init__(self, model: str, api_key: str, base_url: str, temperature: float = 0.7, **kwargs: Any):
        super().__init__(model, temperature, **kwargs)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, messages, *, stream=True, **kwargs):
        messages = _normalize_messages(messages)
        if not stream:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=False,
                **kwargs
            )

            return resp.choices[0].message.content
        return self._stream(messages, **kwargs)


    def _stream(self, messages: Messages, **kwargs):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
            **kwargs
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def get_llm(**overrides: Any) -> OpenAICompatibleLLM:
    if config is None:
        raise RuntimeError("未找到config，请确认config正确配置")

    try:
        llm_cfg = dict(config.llm)
    except AttributeError:
        raise RuntimeError("config中缺少llm配置节")

    # config.llm 可能直接是模型参数，也可能按 provider 嵌套（如 llm.ollama）
    providers = {k: v for k, v in llm_cfg.items() if isinstance(v, dict)}
    if providers:
        llm_cfg = dict(next(iter(providers.values())))

    llm_cfg.update(overrides)
    return OpenAICompatibleLLM(**llm_cfg)



if __name__ == "__main__":
    llm = get_llm()
    print(llm.chat("用一句话介绍你自己", stream=False))

    print("\n--- 流式输出 ---")
    for piece in llm.chat("讲个笑话", stream=True):
        print(piece, end="")
    print()
