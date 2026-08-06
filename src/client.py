import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

class Client:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        thinking: bool = False,
        max_tokens: int = 32768,
    ):
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME") or "qwen3.6-27B"
        self.thinking = thinking
        self.max_tokens = max_tokens

        # 根据 thinking 模式设置推荐采样参数
        if thinking:
            self.temperature = 1.0
            self.top_p = 0.95
            self.top_k = 20
            self.min_p = 0.0
            self.presence_penalty = 0.0
            self.repetition_penalty = 1.0
        else:
            self.temperature = 0.7
            self.top_p = 0.80
            self.top_k = 20
            self.min_p = 0.0
            self.presence_penalty = 1.5
            self.repetition_penalty = 1.0

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置，请配置环境变量")

        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

    def chat(
        self,
        messages: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
        stream: bool = True,
        **kwargs,
    ) -> str:
        extra_body = kwargs.pop("extra_body", {})
        if self.thinking:
            extra_body.setdefault("enable_thinking", True)
        extra_body.setdefault("top_k", kwargs.pop("top_k", self.top_k))
        extra_body.setdefault("min_p", kwargs.pop("min_p", self.min_p))
        extra_body.setdefault("repetition_penalty", kwargs.pop("repetition_penalty", self.repetition_penalty))
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages or [],
            max_tokens=max_tokens or self.max_tokens,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            presence_penalty=presence_penalty if presence_penalty is not None else self.presence_penalty,
            stream=stream,
            **kwargs,
        )

        if stream:
            full_text = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_text += text
            print()
            return full_text

        return response.choices[0].message.content or ""

if __name__ == "__main__":
    client = Client()
    # 文本输入
    messages = [
        {"role": "user", "content": "请介绍下汽车物流服务的评价指标"}
    ]

    result = client.chat(messages=messages)
    print(result)

    # 图像输入
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
                }
            },
            {
                "type": "text",
                "text": "The centres of the four illustrated circles are in the corners of the square. The two big circles touch each other and also the two little circles. With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\nChoices:\n(A) $\\frac{2}{9}$\n(B) $\\sqrt{5}$\n(C) $0.8 \\cdot \\pi$\n(D) 2.5\n(E) $1+\\sqrt{2}$"
            }
        ]
    }]
    result = client.chat(messages=messages)
    print(result)

