import json
import os
import dspy
import litellm
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv


class TextEntities(dspy.Signature):
    """从源文本中抽取关键实体，用于构建知识图谱。仅抽取文档主题相关、且可能与其他实体存在关系的实体。宁缺毋滥，聚焦而精简，通常不超过 80 个。避免过于泛化的词（如"运输""物流"），除非它在文档中被明确定义。"""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="关键实体列表，宁缺毋滥，聚焦精简")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation. Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntityItem(BaseModel):
    """单个实体：名称与类别。"""

    name: str
    type: str


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[EntityItem]


def parse_entities_response(raw_json: str) -> List[str]:
    """解析实体抽取响应，兼容多种格式。

    优先严格校验，失败时降级为宽松解析。支持：
    - {"entities": [{"name", "type"}]}
    - {"entities": ["..."]}
    - [{"name", "type"}]
    - ["..."]
    """
    try:
        parsed = EntitiesResponse.model_validate_json(raw_json)
        return [e.name for e in parsed.entities]
    except ValidationError:
        pass

    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return []

    if isinstance(data, dict):
        data = data.get("entities", [])

    if not isinstance(data, list):
        return []

    names = []
    for item in data:
        if isinstance(item, str):
            name = item
        elif isinstance(item, dict):
            name = item.get("name") or item.get("entity")
        else:
            continue
        if name and isinstance(name, str):
            names.append(name.strip())
    return names


def _load_entities_prompt() -> str:
    """Load the entities prompt template from file."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "entities.txt"
    return prompt_path.read_text()


def _get_entities_litellm(
        input_data: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
) -> List[str]:
    prompt_template = _load_entities_prompt()
    user_prompt = f"""
以下是要从中提取实体的文本:

<article>
{input_data}
</article>
    """

    schema = EntitiesResponse.model_json_schema()
    schema["additionalProperties"] = False
    # 同时设置嵌套对象的 additionalProperties
    if "$defs" in schema:
        for def_schema in schema["$defs"].values():
            if def_schema.get("type") == "object":
                def_schema["additionalProperties"] = False

    kwargs = {
        "model": model,
        "input": [
            {"role": "system", "content": prompt_template},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "entities_response",
                "schema": schema,
                "strict": True,
            }
        },
    }

    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    response = litellm.responses(**kwargs)
    raw_json = response.output[-1].content[0].text
    return parse_entities_response(raw_json)


def get_entities(
        input_data: str,
        is_conversation: bool = False,
        use_litellm_prompt: bool = False,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
) -> List[str]:
    if use_litellm_prompt and not is_conversation:
        return _get_entities_litellm(
            input_data,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
        )

    extract = (
        dspy.Predict(ConversationEntities)
        if is_conversation
        else dspy.Predict(TextEntities)
    )
    result = extract(source_text=input_data)
    return result.entities


if __name__ == "__main__":
    input_data = "汽车售后服务备件配送中心是指从事汽车售后服务备件配送业务且具有完善信息系统的组织及场所。"

    load_dotenv()
    model = "dashscope/" + os.getenv("OPENAI_MODEL_NAME")
    print(model)
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")
    result = get_entities(input_data, model=model, use_litellm_prompt=True, api_key=api_key, api_base=api_base)
    print(result)
