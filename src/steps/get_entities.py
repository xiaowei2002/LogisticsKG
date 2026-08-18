import os
import dspy
import litellm
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv


class TextEntities(dspy.Signature):
    """Extract key entities from the source text. Extracted entities are subjects or objects.
    This is for an extraction task, please be THOROUGH and accurate to the reference text."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class ConversationEntities(dspy.Signature):
    """Extract key entities from the conversation. Extracted entities are subjects or objects.
    Consider both explicit entities and participants in the conversation.
    This is for an extraction task, please be THOROUGH and accurate."""

    source_text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField(desc="THOROUGH list of key entities")


class EntitiesResponse(BaseModel):
    """Structured response for entity extraction."""

    entities: List[str]


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
    parsed = EntitiesResponse.model_validate_json(response.output[-1].content[0].text)
    return parsed.entities


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
