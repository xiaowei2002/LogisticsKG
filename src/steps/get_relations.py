import json
from pathlib import Path
from typing import List, Tuple, Optional, Literal, Type

import dspy
import litellm
from pydantic import BaseModel, create_model, ValidationError


def parse_relations_response(
        raw_json: str,
        entities: List[str],
        response_model: Optional[Type[BaseModel]] = None,
) -> List[Tuple[str, str, str]]:
    """
    首先尝试严格的 Pydantic 校验。若失败（如 EntityLiteral 校验失败），
    退化为原始 JSON 解析，并过滤掉主语/宾语无效的条目。
    Args:
        raw_json: LLM 响应的原始 JSON 字符串
        entities: 有效实体字符串列表
        response_model: 用于严格校验的可选 Pydantic 模型
    Returns:
        (主语, 谓词, 宾语) 元组列表，主语和宾语均为有效实体
    """
    entities_set = set(entities)

    # 若提供模型，先尝试严格 Pydantic 校验
    if response_model is not None:
        try:
            parsed = response_model.model_validate_json(raw_json)
            return [(r.subject, r.predicate, r.object) for r in parsed.relations]
        except ValidationError:
            pass  # 降级到原始 JSON 解析

    # 降级：按原始 JSON 解析并过滤
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return []

    # 同时处理 {"relations": [...]} 和直接列表两种格式
    items = data.get("relations", data) if isinstance(data, dict) else data

    if not isinstance(items, list):
        return []

    relations = []
    for item in items:
        if not isinstance(item, dict):
            continue

        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")

        # 缺少必填字段则跳过
        if not all([subject, predicate, obj]):
            continue

        # 主语或宾语不在有效实体列表中则跳过
        if subject not in entities_set or obj not in entities_set:
            continue

        relations.append((subject, predicate, obj))

    return relations


def _load_relations_prompt() -> str:
    """从文件加载关系抽取 prompt 模板。"""
    prompt_path = Path(__file__).parent.parent / "prompts" / "relations.txt"
    return prompt_path.read_text(encoding="utf-8")


def _create_relations_model(entities: List[str]):
    """动态创建带实体字面量约束的 Pydantic 模型。"""
    # 从实体列表创建 Literal 类型
    EntityLiteral = Literal[tuple(entities)]  # type: ignore

    # 创建带主语/宾语约束的 RelationItem
    RelationItem = create_model(
        "RelationItem",
        subject=(EntityLiteral, ...),
        predicate=(str, ...),
        object=(EntityLiteral, ...),
    )

    # 创建包含关系列表的 RelationsResponse
    RelationsResponse = create_model(
        "RelationsResponse",
        relations=(List[RelationItem], ...),
    )

    return RelationItem, RelationsResponse


def _get_relations_litellm(
        input_data: str,
        entities: List[str],
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
) -> List[Tuple[str, str, str]]:
    prompt_template = _load_relations_prompt()
    entities_str = "\n".join(f"- {e}" for e in entities)
    user_prompt = f"""
以下是从源文本中先前提取到的实体列表：

<entities>
{entities_str}
</entities>

以下是待分析的源文本：

<text>
{input_data}
</text>
"""

    # 创建带实体约束的动态模型
    _, RelationsResponse = _create_relations_model(entities)

    # 构建 schema，additionalProperties: false（OpenAI 要求）
    schema = RelationsResponse.model_json_schema()
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
                "name": "relations_response",
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
    return parse_relations_response(raw_json, entities, RelationsResponse)


def extraction_sig(
        Relation: BaseModel, is_conversation: bool, context: str = ""
) -> dspy.Signature:
    if not is_conversation:

        class ExtractTextRelations(dspy.Signature):
            __doc__ = f"""从源文本中抽取主语-谓词-宾语三元组。
主语和宾语必须来自实体列表。实体列表是预先从同一源文本中抽取的。
这是抽取任务，请务必详尽、准确，并忠实于参考文本。{context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="主语-谓词-宾语元组列表。请务必详尽。"
            )

        return ExtractTextRelations
    else:

        class ExtractConversationRelations(dspy.Signature):
            __doc__ = f"""从对话中抽取主语-谓词-宾语三元组，包括：
1. 讨论的概念之间的关系
2. 说话者与概念之间的关系（如用户询问 X）
3. 说话者之间的关系（如助手回复用户）
主语和宾语必须来自实体列表。实体列表是预先从同一源文本中抽取的。
这是抽取任务，请务必详尽、准确，并忠实于参考文本。{context}"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.OutputField(
                desc="主语-谓词-宾语元组列表，主语和宾语必须精确匹配实体列表中的条目。请务必详尽。"
            )

        return ExtractConversationRelations


def fallback_extraction_sig(
        entities, is_conversation, context: str = ""
) -> dspy.Signature:
    """此降级抽取不对主语和宾语字符串做严格类型约束。"""

    entities_str = "\n- ".join(entities)

    class Relation(BaseModel):
        __doc__ = f"""知识图谱主语-谓词-宾语元组。主语和宾语实体必须是以下之一：{entities_str}"""

        subject: str = dspy.InputField(desc="主语实体", examples=["整车物流"])
        predicate: str = dspy.InputField(desc="谓词", examples=["属于"])
        object: str = dspy.InputField(desc="宾语实体", examples=["物流"])

    return Relation, extraction_sig(Relation, is_conversation, context)


def _filter_entities(entities: List[str]) -> List[str]:
    """过滤掉包含引号的实体。"""
    return [e for e in entities if '"' not in e]


def get_relations(
        input_data: str,
        entities: list[str],
        is_conversation: bool = False,
        context: str = "",
        use_litellm_prompt: bool = False,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
) -> List[Tuple[str, str, str]]:
    # 过滤掉包含引号的实体
    entities = _filter_entities(entities)

    if use_litellm_prompt and not is_conversation:
        return _get_relations_litellm(
            input_data,
            entities,
            model=model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
        )

    class Relation(BaseModel):
        """知识图谱主语-谓词-宾语元组。"""

        subject: str = dspy.InputField(desc="主语实体", examples=["整车物流"])
        predicate: str = dspy.InputField(desc="谓词", examples=["属于"])
        object: str = dspy.InputField(desc="宾语实体", examples=["物流"])

    ExtractRelations = extraction_sig(Relation, is_conversation, context)

    try:
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)
        return [(r.subject, r.predicate, r.object) for r in result.relations]

    except Exception:
        Relation, ExtractRelations = fallback_extraction_sig(
            entities, is_conversation, context
        )
        extract = dspy.Predict(ExtractRelations)
        result = extract(source_text=input_data, entities=entities)

        class FixedRelations(dspy.Signature):
            """修正关系，使每个关系的主语和宾语都精确匹配实体列表中的条目。保持谓词不变。每个关系的含义应忠实于参考文本。若无法保持原始关系相对源文本的含义，则不要返回该关系。"""

            source_text: str = dspy.InputField()
            entities: list[str] = dspy.InputField()
            relations: list[Relation] = dspy.InputField()
            fixed_relations: list[Relation] = dspy.OutputField()

        fix = dspy.ChainOfThought(FixedRelations)

        fix_res = fix(
            source_text=input_data, entities=entities, relations=result.relations
        )

        good_relations = []
        for rel in fix_res.fixed_relations:
            if rel.subject in entities and rel.object in entities:
                good_relations.append(rel)
        return [(r.subject, r.predicate, r.object) for r in good_relations]


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from get_entities import get_entities

    load_dotenv()

    input_data = "汽车售后服务备件配送中心是指从事汽车售后服务备件配送业务且具有完善信息系统的组织及场所。"

    model = "dashscope/" + os.getenv("OPENAI_MODEL_NAME")
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL")
    entities = get_entities(input_data, model=model, use_litellm_prompt=True, api_key=api_key, api_base=api_base)

    result = get_relations(
        input_data,
        entities,
        model=model,
        use_litellm_prompt=True,
        api_key=api_key,
        api_base=api_base,
    )
    print(result)
