"""On-demand medical translation using the DeepSeek OpenAI-compatible API."""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def api_is_configured() -> bool:
    """Return whether a non-empty DeepSeek API key is available."""
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def _client() -> OpenAI:
    """Create a DeepSeek client without exposing the API key."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("尚未配置 DEEPSEEK_API_KEY，请先填写项目根目录中的 .env")
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )


def assess_question_scope(question: str) -> dict:
    """Decide whether the ankle-sprain RAG assistant should answer a question."""
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是脚踝扭伤患者教育RAG系统的问题准入分类器。"
                    "你只负责判断问题是否应进入检索与回答流程，不回答医学问题。"
                    "允许范围：脚踝扭伤的受伤机制、症状、严重程度、就医或影像检查、"
                    "早期处理、治疗、康复训练、训练负荷、恢复时间、护具贴扎、"
                    "重返走路跑步运动、疗效、复发预防和危险信号。"
                    "拒绝范围：饮食或吃什么、一般营养、与脚踝扭伤无关的问题、"
                    "仅包含脚踝关键词但实际意图不属于上述患者教育资料的问题。"
                    "如果问题同时包含允许范围和无关问题，只有当核心意图能够由允许范围"
                    "直接回答时才允许。危险信号和就医问题必须允许。"
                    "同时识别问题表达方式：yes_no（是否类）、how_to（怎么做类）、"
                    "what_is（是什么或定义类）、when（何时或多久类）、"
                    "comparison（比较区别类）、multi_part（确有多个独立问题）或 other。"
                    "只输出JSON对象，不要输出Markdown。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "判断下面的问题是否应该由脚踝扭伤资料助手回答。"
                    '输出格式：{"should_answer":true或false,'
                    '"category":"简短主题类别","question_type":"上述英文类型之一",'
                    '"reason":"简短中文原因"}。\n\n'
                    f"问题：{question}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("DeepSeek 返回了空的准入判断")
    result = json.loads(content)
    if not isinstance(result.get("should_answer"), bool):
        raise RuntimeError("DeepSeek 准入判断缺少布尔值 should_answer")
    allowed_question_types = {
        "yes_no",
        "how_to",
        "what_is",
        "when",
        "comparison",
        "multi_part",
        "other",
    }
    question_type = str(result.get("question_type", "other")).strip()
    if question_type not in allowed_question_types:
        question_type = "other"
    return {
        "should_answer": result["should_answer"],
        "category": str(result.get("category", "未分类")).strip(),
        "question_type": question_type,
        "reason": str(result.get("reason", "未提供原因")).strip(),
        "source": "DeepSeek API",
    }


def assess_ankle_risk(question: str) -> dict:
    """Classify a possible ankle-injury warning signal without diagnosing it."""
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是脚踝损伤患者教育系统的风险分级器，不进行诊断。"
                    "只有在文本已经命中潜在危险症状同义词后才会调用你。"
                    "将风险分为：emergency（明显变形、开放伤口、足部发冷发紫、"
                    "感觉丧失等可能威胁肢体或需要立即急诊的表现）；"
                    "urgent_review（不能负重或走四步、严重或持续加重的疼痛肿胀等，"
                    "可先保护、停止运动和减少负重，但应尽快或当日医疗评估）；"
                    "self_care（语境并未描述当前危险症状，可提供一般处理及观察建议）。"
                    "必须结合完整语义，区分否定、假设、询问定义和患者正在出现症状。"
                    "不得因为单个关键词自动判为 emergency，也不得把明确的 emergency "
                    "表现降级。只输出JSON对象，不要输出Markdown。"
                ),
            },
            {
                "role": "user",
                "content": (
                    '输出格式：{"risk_level":"emergency、urgent_review或self_care",'
                    '"reason":"简短中文原因","immediate_action":"一条安全的立即措施"}。\n\n'
                    f"问题：{question}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("DeepSeek 返回了空的风险判断")
    result = json.loads(content)
    risk_level = str(result.get("risk_level", "")).strip()
    if risk_level not in {"emergency", "urgent_review", "self_care"}:
        raise RuntimeError("DeepSeek 风险判断包含无效的 risk_level")
    return {
        "risk_level": risk_level,
        "reason": str(result.get("reason", "未提供原因")).strip(),
        "immediate_action": str(result.get("immediate_action", "")).strip(),
        "source": "DeepSeek API",
    }


def translate_to_chinese(text: str) -> str:
    """Translate an English medical passage without adding medical advice."""
    response = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是严谨的医学患者教育资料翻译助手。"
                    "你的任务仅限忠实翻译，不提供诊断或治疗建议。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请将下面的英文脚踝扭伤资料准确翻译成简体中文。"
                    "保留全部医学含义、数字、分级、警告和条件；"
                    "不要概括、解释、改写或添加原文没有的内容。"
                    "只输出中文译文。\n\n英文原文：\n"
                    f"{text}"
                ),
            },
        ],
        stream=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    translated = response.choices[0].message.content
    if not translated or not translated.strip():
        raise RuntimeError("DeepSeek 返回了空译文")
    return translated.strip()
