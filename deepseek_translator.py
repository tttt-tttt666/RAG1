"""On-demand medical translation using the DeepSeek OpenAI-compatible API."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def api_is_configured() -> bool:
    """Return whether a non-empty DeepSeek API key is available."""
    return bool(os.getenv("DEEPSEEK_API_KEY", "").strip())


def translate_to_chinese(text: str) -> str:
    """Translate an English medical passage without adding medical advice."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("尚未配置 DEEPSEEK_API_KEY，请先填写项目根目录中的 .env")

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )
    response = client.chat.completions.create(
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
