#!/usr/bin/env python3
"""Minimal local Streamlit interface for ankle-sprain RAG retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index" / "ankle_sprain"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings" / "embeddings.npz"
EMBEDDING_METADATA_PATH = INDEX_DIR / "embeddings" / "metadata.json"

WARNING_TERMS = (
    "畸形",
    "无法负重",
    "不能走",
    "麻木",
    "失去知觉",
    "发紫",
    "发冷",
    "剧烈疼痛",
    "呼吸困难",
    "高烧",
    "deformity",
    "cannot walk",
    "unable to bear weight",
    "numb",
    "cold foot",
    "blue foot",
    "severe pain",
    "difficulty breathing",
)


@st.cache_data
def load_index() -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    chunks = [
        json.loads(line)
        for line in CHUNKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    archive = np.load(EMBEDDINGS_PATH, allow_pickle=False)
    embeddings = archive["embeddings"]
    chunk_ids = archive["chunk_ids"]
    metadata = json.loads(EMBEDDING_METADATA_PATH.read_text(encoding="utf-8"))

    expected_ids = [chunk["chunk_id"] for chunk in chunks]
    if chunk_ids.tolist() != expected_ids:
        raise ValueError("Embedding order does not match chunks.jsonl")
    return chunks, embeddings, chunk_ids, metadata


@st.cache_resource
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[tuple[float, dict]]:
    query_vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scores = embeddings @ query_vector
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(float(scores[index]), chunks[index]) for index in top_indices]


def contains_warning_term(query: str) -> bool:
    normalized = query.casefold()
    return any(term.casefold() in normalized for term in WARNING_TERMS)


def generate_short_chinese_answer(query: str, warning: bool) -> str:
    """Return a conservative Chinese summary from predefined medical templates."""
    if warning:
        return (
            "你的描述包含需要专业评估的危险信号。请停止运动，并尽快联系医生、"
            "急诊或当地紧急医疗服务。不要仅依赖本工具判断伤情。"
        )

    normalized = query.casefold()
    intent_templates = (
        (
            ("48 hour", "first day", "early treatment", "刚受伤", "早期", "前48"),
            "受伤早期应保护脚踝、适当休息、抬高患肢，并隔着毛巾短时间冷敷。"
            "在疼痛允许的范围内逐渐进行轻柔活动。",
        ),
        (
            ("exercise", "movement", "range of motion", "锻炼", "训练", "活动度"),
            "可从轻柔的脚踝屈伸和画圈开始，再逐步加入小腿拉伸、提踵和单脚平衡训练。"
            "如果练习明显加重疼痛或肿胀，应暂停并咨询专业人员。",
        ),
        (
            ("balance", "stability", "prevent", "平衡", "稳定", "预防"),
            "恢复期可逐步进行单脚站立、提踵及力量训练，以改善稳定性并降低再次扭伤风险。"
            "训练难度应循序渐进。",
        ),
        (
            ("walk", "weight bearing", "running", "sport", "return", "走路", "负重", "跑步", "运动"),
            "疼痛允许时可逐渐恢复负重和正常步行。只有在走路舒适、疼痛和肿胀明显改善后，"
            "再分阶段恢复慢跑及运动。",
        ),
        (
            ("heal", "recovery", "how long", "恢复", "多久", "愈合"),
            "轻度扭伤通常会在数周内改善，但完全恢复可能需要约六周；"
            "较严重的扭伤可能需要更长时间。恢复速度应以疼痛、肿胀和功能变化为参考。",
        ),
        (
            ("ice", "cold", "swelling", "冰敷", "冷敷", "肿胀"),
            "冷敷时应使用毛巾隔开皮肤，并配合抬高患肢。不要把冰直接放在皮肤上；"
            "如果肿胀持续、加重或伴随感觉异常，应寻求医疗评估。",
        ),
        (
            ("medical help", "doctor", "hospital", "seek help", "医生", "医院", "就医"),
            "如果不能正常负重、疼痛或肿胀没有改善、症状持续加重，"
            "或脚部出现麻木、发冷、明显变色，应及时接受专业医疗评估。",
        ),
    )
    for terms, answer in intent_templates:
        if any(term in normalized for term in terms):
            return answer

    return (
        "现有资料建议根据疼痛和肿胀情况循序渐进地恢复活动，并避免强行训练。"
        "请结合下方英文原文核对；如果症状持续或加重，应咨询医生或物理治疗师。"
    )


st.set_page_config(
    page_title="脚踝康复资料助手",
    page_icon="🦶",
    layout="centered",
)

st.title("脚踝康复资料助手")
st.caption("基于 32 份医院、政府卫生机构和专业医学组织可信资料的本地语义检索 Demo")

st.info(
    "本工具仅提供健康教育资料检索，不能诊断伤情或替代医生。"
    "当前模型更适合英文问题，建议优先使用英文提问。"
)

try:
    chunks, embeddings, _, embedding_metadata = load_index()
    model = load_model(embedding_metadata["model"])
except Exception as error:
    st.error(f"索引加载失败：{error}")
    st.stop()

with st.form("question_form"):
    question = st.text_area(
        "请输入关于脚踝扭伤或康复的问题",
        placeholder="Example: When should I seek medical help for ankle swelling?",
        height=100,
    )
    submitted = st.form_submit_button("检索资料", type="primary")

if submitted:
    question = question.strip()
    if not question:
        st.warning("请先输入问题。")
    else:
        warning_detected = contains_warning_term(question)
        if warning_detected:
            st.error(
                "你的描述可能包含需要专业医疗评估的情况。"
                "请停止运动，并及时联系医生、急诊或当地紧急医疗服务。"
            )

        results = retrieve(question, chunks, embeddings, model)
        st.subheader("简短中文回答")
        st.success(generate_short_chinese_answer(question, warning_detected))
        st.caption("该回答由受控模板生成，仅供健康教育；请用下方官方原文核对。")

        st.subheader("相关资料")
        st.write(
            "以下内容是从原始英文患者资料和研究文献中检索出的段落，"
            "不是自动诊断或个性化治疗方案。"
        )

        for rank, (score, chunk) in enumerate(results, start=1):
            page_label = (
                str(chunk["page_start"])
                if chunk["page_start"] == chunk["page_end"]
                else f'{chunk["page_start"]}-{chunk["page_end"]}'
            )
            with st.expander(
                f"{rank}. {chunk['institution']} · 第 {page_label} 页 "
                f"· 相似度 {score:.3f}",
                expanded=rank == 1,
            ):
                st.write(chunk["text"])
                st.markdown(f"[查看官方原始资料]({chunk['source_url']})")
                st.code(chunk["chunk_id"], language=None)

st.divider()
st.caption(
    f"本地索引：{len(chunks)} 个文本块 · "
    f"{embeddings.shape[1]} 维向量 · "
    f"模型 {embedding_metadata['model']}"
)
