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


st.set_page_config(
    page_title="脚踝康复资料助手",
    page_icon="🦶",
    layout="centered",
)

st.title("脚踝康复资料助手")
st.caption("基于 4 份 NHS 患者教育资料的本地语义检索 Demo")

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
        if contains_warning_term(question):
            st.error(
                "你的描述可能包含需要专业医疗评估的情况。"
                "请停止运动，并及时联系医生、急诊或当地紧急医疗服务。"
            )

        results = retrieve(question, chunks, embeddings, model)
        st.subheader("相关资料")
        st.write(
            "以下内容是从原始英文患者教育资料中检索出的段落，"
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
