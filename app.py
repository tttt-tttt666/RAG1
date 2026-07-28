#!/usr/bin/env python3
"""Minimal local Streamlit interface for ankle-sprain RAG retrieval."""

from __future__ import annotations

import json
import re
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
def load_index(
    index_version: tuple[tuple[int, int], ...],
) -> tuple[list[dict], np.ndarray, np.ndarray, dict]:
    """Load index data, invalidating the cache whenever an index file changes."""
    del index_version
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
    return SentenceTransformer(
        model_name,
        cache_folder=str(ROOT / ".cache" / "huggingface" / "hub"),
        local_files_only=True,
    )


def get_index_version() -> tuple[tuple[int, int], ...]:
    """Return stable file fingerprints used as the Streamlit cache key."""
    return tuple(
        (path.stat().st_mtime_ns, path.stat().st_size)
        for path in (CHUNKS_PATH, EMBEDDINGS_PATH, EMBEDDING_METADATA_PATH)
    )


def retrieve(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    query_prefix: str,
    top_k: int = 3,
) -> list[tuple[float, dict]]:
    retrieval_query = canonicalize_retrieval_query(query)
    prefixed_query = query_prefix + retrieval_query
    query_vector = model.encode(
        [prefixed_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scores = embeddings @ query_vector
    ranked_indices = np.argsort(scores)[::-1]

    # Prefer useful body text from different documents. Research PDFs contain
    # long bibliographies whose repeated terms can otherwise outrank actual
    # patient guidance.
    results: list[tuple[float, dict]] = []
    seen_documents: set[str] = set()
    for index in ranked_indices:
        chunk = chunks[int(index)]
        if is_low_information_chunk(chunk["text"]):
            continue
        document_id = chunk["document_id"]
        if document_id in seen_documents:
            continue
        results.append((float(scores[index]), chunk))
        seen_documents.add(document_id)
        if len(results) == top_k:
            break
    return results


def canonicalize_retrieval_query(query: str) -> str:
    """Map bilingual domain intents to the same English retrieval query."""
    normalized = query.casefold()
    intent_queries = (
        (
            ("return to sport", "returning to", "running", "basketball", "恢复运动", "恢复跑步", "打篮球"),
            "ankle sprain criteria for return to running and sport: pain free, no swelling, full range of motion, strength, balance, agility and sport-specific drills",
        ),
        (
            ("x-ray", "x ray", "hospital", "doctor", "医院", "就医", "拍片", "x光"),
            "ankle sprain red flags and criteria for medical assessment or x-ray",
        ),
        (
            ("range of motion", "strengthening", "balance exercise", "康复训练", "活动度", "力量训练", "平衡训练"),
            "ankle sprain rehabilitation progression: range of motion, strengthening, balance and functional exercises",
        ),
        (
            ("weight bearing", "walking normally", "负重", "正常走路"),
            "ankle sprain progression to weight bearing and normal walking without pain or limping",
        ),
        (
            ("mild", "moderate", "severe", "grade", "轻度", "中度", "重度", "损伤程度"),
            "ankle sprain severity grading: mild moderate severe symptoms and functional limitations",
        ),
        (
            ("brace", "bracing", "taping", "护具", "护踝", "贴扎"),
            "ankle brace or taping after ankle sprain during rehabilitation and return to sport",
        ),
        (
            ("prevent", "another ankle sprain", "再次扭伤", "预防复发"),
            "prevent recurrent ankle sprain with balance training, strengthening, bracing and neuromuscular exercise",
        ),
    )
    matched = [
        canonical
        for terms, canonical in intent_queries
        if any(term in normalized for term in terms)
    ]
    return " ".join(matched) if matched else query


def is_low_information_chunk(text: str) -> bool:
    """Detect bibliography and publication-administration chunks."""
    normalized = " ".join(text.casefold().split())
    doi_count = normalized.count("doi:")
    citation_years = len(re.findall(r"\((?:19|20)\d{2}\)", normalized))
    administrative_markers = (
        "all authors read and approved",
        "author contributions",
        "competing interests",
        "publisher's note",
        "references 1.",
    )
    return (
        doi_count >= 2
        or citation_years >= 4
        or any(marker in normalized for marker in administrative_markers)
    )


def contains_warning_term(query: str) -> bool:
    normalized = query.casefold()
    return any(term.casefold() in normalized for term in WARNING_TERMS)


def generate_detailed_chinese_answer(query: str, warning: bool) -> str:
    """Return a detailed, conservative Chinese answer from medical templates."""
    if warning:
        return (
            "### 首要建议\n"
            "你的描述包含需要专业评估的危险信号。请停止运动，避免继续负重，"
            "并尽快联系医生、急诊或当地紧急医疗服务。\n\n"
            "### 等待就医时\n"
            "保护受伤脚踝并保持舒适姿势；不要强行活动、按摩或自行判断损伤等级。"
            "如果脚部明显变形、发冷发紫、持续麻木或疼痛迅速加重，应立即寻求急救。\n\n"
            "### 重要说明\n"
            "本工具不能通过文字排除骨折、脱位或严重韧带损伤，请不要仅依赖检索结果。"
        )

    normalized = query.casefold()
    intent_templates = (
        (
            "受伤后现在如何处理",
            (
                "48 hour",
                "first day",
                "early treatment",
                "刚受伤",
                "早期",
                "前48",
                "昨天",
                "今天扭伤",
                "如何处理",
            ),
            "### 早期处理\n"
            "先停止引起疼痛的运动，保护脚踝并减少不必要的负重。休息时可把脚踝抬高；"
            "冷敷应隔着毛巾短时间进行，避免冰块直接接触皮肤。\n\n"
            "### 接下来怎么做\n"
            "不要长时间完全不动。在疼痛允许且没有明显危险信号时，可逐渐尝试轻柔屈伸脚踝，"
            "再根据疼痛、肿胀和步行能力逐步增加活动。\n\n"
            "### 需要就医的情况\n"
            "如果无法负重、疼痛或肿胀持续加重、脚踝明显变形，或脚部麻木、发冷、变色，"
            "应及时接受医疗评估。",
        ),
        (
            "何时以及怎样开始康复训练",
            ("exercise", "movement", "range of motion", "锻炼", "训练", "活动度"),
            "### 建议的训练顺序\n"
            "1. **活动度**：先做轻柔的脚踝屈伸、画圈或用脚写字。\n"
            "2. **柔韧性**：活动较舒适后，逐渐加入小腿后侧拉伸。\n"
            "3. **力量**：再加入提踵或阻力带练习。\n"
            "4. **平衡与功能**：最后逐步练习双脚到单脚站立，并过渡到日常或运动动作。\n\n"
            "### 如何控制强度\n"
            "动作应缓慢、循序渐进。轻微不适可能出现，但训练后疼痛或肿胀明显增加，"
            "说明当前强度可能过高，应降低难度或暂停。\n\n"
            "### 注意\n"
            "具体开始时间取决于损伤程度和负重能力；严重扭伤或反复不稳者应由医生或物理治疗师评估。",
        ),
        (
            "如何恢复稳定性并预防再次扭伤",
            ("balance", "stability", "prevent", "平衡", "稳定", "预防"),
            "### 恢复重点\n"
            "反复扭伤通常不仅需要消肿，还要恢复脚踝力量、本体感觉和平衡控制。"
            "可从扶着固定物双脚站立开始，逐渐过渡到单脚站立、提踵和方向变化练习。\n\n"
            "### 进阶原则\n"
            "只有在当前练习能稳定完成、没有明显疼痛或次日肿胀增加时，再减少手扶、延长时间，"
            "或加入更复杂动作。恢复运动初期可依据专业建议使用护踝或贴扎。\n\n"
            "### 何时评估\n"
            "如果脚踝经常“打软腿”、持续不稳或多次扭伤，应咨询医生或物理治疗师。",
        ),
        (
            "何时恢复走路、跑步或运动",
            (
                "return to sport",
                "return to running",
                "returning to sport",
                "returning to running",
                "when can i run",
                "恢复走路",
                "何时负重",
                "什么时候负重",
                "恢复跑步",
                "恢复运动",
                "返回运动",
                "重返运动",
            ),
            "### 恢复负重\n"
            "在疼痛允许的范围内逐渐增加负重，目标是先恢复不明显跛行的正常步行。"
            "如果走路后疼痛或肿胀明显反弹，应减少距离或强度。\n\n"
            "### 返回跑步和运动\n"
            "通常应先达到：日常走路和上下楼较舒适、活动度接近另一侧、单脚站立和提踵能够稳定完成。"
            "随后按快走、慢跑、直线跑、变向动作、专项训练的顺序逐级恢复。\n\n"
            "### 暂缓运动的情况\n"
            "仍明显跛行、脚踝不稳、活动后持续肿胀，或无法完成基本单脚动作时，不宜直接恢复比赛。",
        ),
        (
            "通常需要多久恢复",
            (
                "how long",
                "recovery time",
                "time to heal",
                "多久",
                "多长时间",
                "恢复时间",
                "多久能好",
                "多久痊愈",
            ),
            "### 大致恢复时间\n"
            "轻度扭伤通常在数周内明显改善，完全恢复常需约六周；较严重损伤可能需要更久。"
            "时间只是参考，不能单独作为恢复运动的标准。\n\n"
            "### 更重要的判断依据\n"
            "应同时观察疼痛和肿胀是否下降、能否正常负重行走、活动度和力量是否恢复，"
            "以及单脚平衡和运动动作是否稳定。\n\n"
            "### 需要复查的情况\n"
            "症状没有逐步改善、持续无法负重、经常不稳，或恢复后反复扭伤时，应接受专业评估。",
        ),
        (
            "恢复运动时是否使用护具或贴扎",
            (
                "ankle brace",
                "bracing",
                "taping",
                "护具",
                "护踝",
                "贴扎",
                "运动贴",
            ),
            "### 是否可以使用\n"
            "恢复篮球等需要跳跃和变向的运动时，护踝或专业贴扎可以作为短期辅助，"
            "尤其适用于既往反复扭伤、刚开始恢复专项训练，或专业人员建议使用的人。\n\n"
            "### 不能替代康复训练\n"
            "护具和贴扎不能替代活动度、力量、平衡及变向控制训练，也不能证明脚踝已经适合比赛。"
            "恢复前仍应确认正常走路、单脚站立、提踵、跳跃和变向动作能够稳定完成，"
            "且运动后疼痛与肿胀没有明显增加。\n\n"
            "### 使用注意\n"
            "护具应尺寸合适，不应造成麻木、发冷、变色或明显压痛；贴扎最好由受过训练的人员指导。"
            "如果脚踝持续不稳、疼痛或反复肿胀，应咨询医生或物理治疗师，而不是只依赖护具。",
        ),
        (
            "如何处理肿胀及安全冷敷",
            ("ice", "cold", "swelling", "冰敷", "冷敷", "肿胀"),
            "### 冷敷与抬高\n"
            "冷敷时用毛巾隔开皮肤，短时间进行，并在休息时抬高患肢。"
            "不要让冰块直接接触皮肤，也不要在感觉减退的部位长时间冷敷。\n\n"
            "### 活动安排\n"
            "肿胀明显时减少会加重症状的活动；在疼痛允许时保留轻柔活动，避免长期完全制动。\n\n"
            "### 何时就医\n"
            "如果肿胀快速增加、长时间不改善，或伴随无法负重、麻木、发冷、明显变色，应及时就医。",
        ),
        (
            "哪些情况需要去医院",
            ("medical help", "doctor", "hospital", "seek help", "医生", "医院", "就医"),
            "### 建议尽快就医\n"
            "无法正常负重或走四步、疼痛或肿胀持续加重、按压骨头处明显疼痛，"
            "或者数日后仍没有改善，都值得接受专业评估。\n\n"
            "### 需要立即处理的危险信号\n"
            "脚踝明显变形、脚部麻木或失去知觉、发冷发紫、剧烈疼痛，或伤后出现开放性伤口时，"
            "应立即联系急诊或当地紧急医疗服务。\n\n"
            "### 为什么需要检查\n"
            "仅凭文字无法可靠区分普通扭伤、骨折、脱位或严重韧带损伤；医生可能根据检查决定是否需要影像学检查。",
        ),
    )
    matched_answers = []
    for title, terms, answer in intent_templates:
        if any(term in normalized for term in terms):
            matched_answers.append(f"## {title}\n\n{answer}")

    if matched_answers:
        introduction = (
            f"我识别到你的问题包含 **{len(matched_answers)} 个方面**，下面逐项回答。"
            if len(matched_answers) > 1
            else ""
        )
        return "\n\n---\n\n".join(part for part in [introduction, *matched_answers] if part)

    return (
        "### 总体建议\n"
        "先保护受伤脚踝，并根据疼痛、肿胀和负重能力循序渐进地恢复活动。"
        "避免忍痛强行训练，也不建议在没有专业建议的情况下长期完全制动。\n\n"
        "### 恢复原则\n"
        "一般先恢复轻柔活动度和正常步行，再逐步加入力量、平衡以及运动专项训练。"
        "如果某一级活动导致疼痛或肿胀明显增加，应退回较轻的强度。\n\n"
        "### 安全提示\n"
        "请结合下方检索出的官方英文原文核对。症状持续、加重、无法负重或反复不稳时，"
        "应咨询医生或物理治疗师。"
    )


st.set_page_config(
    page_title="脚踝康复资料助手",
    page_icon="🦶",
    layout="centered",
)

st.title("脚踝康复资料助手")
st.caption("基于 42 份医院、政府卫生机构和专业医学组织可信资料的本地语义检索 Demo")

st.info(
    "本工具仅提供健康教育资料检索，不能诊断伤情或替代医生。"
    "当前多语言模型支持中文问题检索英文医学资料，也支持直接使用英文提问。"
)

try:
    chunks, embeddings, _, embedding_metadata = load_index(get_index_version())
    model = load_model(embedding_metadata["model"])
except Exception as error:
    st.error(f"索引加载失败：{error}")
    st.stop()

with st.form("question_form"):
    question = st.text_area(
        "请输入关于脚踝扭伤或康复的问题",
        placeholder="例如：脚踝扭伤后达到什么条件才能恢复打篮球？",
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

        results = retrieve(
            question,
            chunks,
            embeddings,
            model,
            query_prefix=embedding_metadata.get("query_prefix", ""),
        )
        st.subheader("详细中文回答")
        st.success(generate_detailed_chinese_answer(question, warning_detected))
        st.caption(
            "该回答由详细的受控模板生成，仅供健康教育，不能替代诊断或个体化康复方案；"
            "请用下方官方原文核对。"
        )

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
