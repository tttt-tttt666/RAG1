#!/usr/bin/env python3
"""Minimal local Streamlit interface for ankle-sprain RAG retrieval."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
from sentence_transformers import CrossEncoder, SentenceTransformer

from deepseek_translator import (
    api_is_configured,
    assess_ankle_risk,
    assess_question_scope,
    translate_to_chinese,
)


ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "index" / "ankle_sprain"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings" / "embeddings.npz"
EMBEDDING_METADATA_PATH = INDEX_DIR / "embeddings" / "metadata.json"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_ENCODER_CANDIDATES = 20
CROSS_ENCODER_MIN_SCORE = 0.50
DEFAULT_ANSWER_THRESHOLD = 0.65

WARNING_TERMS = (
    "畸形",
    "变形",
    "无法负重",
    "不能负重",
    "不能承重",
    "无法承重",
    "走不了四步",
    "不能走",
    "无法行走",
    "麻木",
    "失去知觉",
    "没有感觉",
    "发紫",
    "变紫",
    "发冷",
    "冰冷",
    "剧烈疼痛",
    "疼痛加重",
    "肿胀加重",
    "开放性伤口",
    "伤口开放",
    "呼吸困难",
    "高烧",
    "deformity",
    "cannot walk",
    "unable to bear weight",
    "cannot bear weight",
    "cannot take four steps",
    "numb",
    "loss of sensation",
    "cold foot",
    "blue foot",
    "severe pain",
    "worsening pain",
    "worsening swelling",
    "open wound",
    "difficulty breathing",
)

EMERGENCY_TERMS = (
    "畸形",
    "变形",
    "失去知觉",
    "没有感觉",
    "发紫",
    "变紫",
    "发冷",
    "冰冷",
    "开放性伤口",
    "伤口开放",
    "deformity",
    "loss of sensation",
    "cold foot",
    "blue foot",
    "open wound",
)

URGENT_REVIEW_TERMS = (
    "无法负重",
    "不能负重",
    "不能承重",
    "无法承重",
    "走不了四步",
    "不能走",
    "无法行走",
    "剧烈疼痛",
    "疼痛加重",
    "肿胀加重",
    "unable to bear weight",
    "cannot bear weight",
    "cannot take four steps",
    "cannot walk",
    "severe pain",
    "worsening pain",
    "worsening swelling",
)

BASKETBALL_FUNCTION_MARKERS = (
    "single-leg heel raises, hopping, and cutting",
    "single leg heel raises, hopping, and cutting",
    "heel raises, hopping",
    "无痛完成单脚提踵、跳跃和变向",
    "单脚提踵、跳跃和变向",
    "提踵、跳跃和变向",
)

BASKETBALL_EVIDENCE_TERMS = (
    "single leg heel raise",
    "single-leg heel raise",
    "hopping",
    "cutting",
    "jumping",
    "sport-specific",
    "agility",
    "criteria to progress",
    "pain-free",
    "without pain",
    "full strength",
)

DANGER_EVIDENCE_TERMS = (
    "deformity",
    "unable to put weight",
    "unable to bear weight",
    "cannot bear weight",
    "cannot walk",
    "numb",
    "tingling",
    "coldness",
    "feels cold",
    "blue foot",
    "blue toes",
    "discolored",
    "changes in sensation",
    "symptoms get worse",
    "pain or swelling hasn’t improved",
    "severe pain",
)


def is_load_response_query(query: str) -> bool:
    """Return whether a question asks how to react to post-exercise symptoms."""
    normalized = query.casefold()
    fixed_markers = (
        "pain and swelling increase the day after",
        "continue training or reduce",
        "worse the next day",
        "第二天疼痛和肿胀增加",
        "继续训练还是降低",
        "训练后疼痛或肿胀",
        "次日肿胀增加",
    )
    concept_match = (
        any(term in normalized for term in ("训练", "练习", "exercise", "training"))
        and any(term in normalized for term in ("第二天", "次日", "next day"))
        and any(
            term in normalized
            for term in (
                "更肿",
                "肿胀增加",
                "疼痛增加",
                "症状加重",
                "more swollen",
                "increased swelling",
                "more pain",
                "worse",
            )
        )
    )
    return concept_match or any(marker in normalized for marker in fixed_markers)


def prioritized_evidence_terms(query: str) -> tuple[str, ...]:
    """Return strict evidence terms for topics prone to keyword-only matches."""
    normalized = query.casefold()
    if any(
        term in normalized
        for term in (
            "腓总神经",
            "腓骨神经",
            "足下垂",
            "脚背麻木",
            "common peroneal nerve",
            "common fibular nerve",
            "foot drop",
            "dorsum of the foot",
        )
    ):
        return (
            "common peroneal nerve",
            "common fibular nerve",
            "foot drop",
            "dorsum of the foot",
            "numbness",
            "tingling",
        )
    if any(
        term in normalized
        for term in (
            "踝关节周围",
            "踝关节解剖",
            "骨骼、韧带",
            "骨骼和韧带",
            "肌肉、神经",
            "肌肉和神经",
            "blood vessels",
            "ankle anatomy",
            "bones, ligaments",
            "muscles, nerves",
        )
    ):
        return (
            "bones in the ankle",
            "ligaments in the ankle",
            "muscles in the ankle",
            "nerves in the ankle",
            "blood vessels in the ankle",
        )
    if (
        any(term in normalized for term in ("六周", "6周", "6 weeks", "persistent", "持续"))
        and any(term in normalized for term in ("x-ray", "x ray", "mri", "影像", "拍片", "x光"))
    ):
        return ("chronic ankle pain", "6 weeks", "x-ray", "mri", "ultrasound", "ct")
    if any(
        term in normalized
        for term in (
            "x-ray",
            "x ray",
            "radiograph",
            "拍片",
            "x光",
        )
    ):
        return (
            "x-ray",
            "x ray",
            "radiograph",
            "ottawa ankle",
        )
    if is_load_response_query(query):
        return (
            "exercise",
            "pain",
            "swelling",
            "increase",
            "stop",
            "reduce",
            "progress",
        )
    if contains_warning_term(query):
        return DANGER_EVIDENCE_TERMS
    if any(term in normalized for term in ("护踝", "贴扎", "brace", "bracing", "taping")):
        return ("brace", "bracing", "taping", "strapping")
    if any(
        term in normalized
        for term in (
            "一级",
            "二级",
            "三级",
            "grade 1",
            "grade 2",
            "grade 3",
            "轻度",
            "中度",
            "重度",
            "损伤程度",
        )
    ):
        return ("grade i", "grade ii", "grade iii", "grade 1", "grade 2", "grade 3")
    if any(
        term in normalized
        for term in (
            "再次扭伤",
            "预防复发",
            "prevent",
            "another ankle sprain",
            "reduce the risk",
            "recurrent",
        )
    ):
        return ()
    if any(
        term in normalized
        for term in (
            "恢复跑步",
            "打篮球",
            "重返运动",
            "return to sport",
            "returning to running",
            "returning to sport",
        )
    ):
        return (
            "return to sport",
            "return to running",
            "running",
            "hop",
            "agility",
            "sport-specific",
            "pain-free",
            "full strength",
        )
    return ()


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


@st.cache_resource
def load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load the optional reranker only when the user selects that mode."""
    return CrossEncoder(
        model_name,
        cache_folder=str(ROOT / ".cache" / "huggingface" / "hub"),
        local_files_only=True,
        max_length=512,
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
    reranker: CrossEncoder | None = None,
    candidate_k: int = CROSS_ENCODER_CANDIDATES,
    reranker_min_score: float = CROSS_ENCODER_MIN_SCORE,
) -> list[tuple[float, dict]]:
    retrieval_query = canonicalize_retrieval_query(query)
    prefixed_query = query_prefix + retrieval_query
    query_vector = model.encode(
        [prefixed_query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
    scores = embeddings @ query_vector
    quality_adjustments = np.asarray(
        [chunk_quality_adjustment(chunk["text"]) for chunk in chunks],
        dtype=np.float32,
    )
    scores = np.minimum(scores + quality_adjustments, 1.0)
    danger_query = contains_warning_term(query)
    priority_evidence_terms = prioritized_evidence_terms(query)
    if priority_evidence_terms:
        priority_scores = scores.copy()
        for index, chunk in enumerate(chunks):
            text = chunk["text"].casefold()
            evidence_hits = sum(term in text for term in priority_evidence_terms)
            if evidence_hits:
                priority_scores[index] += min(evidence_hits * 0.035, 0.14)
            else:
                priority_scores[index] -= 0.12
        ranked_indices = np.argsort(priority_scores)[::-1]
        result_scores = np.minimum(priority_scores, 1.0)
    else:
        ranked_indices = np.argsort(scores)[::-1]
        result_scores = scores
    basketball_function_query = any(
        marker in query.casefold() for marker in BASKETBALL_FUNCTION_MARKERS
    )
    if basketball_function_query:
        rerank_scores = scores.copy()
        for index, chunk in enumerate(chunks):
            text = chunk["text"].casefold()
            evidence_hits = sum(term in text for term in BASKETBALL_EVIDENCE_TERMS)
            rerank_scores[index] += min(evidence_hits * 0.035, 0.175)
            if "surgery" in text or "postoperative" in text:
                rerank_scores[index] -= 0.12
            if "no studies propose" in text or "hypothetic algorithm" in text:
                rerank_scores[index] -= 0.10
        ranked_indices = np.argsort(rerank_scores)[::-1]
        result_scores = rerank_scores
    elif not danger_query and not priority_evidence_terms:
        ranked_indices = np.argsort(scores)[::-1]
        result_scores = scores

    # Preserve the original fast-search behaviour when reranking is disabled.
    if reranker is None:
        results: list[tuple[float, dict]] = []
        seen_documents: set[str] = set()
        for index in ranked_indices:
            chunk = chunks[int(index)]
            if is_low_information_chunk(chunk["text"]):
                continue
            if priority_evidence_terms and not any(
                term in chunk["text"].casefold() for term in priority_evidence_terms
            ):
                continue
            if (
                basketball_function_query
                and not any(
                    term in chunk["text"].casefold()
                    for term in BASKETBALL_EVIDENCE_TERMS
                )
            ):
                continue
            document_id = chunk["document_id"]
            if document_id in seen_documents:
                continue
            results.append((float(result_scores[index]), chunk))
            seen_documents.add(document_id)
            if len(results) == top_k:
                break
        return results

    # First-stage Bi-Encoder recall for CrossEncoder reranking. Research PDFs
    # contain long bibliographies whose repeated terms can otherwise outrank
    # actual patient guidance.
    candidate_limit = max(candidate_k, top_k)
    candidates: list[tuple[float, dict]] = []
    for index in ranked_indices:
        chunk = chunks[int(index)]
        if is_low_information_chunk(chunk["text"]):
            continue
        if priority_evidence_terms and not any(
            term in chunk["text"].casefold() for term in priority_evidence_terms
        ):
            continue
        if (
            basketball_function_query
            and not any(
                term in chunk["text"].casefold()
                for term in BASKETBALL_EVIDENCE_TERMS
            )
        ):
            continue
        candidates.append((float(result_scores[index]), chunk))
        if len(candidates) == candidate_limit:
            break

    # Second-stage CrossEncoder: read each query/passage pair jointly and
    # reorder the Bi-Encoder candidates by direct relevance.
    if candidates:
        pairs = [(retrieval_query, chunk["text"]) for _, chunk in candidates]
        rerank_scores = np.asarray(
            reranker.predict(
                pairs,
                batch_size=4,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        ).reshape(-1)
        candidates = sorted(
            [
                (float(rerank_score), chunk)
                for rerank_score, (_, chunk) in zip(rerank_scores, candidates)
            ],
            key=lambda item: item[0],
            reverse=True,
        )

    # Return useful body text from different documents.
    results = []
    seen_documents = set()
    for score, chunk in candidates:
        if score < reranker_min_score:
            continue
        document_id = chunk["document_id"]
        if document_id in seen_documents:
            continue
        results.append((score, chunk))
        seen_documents.add(document_id)
        if len(results) == top_k:
            break
    return results


def canonicalize_retrieval_query(query: str) -> str:
    """Map bilingual domain intents to the same English retrieval query."""
    normalized = query.casefold()
    if any(
        term in normalized
        for term in (
            "腓总神经",
            "腓骨神经",
            "足下垂",
            "脚背麻木",
            "common peroneal nerve",
            "common fibular nerve",
            "foot drop",
            "dorsum of the foot",
        )
    ):
        return (
            "common peroneal nerve injury anatomy and function: foot drop, "
            "weak ankle dorsiflexion, numbness or tingling on the dorsum of the foot"
        )
    if any(
        term in normalized
        for term in (
            "踝关节周围",
            "踝关节解剖",
            "骨骼、韧带",
            "骨骼和韧带",
            "肌肉、神经",
            "肌肉和神经",
            "blood vessels",
            "ankle anatomy",
            "bones, ligaments",
            "muscles, nerves",
        )
    ):
        return (
            "ankle anatomy: names of bones, cartilage, ligaments, muscles, "
            "nerves and blood vessels around the ankle joint"
        )
    if (
        any(term in normalized for term in ("六周", "6周", "6 weeks", "persistent", "持续"))
        and any(term in normalized for term in ("x-ray", "x ray", "mri", "影像", "拍片", "x光"))
    ):
        return (
            "chronic ankle pain persisting for 6 weeks or more: x-ray as first "
            "imaging test and when MRI, CT or ultrasound may be appropriate"
        )
    if contains_warning_term(query):
        return (
            "ankle injury emergency red flags: deformity, inability to bear weight, "
            "numbness, cold or blue foot, severe or worsening pain, and when to seek "
            "urgent medical assessment"
        )
    if any(
        marker in normalized
        for marker in (
            "康复训练应该按照什么顺序",
            "康复训练顺序",
            "rehabilitation exercises be performed",
            "order should rehabilitation",
        )
    ):
        return (
            "ankle sprain rehabilitation progression in order: range of motion, "
            "strengthening, balance, proprioception and functional exercises"
        )
    high_ankle_comparison_markers = (
        "difference between a high ankle sprain",
        "high ankle sprain and a common lateral",
        "high ankle sprain take longer",
        "高位踝扭伤和普通外侧踝扭伤",
        "高位踝扭伤与普通外侧踝扭伤",
        "高位踝扭伤和外侧踝扭伤",
    )
    if any(marker in normalized for marker in high_ankle_comparison_markers):
        return (
            "compare high syndesmotic ankle sprain versus common lateral ankle "
            "sprain: injured ligaments, location, mechanism, symptoms, severity "
            "and why high ankle sprains take longer to recover"
        )

    if is_load_response_query(query):
        return (
            "ankle rehabilitation exercise dosage: reduce intensity or pause "
            "when pain and swelling increase after exercise or the next day"
        )

    if any(marker in normalized for marker in BASKETBALL_FUNCTION_MARKERS):
        return (
            "ankle sprain return to basketball functional criteria: pain-free "
            "single-leg heel raises, balance, hopping, cutting and sport-specific drills"
        )

    if any(
        marker in normalized
        for marker in ("ankle brace", "bracing", "taping", "护具", "护踝", "贴扎")
    ):
        return (
            "ankle brace or taping after ankle sprain during rehabilitation "
            "and return to sport"
        )

    recurrence_markers = (
        "prevent",
        "another ankle sprain",
        "reduce the risk",
        "risk of another",
        "再次扭伤",
        "降低脚踝",
        "降低再次",
        "预防复发",
    )
    if any(marker in normalized for marker in recurrence_markers):
        return (
            "prevent recurrent ankle sprain with balance training, "
            "strengthening, bracing and neuromuscular exercise"
        )

    intent_queries = (
        (
            (
                "sprained my ankle yesterday",
                "what should i do now",
                "扭伤了脚踝",
                "昨天扭伤",
                "现在应该如何处理",
            ),
            "early care after an acute ankle sprain: protect, ice, compression, elevation and gradual weight bearing",
        ),
        (
            ("how long", "recovery time", "time to heal", "多久恢复", "恢复时间", "多久能好"),
            "ankle sprain recovery time: how many days or weeks pain and swelling take to improve and heal",
        ),
        (
            ("ice and elevation", "use ice", "cold pack", "冷敷", "冰敷", "抬高"),
            "ankle sprain swelling care: safe ice or cold pack use and elevation",
        ),
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
            (
                "prevent",
                "another ankle sprain",
                "reduce the risk",
                "risk of another",
                "再次扭伤",
                "降低脚踝",
                "降低再次",
                "预防复发",
            ),
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
    compact = re.sub(r"\s+", "", normalized)
    doi_count = normalized.count("doi:")
    doi_link_count = compact.count("doi.org")
    citation_years = len(re.findall(r"\((?:19|20)\d{2}\)", normalized))
    reference_signals = (
        doi_count
        + doi_link_count
        + normalized.count("pmid:")
        + normalized.count(" et al.")
    )
    administrative_markers = (
        "all authors read and approved",
        "author contributions",
        "competing interests",
        "publisher's note",
        "references 1.",
        "correspondence:",
        "*correspondence",
        "author affiliations",
    )
    footer_markers = (
        "privacy",
        "cookies",
        "accessibility",
        "copyright",
        "terms and conditions",
        "contact us",
        "switchboard",
        "follow us",
        "staff intranet",
    )
    footer_count = sum(marker in normalized for marker in footer_markers)
    title_page_admin = (
        "abstract" in normalized
        and ("correspondence" in normalized or "@" in normalized)
        and ("department of" in normalized or "university" in normalized)
    )
    return (
        doi_count >= 2
        or doi_link_count >= 2
        or citation_years >= 4
        or reference_signals >= 4
        or (citation_years >= 3 and reference_signals >= 2)
        or footer_count >= 3
        or title_page_admin
        or any(marker in normalized for marker in administrative_markers)
    )


def chunk_quality_adjustment(text: str) -> float:
    """Give actionable clinical passages a small boost and demote publication noise."""
    normalized = " ".join(text.casefold().split())
    compact = re.sub(r"\s+", "", normalized)
    if is_low_information_chunk(text):
        return -0.30

    actionable_markers = (
        "when to seek medical",
        "contact your",
        "you should",
        "do not",
        "start with",
        "progress to",
        "criteria to progress",
        "return to sport",
        "unable to bear weight",
        "symptoms get worse",
        "range of motion",
        "strengthening",
        "balance exercise",
        "grade i",
        "grade ii",
        "grade iii",
        "pain-free",
        "without pain",
    )
    actionable_hits = sum(marker in normalized for marker in actionable_markers)
    adjustment = min(actionable_hits * 0.004, 0.016)

    soft_noise_markers = (
        "doi:",
        "pmid:",
        "references",
        "journal of",
        "volume ",
        "issue ",
        "table 1",
        "fig. ",
        "copyright",
        "publication date",
    )
    soft_noise_hits = sum(marker in normalized for marker in soft_noise_markers)
    if compact.count("doi.org"):
        soft_noise_hits += 1
    adjustment -= min(soft_noise_hits * 0.008, 0.032)
    return adjustment


def term_is_affirmed(text: str, term: str) -> bool:
    """Return whether a risk term appears outside a simple negated context."""
    start = 0
    while True:
        index = text.find(term, start)
        if index < 0:
            return False
        prefix = text[max(0, index - 12) : index]
        suffix = text[index + len(term) : index + len(term) + 8]
        misleading_no_sensation_phrase = (
            term in {"没有感觉", "loss of sensation"}
            and re.match(r"\s*(?:异常|变化|问题|abnormality|change)", suffix)
        )
        negated = bool(
            re.search(
                r"(?:没有|并无|未见|看不出|无)(?:明显)?\s*$"
                r"|(?:没有|并无|未见|无)(?:明显)?[^，。；;]{1,10}(?:或|和|及|、)\s*$"
                r"|(?:no|without|not)\s+(?:obvious\s+)?$"
                r"|(?:no|without|not)\s+[^,.;]{1,20}(?:or|and)\s+$",
                prefix,
            )
        )
        if not negated and not misleading_no_sensation_phrase:
            return True
        start = index + len(term)


def has_worsening_pain_or_swelling(query: str) -> bool:
    """Recognize worsening symptoms when pain and swelling share one predicate."""
    normalized = query.casefold()
    return bool(
        re.search(
            r"(?:疼痛|痛感)\s*(?:和|或|、|以及)\s*肿胀(?:都|均)?(?:持续)?加重"
            r"|肿胀\s*(?:和|或|、|以及)\s*(?:疼痛|痛感)(?:都|均)?(?:持续)?加重"
            r"|(?:pain|swelling)\s+(?:and|or)\s+(?:pain|swelling)"
            r"\s+(?:is|are|keeps?|continue(?:s)? to)?\s*worsen",
            normalized,
        )
    )


def contains_warning_term(query: str) -> bool:
    normalized = query.casefold()
    return has_worsening_pain_or_swelling(query) or any(
        term_is_affirmed(normalized, term.casefold()) for term in WARNING_TERMS
    )


def evidence_support_for_question(
    query: str,
    rows: list[tuple[float, dict]],
    warning: bool = False,
) -> dict:
    """Check that at least one retrieved passage directly supports the question topic."""
    normalized = query.casefold()
    profiles = (
        (warning, DANGER_EVIDENCE_TERMS, "危险症状与紧急就医"),
        (
            any(term in normalized for term in ("护踝", "贴扎", "brace", "bracing", "taping")),
            ("brace", "bracing", "taping", "strapping"),
            "护踝或贴扎",
        ),
        (
            any(
                term in normalized
                for term in (
                    "一级",
                    "二级",
                    "三级",
                    "grade 1",
                    "grade 2",
                    "grade 3",
                    "轻度",
                    "中度",
                    "重度",
                    "损伤程度",
                )
            ),
            ("grade i", "grade ii", "grade iii", "grade 1", "grade 2", "grade 3"),
            "脚踝扭伤分级",
        ),
        (
            any(term in normalized for term in ("冷敷", "冰敷", "抬高", "ice", "cold pack", "elevation")),
            ("ice", "cold pack", "elevation", "elevate"),
            "冷敷与抬高",
        ),
        (
            any(term in normalized for term in ("再次扭伤", "预防复发", "prevent", "recurrent")),
            ("recurrent", "re-injury", "prevention", "balance", "brace"),
            "复发预防",
        ),
        (
            any(term in normalized for term in ("跑步", "篮球", "重返运动", "return to sport", "running")),
            ("return to sport", "running", "strength", "balance", "hop", "agility"),
            "重返运动",
        ),
        (
            any(term in normalized for term in ("训练", "活动度", "力量", "平衡", "exercise", "rehabilitation")),
            ("exercise", "rehabilitation", "range of motion", "strength", "balance"),
            "康复训练",
        ),
    )
    selected = next(
        ((terms, topic) for matched, terms, topic in profiles if matched),
        ((), "脚踝扭伤患者教育"),
    )
    terms, topic = selected
    if not rows:
        return {"supported": False, "topic": topic, "matched_chunk_ids": []}
    if not terms:
        return {
            "supported": True,
            "topic": topic,
            "matched_chunk_ids": [rows[0][1]["chunk_id"]],
        }
    matched_chunk_ids = [
        chunk["chunk_id"]
        for _, chunk in rows
        if any(term in chunk["text"].casefold() for term in terms)
    ]
    return {
        "supported": bool(matched_chunk_ids),
        "topic": topic,
        "matched_chunk_ids": matched_chunk_ids,
    }


def contains_chinese(text: str) -> bool:
    """Return whether text contains at least one CJK Unified Ideograph."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


@st.cache_data(show_spinner=False)
def cached_chinese_translation(chunk_id: str, text: str) -> str:
    """Cache translations by immutable chunk ID and source text."""
    del chunk_id
    return translate_to_chinese(text)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_question_scope_assessment(question: str) -> dict:
    """Cache the API admission decision for identical questions."""
    return assess_question_scope(question)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_risk_assessment(question: str) -> dict:
    """Cache the second-stage API risk decision for an identical question."""
    return assess_ankle_risk(question)


def local_risk_assessment(question: str) -> dict:
    """Conservative semantic fallback when the risk API is unavailable."""
    normalized = question.casefold()
    emergency_hits = [
        term for term in EMERGENCY_TERMS if term_is_affirmed(normalized, term)
    ]
    urgent_hits = [
        term for term in URGENT_REVIEW_TERMS if term_is_affirmed(normalized, term)
    ]
    if has_worsening_pain_or_swelling(question):
        urgent_hits.append("疼痛和肿胀持续加重")
    if emergency_hits:
        return {
            "risk_level": "emergency",
            "reason": f"命中需要立即评估的表现：{emergency_hits[0]}。",
            "immediate_action": "停止运动、保护脚踝并立即寻求急诊评估。",
            "source": "本地保守分级",
        }
    if urgent_hits:
        return {
            "risk_level": "urgent_review",
            "reason": f"命中需要尽快医疗评估的表现：{urgent_hits[0]}。",
            "immediate_action": "停止运动、保护脚踝、减少负重，并尽快接受医疗评估。",
            "source": "本地保守分级",
        }
    return {
        "risk_level": "self_care",
        "reason": "没有识别到需要立即或尽快就医的当前危险表现。",
        "immediate_action": "可按一般扭伤原则处理并观察症状变化。",
        "source": "本地保守分级",
    }


def unsupported_request_reason(question: str) -> str | None:
    """Identify requests that the text-only patient-education corpus cannot support."""
    normalized = question.casefold()
    medication_markers = (
        "多少毫克",
        "具体剂量",
        "每次应该吃",
        "每天应该吃",
        "注射哪一种",
        "注射什么",
        "开哪种药",
        "用药剂量",
        "what dose",
        "how many mg",
        "how much medication",
        "which injection",
        "prescribe",
    )
    if any(marker in normalized for marker in medication_markers):
        return "当前资料不能提供个体化处方、注射选择或具体用药剂量。"

    visual_inputs = (
        "上传的",
        "照片",
        "图片",
        "影像图",
        "mri图像",
        "mri片",
        "x光片",
        "ct图像",
        "uploaded",
        "photo",
        "image",
        "scan",
    )
    visual_judgments = (
        "判断",
        "诊断",
        "解读",
        "指出",
        "分析",
        "几级",
        "断裂",
        "interpret",
        "diagnose",
        "identify",
        "grade",
        "torn",
    )
    if any(term in normalized for term in visual_inputs) and any(
        term in normalized for term in visual_judgments
    ):
        return "当前系统只能检索文字资料，不能查看或诊断照片及医学影像。"

    exact_prediction = (
        any(term in normalized for term in ("精确", "准确计算", "具体日期", "哪一天", "exact"))
        and any(
            term in normalized
            for term in ("恢复日期", "恢复时间", "多久恢复", "康复日期", "recovery date")
        )
    )
    if exact_prediction:
        return "资料只能提供一般恢复范围，不能精确预测个人恢复日期。"

    provider_terms = (
        "医院",
        "诊所",
        "医生",
        "专科",
        "hospital",
        "clinic",
        "doctor",
        "specialist",
    )
    external_action_terms = (
        "离我最近",
        "附近",
        "电话",
        "地址",
        "替我预约",
        "帮我预约",
        "挂号",
        "费用",
        "哪位",
        "nearest",
        "phone",
        "address",
        "book",
        "appointment",
        "cost",
        "recommend",
    )
    if any(term in normalized for term in provider_terms) and any(
        term in normalized for term in external_action_terms
    ):
        return "当前资料库不提供本地机构查询、医生推荐、费用信息或预约服务。"

    unsupported_treatments = (
        "针灸",
        "穴位",
        "留针",
        "中药",
        "中成药",
        "基因检测",
        "基因预测",
        "acupuncture",
        "herbal formula",
        "traditional chinese medicine",
        "genetic test",
    )
    if any(term in normalized for term in unsupported_treatments):
        return "问题要求的治疗或检测不在当前资料库支持范围内。"
    return None


def local_question_scope_assessment(question: str) -> dict:
    """Conservative offline fallback when the admission API is unavailable."""
    normalized = question.casefold()
    out_of_scope_terms = (
        "吃什么",
        "饮食",
        "食谱",
        "营养",
        "减肥",
        "what to eat",
        "what should i eat",
        "diet",
        "nutrition",
        "meal",
        "recipe",
        "阿莫西林",
        "抗生素",
        "amoxicillin",
        "antibiotic dose",
        "acupuncture point",
    )
    in_scope_terms = (
        "脚踝",
        "踝关节",
        "扭伤",
        "康复",
        "ankle",
        "sprain",
        "rehab",
        "rehabilitation",
        "护踝",
        "贴扎",
        "brace",
        "taping",
    )
    rehabilitation_scope_terms = (
        "恢复走路",
        "调整负重",
        "跛行",
        "单脚站立",
        "平衡训练",
        "提踵",
        "跳跃",
        "变向",
        "重返运动",
        "恢复跑步",
        "参加篮球",
        "return to walking",
        "adjust weight bearing",
        "limping",
        "single-leg stance",
        "balance training",
        "heel raise",
        "hopping",
        "cutting",
        "return to sport",
    )
    if any(term in normalized for term in ("区别", "比较", "difference", "compare")):
        question_type = "comparison"
    elif any(term in normalized for term in ("什么时候", "何时", "多久", "when", "how long")):
        question_type = "when"
    elif any(term in normalized for term in ("是什么", "什么意思", "what is", "define")):
        question_type = "what_is"
    elif any(
        term in normalized
        for term in ("是否", "能否", "可不可以", "应该吗", "should i", "can i", "is it")
    ):
        question_type = "yes_no"
    elif any(
        term in normalized
        for term in ("怎么", "如何", "做什么", "怎么办", "how do", "what should i do")
    ):
        question_type = "how_to"
    else:
        question_type = "other"
    unsupported_reason = unsupported_request_reason(question)
    if unsupported_reason or any(term in normalized for term in out_of_scope_terms):
        return {
            "should_answer": False,
            "category": "超出资料范围",
            "question_type": question_type,
            "reason": unsupported_reason
            or "问题要求的具体内容不在当前脚踝扭伤资料库支持范围内。",
            "source": (
                "本地能力边界审查" if unsupported_reason else "本地保守规则"
            ),
        }
    allowed = any(term in normalized for term in in_scope_terms) or any(
        term in normalized for term in rehabilitation_scope_terms
    )
    return {
        "should_answer": allowed,
        "category": "脚踝扭伤患者教育" if allowed else "超出资料范围",
        "question_type": question_type,
        "reason": (
            "问题与脚踝扭伤或康复直接相关。"
            if allowed
            else "问题与当前脚踝扭伤患者教育资料范围没有明确关系。"
        ),
        "source": "本地保守规则",
    }


def generate_detailed_chinese_answer(
    query: str,
    warning: bool,
    question_type: str | None = None,
    question_category: str | None = None,
    risk_level: str | None = None,
) -> str:
    """Return a detailed, conservative Chinese answer from medical templates."""
    if warning and risk_level == "urgent_review":
        return (
            "### 现在先怎么做\n"
            "停止运动并保护脚踝，避免勉强继续训练。可减少或暂时停止负重，"
            "休息时抬高患肢；如果需要冷敷，应隔着毛巾短时间进行，避免冻伤。\n\n"
            "### 就医建议\n"
            "这些表现不一定意味着必须急诊，但可能需要排除骨折或较严重损伤。"
            "建议尽快或当日联系医生、急诊门诊或骨科进行评估；"
            "在明确能够安全负重前，不要强行走路或恢复训练。\n\n"
            "### 如果出现以下变化\n"
            "若脚踝明显变形、出现开放伤口，或脚部发冷发紫、持续麻木、感觉丧失，"
            "应立即寻求急诊帮助。"
        )
    if warning and risk_level == "emergency":
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
    next_day_action_markers = (
        "训练后，第二天做什么",
        "训练后第二天做什么",
        "训练后的第二天做什么",
        "次日做什么",
        "第二天怎么办",
        "what should i do the day after",
        "what should i do the next day",
    )
    if any(marker in normalized for marker in next_day_action_markers):
        return (
            "**先检查脚踝对前一天训练的反应，再决定当天做什么；不要直接重复或加量。**\n\n"
            "### 第一步：早晨先检查\n"
            "比较训练前后是否出现新的或更明显的疼痛、肿胀、僵硬、跛行或不稳，"
            "并试走几步，确认负重是否比前一天更困难。\n\n"
            "### 第二步：根据反应安排当天活动\n"
            "- **没有明显加重**：可以做轻柔的脚踝屈伸、画圈和舒适范围内的日常走路；"
            "当天先维持原训练级别，不要立刻增加阻力、次数或单脚难度。\n"
            "- **轻度加重但仍能正常走路**：减少次数、阻力或站立时间，改做较轻的活动度练习，"
            "必要时安排恢复日。\n"
            "- **疼痛或肿胀明显增加、出现跛行或不稳**：暂停力量和平衡训练，保护脚踝并减少负重；"
            "待症状回到训练前水平后，再从较低一级恢复。\n\n"
            "### 何时继续进阶\n"
            "只有当前练习能够稳定完成，而且训练中和第二天都没有明显症状反弹时，"
            "才逐步增加一个变量，例如次数、阻力或动作难度。\n\n"
            "如果症状反复加重、无法正常负重或脚踝持续不稳，应咨询医生或物理治疗师。"
        )
    high_ankle_comparison_markers = (
        "difference between a high ankle sprain",
        "high ankle sprain and a common lateral",
        "high ankle sprain take longer",
        "高位踝扭伤和普通外侧踝扭伤",
        "高位踝扭伤与普通外侧踝扭伤",
        "高位踝扭伤和外侧踝扭伤",
    )
    if any(marker in normalized for marker in high_ankle_comparison_markers):
        return (
            "**问题一：高位踝扭伤和普通外侧踝扭伤有什么区别？**\n\n"
            "- **受伤位置不同**：高位踝扭伤损伤的是踝关节上方、连接胫骨和腓骨的下胫腓联合"
            "韧带；普通外侧踝扭伤主要损伤脚踝外侧的距腓前韧带、跟腓韧带等结构。\n"
            "- **常见受伤方式不同**：高位扭伤常与足部被迫向外旋转、背屈有关；外侧扭伤更常见于"
            "脚向内翻、身体重心压到脚踝外侧。\n"
            "- **症状位置不同**：高位扭伤的疼痛可位于踝关节前方或上方的小腿下段，外观肿胀有时"
            "并不明显，但走路、蹬地或向外转脚可能更痛；外侧扭伤通常在外踝周围出现压痛、肿胀和"
            "瘀青。\n"
            "- **稳定性风险不同**：如果下胫腓联合变得不稳定，高位扭伤需要更严格的固定和专科评估；"
            "多数稳定的外侧扭伤可通过保护、逐渐负重和康复治疗恢复。\n\n"
            "**问题二：高位踝扭伤的恢复时间是否更长？**\n\n"
            "**通常是的，但具体时间取决于损伤等级和关节是否稳定。**轻度外侧踝扭伤常在数周内"
            "明显改善；高位踝扭伤通常需要数周到数月，存在明显不稳、无法负重或需要手术时可能更久。"
            "恢复不能只看时间，还应确认疼痛和肿胀下降、正常走路、活动度、力量、平衡以及专项动作"
            "已经恢复。\n\n"
            "仅凭疼痛位置不能自行确定是哪一种扭伤。如果疼痛主要位于踝关节上方、无法正常负重、"
            "按压胫腓骨之间明显疼痛，或症状没有逐步改善，应由医生或运动医学专业人员评估。"
        )

    if is_load_response_query(query):
        return (
            "**应该降低强度；如果症状增加明显，应暂停当前练习，而不是按原强度继续。**\n\n"
            "训练后或第二天疼痛、肿胀比训练前明显增加，通常说明这次训练负荷超过了脚踝目前"
            "能够承受的程度。可以先减少阻力、次数、站立时间或动作难度；待症状回到训练前水平后，"
            "再从较低一级重新尝试。\n\n"
            "重新训练时，应选择能够稳定完成且不会导致疼痛、肿胀持续增加的强度。"
            "如果即使降低强度仍反复加重，或者出现明显跛行、无法正常负重、脚踝持续不稳，"
            "应停止自行进阶并咨询医生或物理治疗师。"
        )

    if any(marker in normalized for marker in BASKETBALL_FUNCTION_MARKERS):
        return (
            "**是的。无痛、稳定地完成单脚提踵、跳跃和变向，是重返篮球前应检查的重要功能标准，"
            "但不能只看这三项。**\n\n"
            "重返完整训练或比赛前，还应确认：日常走路和上下楼没有明显疼痛或跛行；"
            "脚踝活动度和力量接近未受伤侧；单脚站立稳定；能够逐级完成双脚跳、单脚跳、"
            "直线跑、急停和变向；训练过程中及第二天没有明显疼痛或肿胀增加。\n\n"
            "如果完成动作时需要代偿、落地不稳、害怕发力，或者运动后症状反弹，"
            "应继续较低级别训练，而不是直接参加比赛。反复扭伤或持续不稳者应由医生或"
            "物理治疗师进行功能评估。"
        )

    recurrence_markers = (
        "prevent",
        "another ankle sprain",
        "reduce the risk",
        "risk of another",
        "再次扭伤",
        "降低脚踝",
        "降低再次",
        "预防复发",
    )
    recurrence_intent = any(marker in normalized for marker in recurrence_markers)
    intent_templates = (
        (
            "受伤后现在如何处理",
            (
                "48 hour",
                "first day",
                "early treatment",
                "sprained my ankle yesterday",
                "what should i do now",
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
            (
                "balance",
                "stability",
                "prevent",
                "reduce the risk",
                "risk of another",
                "another ankle sprain",
                "平衡",
                "稳定",
                "预防",
                "再次扭伤",
                "降低脚踝",
                "降低再次",
            ),
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
            "脚踝扭伤的严重程度",
            (
                "grade 1",
                "grade 2",
                "grade 3",
                "一级",
                "二级",
                "三级",
                "mild",
                "moderate",
                "severe",
                "轻度",
                "中度",
                "重度",
                "损伤程度",
                "几级",
                "分级",
            ),
            "### 常见的三级划分\n"
            "- **轻度（1级）**：韧带受到牵拉或有微小纤维损伤，通常只有轻度压痛和肿胀，"
            "一般仍能负重，检查时通常没有明显不稳。\n"
            "- **中度（2级）**：韧带部分撕裂，疼痛、肿胀和瘀青更明显，走路可能疼痛或跛行，"
            "也可能出现轻度松弛或不稳。\n"
            "- **重度（3级）**：韧带完全撕裂，常有明显肿胀、瘀青和关节不稳，"
            "可能无法正常负重或行走。\n\n"
            "### 治疗通常如何变化\n"
            "轻度损伤多采用保护、逐渐负重和循序康复；中度损伤可能需要护具、行走靴或短期限制活动；"
            "重度损伤应由医生评估，可能需要更长时间固定和专业康复。即使是完全撕裂，很多单纯外侧踝扭伤"
            "仍可通过适当固定和康复治疗，手术通常只用于持续不稳、合并损伤或保守治疗失败等情况。\n\n"
            "### 不要自行确定等级\n"
            "疼痛和肿胀程度不能可靠排除骨折或其他损伤。无法走四步、骨性压痛明显、脚踝变形、"
            "麻木发冷，或症状持续加重时，应接受医疗评估；医生可能根据检查决定是否需要X光、超声或MRI。",
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
                "weight bearing",
                "walking normally",
                "负重",
                "正常走路",
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
            "**可以考虑使用，但它只能作为辅助，不能代替康复训练或重返运动评估。**\n\n"
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
            ("ice", "cold", "elevation", "冰敷", "冷敷", "抬高"),
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
        if recurrence_intent and title == "何时恢复走路、跑步或运动":
            continue
        if title == "如何恢复稳定性并预防再次扭伤" and not recurrence_intent:
            continue
        matched_terms = [term for term in terms if term in normalized]
        if matched_terms:
            matched_answers.append(
                {
                    "title": title,
                    "answer": f"## {title}\n\n{answer}",
                    "matched_terms": matched_terms,
                }
            )

    if matched_answers:
        category = (question_category or "").casefold()
        preferred_title = None
        topic_routes = (
            (
                ("护踝", "贴扎", "brace", "bracing", "taping"),
                "恢复运动时是否使用护具或贴扎",
            ),
            (
                ("一级", "二级", "三级", "损伤程度", "分级", "grade"),
                "脚踝扭伤的严重程度",
            ),
            (("冷敷", "冰敷", "抬高", "ice", "cold"), "如何处理肿胀及安全冷敷"),
            (("多久", "恢复时间", "how long"), "通常需要多久恢复"),
            (
                ("再次扭伤", "预防复发", "prevent", "recurrent"),
                "如何恢复稳定性并预防再次扭伤",
            ),
            (
                ("跑步", "篮球", "重返运动", "return to sport", "running"),
                "何时恢复走路、跑步或运动",
            ),
            (("医院", "就医", "doctor", "hospital"), "哪些情况需要去医院"),
            (
                ("负重", "正常走路", "weight bearing", "walking normally"),
                "何时恢复走路、跑步或运动",
            ),
        )
        combined_topic_text = f"{normalized} {category}"
        for markers, title in topic_routes:
            if any(marker in combined_topic_text for marker in markers):
                preferred_title = title
                break
        matched_answers.sort(
            key=lambda item: (
                item["title"] == preferred_title,
                len(item["matched_terms"]),
                max(len(term) for term in item["matched_terms"]),
            ),
            reverse=True,
        )
        if question_type in {"yes_no", "how_to", "what_is", "when", "comparison"}:
            matched_answers = matched_answers[:1]
        introduction = (
            f"我识别到你的问题包含 **{len(matched_answers)} 个方面**，下面逐项回答。"
            if len(matched_answers) > 1
            else ""
        )
        answers = [item["answer"] for item in matched_answers]
        return "\n\n---\n\n".join(part for part in [introduction, *answers] if part)

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
    page_icon=":material/health_and_safety:",
    layout="wide",
)

st.html(
    """
    <style>
      :root {
        --rag-navy: #07162f;
        --rag-blue: #1236f5;
        --rag-cyan: #25d6c8;
        --rag-ink: #17213a;
        --rag-muted: #61708f;
        --rag-surface: #ffffff;
        --rag-soft: #f3f7ff;
        --rag-line: #dfe7f4;
      }

      .stApp {
        background:
          radial-gradient(circle at 8% 4%, rgba(37, 214, 200, 0.10), transparent 22rem),
          linear-gradient(180deg, #f7faff 0%, #ffffff 32rem);
      }

      [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 4rem;
      }

      .rag-hero {
        position: relative;
        overflow: hidden;
        min-height: 290px;
        padding: clamp(28px, 5vw, 62px);
        border-radius: 28px;
        color: white;
        background:
          radial-gradient(circle at 82% 18%, rgba(37, 214, 200, 0.42), transparent 18%),
          radial-gradient(circle at 92% 92%, rgba(84, 109, 255, 0.70), transparent 24%),
          linear-gradient(125deg, #07162f 0%, #1236f5 62%, #07162f 130%);
        box-shadow: 0 28px 70px rgba(16, 43, 130, 0.22);
      }

      .rag-hero::after {
        content: "";
        position: absolute;
        right: -7%;
        top: -20%;
        width: 46%;
        aspect-ratio: 1;
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 50%;
        box-shadow:
          0 0 0 34px rgba(255, 255, 255, 0.035),
          0 0 0 72px rgba(255, 255, 255, 0.025);
      }

      .rag-eyebrow {
        margin: 0 0 16px;
        color: #a9fff5;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
      }

      .rag-hero h1 {
        position: relative;
        z-index: 1;
        max-width: 720px;
        margin: 0;
        color: white;
        font-size: clamp(2.25rem, 6vw, 4.3rem);
        line-height: 1.03;
        letter-spacing: -0.04em;
      }

      .rag-hero-copy {
        position: relative;
        z-index: 1;
        max-width: 670px;
        margin: 20px 0 26px;
        color: rgba(255, 255, 255, 0.82);
        font-size: clamp(1rem, 2vw, 1.16rem);
        line-height: 1.75;
      }

      .rag-chips {
        position: relative;
        z-index: 1;
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }

      .rag-chip {
        padding: 8px 13px;
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.10);
        color: white;
        font-size: 0.82rem;
        backdrop-filter: blur(10px);
      }

      .rag-intro {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 18px 0 28px;
      }

      .rag-intro-card {
        padding: 17px 18px;
        border: 1px solid var(--rag-line);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.86);
        color: var(--rag-muted);
        box-shadow: 0 8px 24px rgba(24, 51, 105, 0.06);
      }

      .rag-intro-card strong {
        display: block;
        margin-bottom: 5px;
        color: var(--rag-ink);
      }

      .st-key-question_form {
        margin-top: 8px;
        padding: clamp(20px, 4vw, 34px);
        border: 1px solid var(--rag-line);
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.94);
        box-shadow: 0 18px 48px rgba(24, 51, 105, 0.10);
      }

      .st-key-question_form textarea {
        min-height: 128px;
        border-radius: 16px;
        background: #f7f9ff;
        font-size: 1rem;
        line-height: 1.65;
      }

      .st-key-question_form [data-testid="stFormSubmitButton"] button {
        min-height: 48px;
        border: 0;
        border-radius: 14px;
        background: linear-gradient(105deg, var(--rag-blue), #3158ff);
        box-shadow: 0 12px 24px rgba(18, 54, 245, 0.22);
        font-weight: 700;
      }

      [data-testid="stAlert"] {
        border-radius: 16px;
      }

      [data-testid="stExpander"] {
        overflow: hidden;
        border-color: var(--rag-line);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.88);
        box-shadow: 0 7px 22px rgba(24, 51, 105, 0.055);
      }

      [data-testid="stCode"] {
        border-radius: 12px;
      }

      h2, h3 {
        color: var(--rag-ink);
        letter-spacing: -0.025em;
      }

      @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
          padding-top: 1rem;
          padding-left: 1rem;
          padding-right: 1rem;
        }

        .rag-hero {
          min-height: 0;
          border-radius: 20px;
        }

        .rag-hero::after {
          opacity: 0.5;
        }

        .rag-intro {
          grid-template-columns: 1fr;
        }

        .st-key-question_form {
          border-radius: 19px;
        }
      }
    </style>
    """
)

st.html(
    """
    <section class="rag-hero">
      <p class="rag-eyebrow">Evidence-guided bilingual RAG</p>
      <h1>脚踝康复资料助手</h1>
      <p class="rag-hero-copy">
        用中文或英文提问，从医院、政府卫生机构与专业医学组织资料中检索依据，
        获得结构清晰、可追溯来源的中文健康教育回答。
      </p>
      <div class="rag-chips">
        <span class="rag-chip">70 份可信资料</span>
        <span class="rag-chip">中英双语检索</span>
        <span class="rag-chip">风险分级提醒</span>
      </div>
    </section>
    <section class="rag-intro">
      <div class="rag-intro-card"><strong>循证检索</strong>回答与原始资料段落同时呈现</div>
      <div class="rag-intro-card"><strong>双模型模式</strong>快速召回或 CrossEncoder 精排</div>
      <div class="rag-intro-card"><strong>安全边界</strong>资料不足时拒绝强行生成结论</div>
    </section>
    """
)
st.info(
    "本工具仅提供健康教育资料检索，不能诊断伤情或替代医生。"
    "提交的问题会先进行范围与风险判断，通过后才进入本地检索。",
    icon=":material/health_and_safety:",
)

try:
    chunks, embeddings, _, embedding_metadata = load_index(get_index_version())
    model = load_model(embedding_metadata["model"])
except Exception as error:
    st.error(f"索引加载失败：{error}")
    st.stop()

with st.form("question_form", border=False):
    st.subheader("开始提问", anchor=False)
    st.caption("描述受伤时间、症状、负重能力和你想了解的具体问题，会更容易找到合适资料。")
    question = st.text_area(
        "你的问题",
        placeholder="例如：脚踝扭伤后达到什么条件才能恢复打篮球？",
        height=128,
    )
    search_mode = st.segmented_control(
        "检索模式",
        ("Bi-Encoder 快速检索", "Bi-Encoder + CrossEncoder 精排"),
        default="Bi-Encoder 快速检索",
        help=(
            "快速检索只使用现有 E5 向量；精排模式先召回 20 个候选段落，"
            "再让 CrossEncoder 同时阅读问题和段落并重新排序。"
        ),
    )
    with st.expander("高级检索设置", icon=":material/tune:"):
        source_threshold = st.slider(
            "资料显示相关性阈值",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help="低于这个分数的资料段落不会显示。数值越高，筛选越严格。",
        )
        answer_threshold = st.slider(
            "回答生成相关性阈值",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_ANSWER_THRESHOLD,
            step=0.05,
            help=(
                "如果最佳资料仍低于这个分数，系统不会生成回答或展示资料，"
                "而是直接提示当前资料不足。"
            ),
        )
    submitted = st.form_submit_button(
        "检索可信资料",
        type="primary",
        icon=":material/search:",
        width="stretch",
    )

if submitted:
    question = question.strip()
    if not question:
        st.warning("请先输入问题。")
    else:
        st.session_state["active_question"] = question
        st.session_state["active_search_mode"] = search_mode
        st.session_state["active_source_threshold"] = source_threshold
        st.session_state["active_answer_threshold"] = answer_threshold

active_question = st.session_state.get("active_question", "")
if active_question:
    active_search_mode = st.session_state.get(
        "active_search_mode",
        "Bi-Encoder 快速检索",
    )
    active_source_threshold = st.session_state.get(
        "active_source_threshold",
        0.50,
    )
    active_answer_threshold = st.session_state.get(
        "active_answer_threshold",
        DEFAULT_ANSWER_THRESHOLD,
    )

    capability_block = unsupported_request_reason(active_question)
    if capability_block:
        scope_assessment = {
            "should_answer": False,
            "category": "超出资料与系统能力范围",
            "question_type": "other",
            "reason": capability_block,
            "source": "本地能力边界审查",
        }
    elif api_is_configured():
        try:
            with st.spinner("正在判断该问题是否适合由脚踝资料助手回答……"):
                scope_assessment = cached_question_scope_assessment(active_question)
        except Exception as error:
            scope_assessment = local_question_scope_assessment(active_question)
            st.warning(
                "API 问题准入判断暂时不可用，已启用本地保守规则。"
                f"错误信息：{error}"
            )
    else:
        scope_assessment = local_question_scope_assessment(active_question)
        st.warning("DeepSeek API 未配置，问题准入判断暂时使用本地保守规则。")

    if not scope_assessment["should_answer"]:
        st.warning(
            "当前资料不足，无法可靠回答这个问题，因此没有执行检索或生成回答。\n\n"
            f"判断类别：{scope_assessment['category']}\n\n"
            f"原因：{scope_assessment['reason']}"
        )
        st.caption(f"问题准入判断来源：{scope_assessment['source']}")
        st.stop()

    st.caption(
        f"问题准入：允许回答 · {scope_assessment['category']} "
        f"· 问题类型：{scope_assessment.get('question_type', 'other')} "
        f"· 判断来源：{scope_assessment['source']}"
    )
    warning_candidate = contains_warning_term(active_question)
    risk_assessment = {
        "risk_level": "self_care",
        "reason": "未命中潜在危险症状表达。",
        "immediate_action": "",
        "source": "本地关键词初筛",
    }
    if warning_candidate:
        if api_is_configured():
            try:
                with st.spinner("正在对潜在危险症状进行第二阶段风险判断……"):
                    risk_assessment = cached_risk_assessment(active_question)
            except Exception as error:
                risk_assessment = local_risk_assessment(active_question)
                st.warning(
                    "API 风险分级暂时不可用，已启用本地保守分级。"
                    f"错误信息：{error}"
                )
        else:
            risk_assessment = local_risk_assessment(active_question)
    warning_detected = risk_assessment["risk_level"] in {
        "emergency",
        "urgent_review",
    }
    if risk_assessment["risk_level"] == "emergency":
        st.error(
            "风险分级：需要立即医疗评估。请停止运动、避免继续负重，"
            "并立即联系急诊或当地紧急医疗服务。"
        )
    elif risk_assessment["risk_level"] == "urgent_review":
        st.warning(
            "风险分级：可先保护脚踝、停止运动并减少负重，"
            "同时建议尽快或当日接受医疗评估。"
        )
    if warning_candidate:
        st.caption(
            f"风险判断来源：{risk_assessment['source']} · "
            f"原因：{risk_assessment['reason']}"
        )

    active_reranker = None
    if active_search_mode == "Bi-Encoder + CrossEncoder 精排":
        try:
            with st.spinner("正在加载 CrossEncoder 并对候选资料精排……"):
                active_reranker = load_cross_encoder(CROSS_ENCODER_MODEL)
        except Exception as error:
            st.error(
                "CrossEncoder 模型尚未下载或加载失败。"
                "请先下载 BAAI/bge-reranker-v2-m3，或切换回 Bi-Encoder 快速检索。"
            )
            st.code(str(error), language=None)
            st.stop()

    raw_results = retrieve(
        active_question,
        chunks,
        embeddings,
        model,
        query_prefix=embedding_metadata.get("query_prefix", ""),
        reranker=active_reranker,
        reranker_min_score=0.0,
    )
    st.caption(
        f"当前模式：{active_search_mode}"
        + (
            f" · 精排模型 {CROSS_ENCODER_MODEL}"
            if active_reranker is not None
            else f" · 召回模型 {embedding_metadata['model']}"
        )
        + f" · 资料阈值 {active_source_threshold:.2f}"
        + f" · 回答阈值 {active_answer_threshold:.2f}"
    )

    best_score = raw_results[0][0] if raw_results else 0.0
    if not raw_results or best_score < active_answer_threshold:
        st.warning(
            "当前资料与问题的相关性不足，系统没有生成回答。"
            f"最佳匹配分为 {best_score:.3f}，低于回答生成阈值 "
            f"{active_answer_threshold:.2f}。请降低阈值、换一种检索模式，"
            "或把问题描述得更具体。"
        )
        st.stop()

    evidence_support = evidence_support_for_question(
        active_question,
        raw_results,
        warning_detected,
    )
    if not evidence_support["supported"]:
        st.warning(
            "当前资料不足，无法回答这个问题。虽然检索到了文字相近的段落，"
            f"但没有找到能够直接支持“{evidence_support['topic']}”结论的原文，"
            "因此系统不会根据相似关键词强行生成回答。"
        )
        st.stop()

    results = [
        (score, chunk)
        for score, chunk in raw_results
        if score >= active_source_threshold
    ]
    if len(results) < len(raw_results):
        st.info(
            f"当前仅有 {len(results)} 条资料达到显示阈值 "
            f"{active_source_threshold:.2f}；系统不会用低分段落强行补足三条。"
        )
    st.subheader("详细中文回答")
    st.success(
        generate_detailed_chinese_answer(
            active_question,
            warning_detected,
            scope_assessment.get("question_type"),
            scope_assessment.get("category"),
            risk_assessment.get("risk_level"),
        )
    )
    st.caption(
        "该回答由详细的受控模板生成，仅供健康教育，不能替代诊断或个体化康复方案；"
        "请用下方官方原文核对。"
    )

    st.subheader("相关资料")
    st.write(
        "以下内容是从原始英文患者资料和研究文献中检索出的段落，"
        "不是自动诊断或个性化治疗方案。"
    )
    st.caption(
        "中文译文由 DeepSeek 按需生成，会将当前英文段落发送至 DeepSeek API。"
        "机器翻译仅供阅读，请以 English 原文为准；译文不会写入索引或 Embedding。"
    )

    default_language = "中文" if contains_chinese(active_question) else "English"
    score_label = "精排分" if active_reranker is not None else "匹配度"
    for rank, (score, chunk) in enumerate(results, start=1):
        page_label = (
            str(chunk["page_start"])
            if chunk["page_start"] == chunk["page_end"]
            else f'{chunk["page_start"]}-{chunk["page_end"]}'
        )
        with st.expander(
            f"{rank}. {chunk['institution']} · 第 {page_label} 页 "
            f"· {score_label} {score:.3f}",
            expanded=rank == 1,
        ):
            language = st.radio(
                "资料语言",
                ("English", "中文"),
                index=1 if default_language == "中文" else 0,
                horizontal=True,
                key=f"language_{chunk['chunk_id']}",
            )
            if language == "English":
                st.write(chunk["text"])
            elif not api_is_configured():
                st.warning(
                    "中文翻译尚未启用：请在项目根目录的 .env 中填写 "
                    "DEEPSEEK_API_KEY，然后重新启动界面。"
                )
                st.write(chunk["text"])
            else:
                try:
                    with st.spinner("正在翻译成中文……"):
                        translation = cached_chinese_translation(
                            chunk["chunk_id"],
                            chunk["text"],
                        )
                    st.write(translation)
                    st.caption("DeepSeek 机器翻译 · 请以 English 原文为准")
                except Exception as error:
                    st.error(f"翻译失败：{error}")
                    st.write(chunk["text"])

            st.markdown(f"[查看官方原始资料]({chunk['source_url']})")
            st.code(chunk["chunk_id"], language=None)

st.divider()
st.caption(
    f"本地索引：{len(chunks)} 个文本块 · "
    f"{embeddings.shape[1]} 维向量 · "
    f"模型 {embedding_metadata['model']}"
)
