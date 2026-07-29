#!/usr/bin/env python3
"""Run deterministic bilingual regression checks against the ankle RAG app."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def percent(value: float) -> float:
    return round(100.0 * value, 1)


def answer_score(answer: str, case: dict) -> tuple[float, dict]:
    expected = f"## {case['expected_heading']}"
    headings = [line for line in answer.splitlines() if line.startswith("## ")]
    intent = 1.0 if expected in answer else 0.0
    focus = 1.0 if headings == [expected] else 0.0
    terms = case["answer_terms"]
    coverage = sum(term in answer for term in terms) / len(terms)
    score = 0.5 * intent + 0.2 * focus + 0.3 * coverage
    return score, {
        "expected_intent_found": bool(intent),
        "single_relevant_intent": bool(focus),
        "answer_concept_coverage": percent(coverage),
        "detected_headings": headings,
    }


def source_score(rows: list[tuple[float, dict]], case: dict) -> tuple[float, list[dict]]:
    terms = [term.casefold() for term in case["evidence_terms"]]
    concept_rules = case.get("evidence_concepts", [])
    minimum_concept_score = float(case.get("evidence_min_score", 1))
    details = []
    hits = 0
    for score, chunk in rows:
        text = chunk["text"].casefold()
        matched = [term for term in terms if term in text]
        matched_concepts = []
        concept_score = 0.0
        for concept in concept_rules:
            concept_terms = [
                term.casefold() for term in concept.get("terms", [])
            ]
            concept_matches = [term for term in concept_terms if term in text]
            if concept_matches:
                weight = float(concept.get("weight", 1))
                concept_score += weight
                matched_concepts.append(
                    {
                        "name": concept["name"],
                        "weight": weight,
                        "matched_terms": concept_matches,
                    }
                )
        is_relevant = (
            concept_score >= minimum_concept_score
            if concept_rules
            else bool(matched)
        )
        hits += is_relevant
        details.append(
            {
                "chunk_id": chunk["chunk_id"],
                "institution": chunk["institution"],
                "similarity": round(score, 4),
                "matched_terms": matched,
                "matched_concepts": matched_concepts,
                "concept_score": concept_score if concept_rules else None,
                "relevance_decision": bool(is_relevant),
            }
        )
    return hits / max(len(rows), 1), details


def bilingual_consistency(zh_ids: list[str], en_ids: list[str]) -> tuple[float, bool]:
    union = set(zh_ids) | set(en_ids)
    overlap = len(set(zh_ids) & set(en_ids)) / len(union) if union else 1.0
    return overlap, zh_ids == en_ids


def render_report(result: dict) -> str:
    m = result["metrics"]
    lines = [
        "# RAG 中英双语评测报告",
        "",
        f"- 生成时间：{result['generated_at_utc']}",
        f"- 测试规模：{result['question_pairs']} 组、{result['query_count']} 次查询",
        f"- 模型：`{result['embedding_model']}`",
        "",
        "## 总指标",
        "",
        f"**{m['overall_score']:.1f} / 100（{result['grade']}）**",
        "",
        "| 指标 | 得分 | 权重 |",
        "|---|---:|---:|",
        f"| 详细中文回答适配度 | {m['answer_adaptation']:.1f} | 40% |",
        f"| 相关资料适配度（Precision@3） | {m['source_relevance_at_3']:.1f} | 35% |",
        f"| 中英文 Top-3 一致性（Jaccard） | {m['bilingual_top3_consistency']:.1f} | 25% |",
        f"| 中英文 Top-3 完全同序比例 | {m['exact_order_pair_rate']:.1f} | 展示项 |",
        "",
        "## 分题结果",
        "",
        "| 主题 | 回答适配 | 资料适配 | Top-3一致性 | 完全同序 |",
        "|---|---:|---:|---:|:---:|",
    ]
    for case in result["cases"]:
        lines.append(
            f"| {case['topic']} | {case['answer_adaptation']:.1f} | "
            f"{case['source_relevance_at_3']:.1f} | "
            f"{case['bilingual_top3_consistency']:.1f} | "
            f"{'是' if case['exact_order_match'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 回答适配度检查预期意图、是否夹带无关意图以及关键概念覆盖。",
            "- 资料适配度是规则化 Precision@3：一般主题使用证据词；就医/影像检查使用分组同义词和加权证据概念。",
            "- 中英文一致性比较 chunk ID；100 表示三条资料集合完全相同。",
            "- 本报告是自动回归测试，不等同于医生对医学正确性与安全性的审核。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    skill_dir = Path(__file__).resolve().parents[1]
    cases = json.loads((skill_dir / "references" / "test_cases.json").read_text())

    sys.path.insert(0, str(project))
    app = importlib.import_module("app")

    case_results = []
    answer_scores = []
    source_scores = []
    consistency_scores = []
    exact_matches = []

    for case in cases:
        language_results = {}
        for language in ("zh", "en"):
            question = case[language]
            answer = app.generate_detailed_chinese_answer(question, False)
            answer_value, answer_detail = answer_score(answer, case)
            rows = app.retrieve(
                question,
                app.chunks,
                app.embeddings,
                app.model,
                app.embedding_metadata.get("query_prefix", ""),
                top_k=3,
            )
            source_value, source_detail = source_score(rows, case)
            answer_scores.append(answer_value)
            source_scores.append(source_value)
            language_results[language] = {
                "question": question,
                "answer": answer,
                "answer_score": percent(answer_value),
                "answer_checks": answer_detail,
                "source_score_at_3": percent(source_value),
                "sources": source_detail,
            }

        zh_ids = [row["chunk_id"] for row in language_results["zh"]["sources"]]
        en_ids = [row["chunk_id"] for row in language_results["en"]["sources"]]
        consistency, exact = bilingual_consistency(zh_ids, en_ids)
        consistency_scores.append(consistency)
        exact_matches.append(exact)
        case_results.append(
            {
                "id": case["id"],
                "topic": case["topic"],
                "answer_adaptation": percent(
                    (
                        language_results["zh"]["answer_score"]
                        + language_results["en"]["answer_score"]
                    )
                    / 200
                ),
                "source_relevance_at_3": percent(
                    (
                        language_results["zh"]["source_score_at_3"]
                        + language_results["en"]["source_score_at_3"]
                    )
                    / 200
                ),
                "bilingual_top3_consistency": percent(consistency),
                "exact_order_match": exact,
                "languages": language_results,
            }
        )

    answer_metric = percent(sum(answer_scores) / len(answer_scores))
    source_metric = percent(sum(source_scores) / len(source_scores))
    consistency_metric = percent(sum(consistency_scores) / len(consistency_scores))
    exact_metric = percent(sum(exact_matches) / len(exact_matches))
    overall = round(
        0.40 * answer_metric + 0.35 * source_metric + 0.25 * consistency_metric,
        1,
    )
    grade = "优秀" if overall >= 90 else "良好" if overall >= 80 else "合格" if overall >= 70 else "需改进"
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_pairs": len(cases),
        "query_count": len(cases) * 2,
        "embedding_model": app.embedding_metadata["model"],
        "metrics": {
            "answer_adaptation": answer_metric,
            "source_relevance_at_3": source_metric,
            "bilingual_top3_consistency": consistency_metric,
            "exact_order_pair_rate": exact_metric,
            "overall_score": overall,
        },
        "grade": grade,
        "weights": {
            "answer_adaptation": 0.40,
            "source_relevance_at_3": 0.35,
            "bilingual_top3_consistency": 0.25,
        },
        "cases": case_results,
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "latest_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    report = render_report(result)
    (output / "latest_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
