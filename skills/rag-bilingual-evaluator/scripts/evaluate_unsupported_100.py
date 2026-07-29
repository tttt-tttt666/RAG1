#!/usr/bin/env python3
"""Run a 100-question pre-retrieval unsupported-intent stress test."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    skill_dir = Path(__file__).resolve().parents[1]
    cases_path = skill_dir / "references" / "unsupported_100_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if len(cases) != 100:
        raise ValueError(f"无答案测试集必须为100题，当前为{len(cases)}题。")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("测试题 ID 必须唯一。")

    sys.path.insert(0, str(project))
    app = importlib.import_module("app")

    results = []
    for case in cases:
        question = case["question"]
        scope = app.local_question_scope_assessment(question)
        rows = []
        if scope["should_answer"]:
            rows = app.retrieve(
                question,
                app.chunks,
                app.embeddings,
                app.model,
                app.embedding_metadata.get("query_prefix", ""),
                top_k=3,
            )
        results.append(
            {
                **case,
                "scope_allowed": scope["should_answer"],
                "scope_category": scope["category"],
                "scope_reason": scope["reason"],
                "scope_source": scope["source"],
                "best_score": round(rows[0][0], 4) if rows else 0.0,
                "top3_chunk_ids": [chunk["chunk_id"] for _, chunk in rows],
                "passed": not scope["should_answer"],
            }
        )

    category_counts = Counter(case["category"] for case in cases)
    category_summary = {}
    for category, total in category_counts.items():
        rows = [row for row in results if row["category"] == category]
        category_summary[category] = {
            "passed": sum(row["passed"] for row in rows),
            "total": total,
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite_label": "RAG 资料无答案100题强化评测",
        "mode": "offline_pre_retrieval_scope_gate",
        "case_count": len(results),
        "passed": sum(row["passed"] for row in results),
        "category_summary": category_summary,
        "results": results,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "unsupported_100_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# RAG 资料无答案100题强化评测",
        "",
        f"- 生成时间：{summary['generated_at_utc']}",
        "- 判定门槛：必须在向量检索前由范围或能力边界审查拒绝",
        f"- 通过：**{summary['passed']} / {summary['case_count']}**",
        "",
        "## 分类结果",
        "",
        "| 类型 | 通过 | 总数 | 通过率 |",
        "|---|---:|---:|---:|",
    ]
    for category, values in category_summary.items():
        rate = 100.0 * values["passed"] / values["total"]
        lines.append(
            f"| {category} | {values['passed']} | {values['total']} | {rate:.1f}% |"
        )

    lines.extend(["", "## 失败项", ""])
    failed = [row for row in results if not row["passed"]]
    if not failed:
        lines.append("- 无。")
    for row in failed:
        lines.append(
            f"- **{row['id']} · {row['category']}**：{row['question']}；"
            f"最佳分：{row['best_score']:.3f}；Top-3："
            f"`{', '.join(row['top3_chunk_ids']) or '无'}`。"
        )
    lines.extend(
        [
            "",
            "## 判定说明",
            "",
            "- 通过：问题在检索前被明确识别为资料或系统能力范围外。",
            "- 失败：问题被放行并进入向量检索；即使后续因分数不足拒答，也仍算失败。",
            "- 本测试检查拒答边界，不等同于临床医学正确性或安全性审核。",
            "",
        ]
    )
    report = "\n".join(lines)
    (output / "unsupported_100_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
