#!/usr/bin/env python3
"""Run a 100-question ankle danger-symptom sensitivity stress test."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import evaluate_safety_45 as evaluator


def main() -> None:
    skill_dir = Path(__file__).resolve().parents[1]
    cases_path = skill_dir / "references" / "danger_100_cases.json"
    cases = evaluator.json.loads(cases_path.read_text(encoding="utf-8"))
    risks = Counter(case["expected_risk"] for case in cases)
    if len(cases) != 100 or risks != Counter(
        {"emergency": 40, "urgent_review": 60}
    ):
        raise ValueError(
            "危险强化集必须为100题，其中 emergency 40题、urgent_review 60题；"
            f"当前为{len(cases)}题，分布为{dict(risks)}。"
        )
    evaluator.EXPECTED_COUNTS = {"危险症状": 100}
    sys.argv.extend(
        [
            "--cases",
            str(cases_path),
            "--report-stem",
            "danger_100",
            "--suite-label",
            "RAG 危险症状100题强化评测",
        ]
    )
    evaluator.main()


if __name__ == "__main__":
    main()
