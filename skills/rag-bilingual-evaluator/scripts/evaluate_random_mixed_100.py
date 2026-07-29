#!/usr/bin/env python3
"""Run the reproducible 100-question mixed ankle/scope regression suite."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import evaluate_safety_45 as evaluator


EXPECTED_COUNTS = {
    "普通康复": 45,
    "危险症状": 20,
    "资料无答案": 20,
    "脚踝无关": 15,
}


def main() -> None:
    skill_dir = Path(__file__).resolve().parents[1]
    cases_path = skill_dir / "references" / "random_mixed_100_cases.json"
    cases = evaluator.json.loads(cases_path.read_text(encoding="utf-8"))
    counts = Counter(case["category"] for case in cases)
    if len(cases) != 100 or counts != Counter(EXPECTED_COUNTS):
        raise ValueError(
            f"混合测试集应为100题，分类为{EXPECTED_COUNTS}；"
            f"当前共{len(cases)}题，分类为{dict(counts)}。"
        )
    evaluator.EXPECTED_COUNTS = EXPECTED_COUNTS
    sys.argv.extend(
        [
            "--cases",
            str(cases_path),
            "--report-stem",
            "random_mixed_100",
            "--suite-label",
            "RAG 随机混合100题安全、证据与范围评测",
        ]
    )
    evaluator.main()


if __name__ == "__main__":
    main()
