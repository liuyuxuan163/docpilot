"""分层评测：把"该拒没拒"和"该答没答"按**题目类型**拆开看。

用法::

    python scripts/eval_stratified.py
    python scripts/eval_stratified.py --cases data/eval/hard_negatives.test.jsonl
    python scripts/eval_stratified.py --cases data/eval/cases.jsonl data/eval/hard_negatives.dev.jsonl

为什么要分层
------------
聚合指标会骗人。同一个 `hallucination_rate`，可能来自"完全超纲"（好防），
也可能来自"话题相近但没答案"（难防）—— 两者的修法完全不同。
不分层就只能看到一个数，然后凭感觉猜是哪一类在拖后腿。

这个脚本按 ``kind`` 分组，每组报三件事：

* **负例误答率**（该拒却答）—— 附 Wilson 95% 区间，因为 60 条里翻 1 条就是 1.7pp；
* **正例误拒率**（该答却拒）—— 这是项目"零误拒"硬约束的直接检验；
* **复合分** —— 与 ``python -m docpilot eval`` 同口径，便于横向比。

Wilson 区间而不是正态区间：样本量小 + 比例接近 0/1 时，正态近似会给出
"误拒率 -3% ~ +3%" 这种荒唐区间。这个项目里误拒率恰好是 0，正好踩中。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docpilot.agent import Agent  # noqa: E402
from docpilot.core import ensure_utf8_stdio, load_config  # noqa: E402
from docpilot.eval import EvalCase, score_case  # noqa: E402
from docpilot.llm import MockLLM  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402
from docpilot.tools import build_registry  # noqa: E402

DEFAULT_CASES = [
    ROOT / "data" / "eval" / "hard_negatives.dev.jsonl",
    ROOT / "data" / "eval" / "hard_negatives.test.jsonl",
]


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 分数区间（小样本 + 极端比例下比正态近似可靠）。"""
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    """读取 JSONL；保留 ``kind`` 等自定义字段（``EvalCase`` 会忽略它们）。"""
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"找不到评测文件：{path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            data = json.loads(line)
            data.setdefault("kind", "untagged")
            data.setdefault("hardness", "unknown")
            data["_source"] = path.name
            # 人工复核判定"文档其实有答案"的条目不再计入统计（它们标错了，不是失败）
            if data.get("verdict") == "drop":
                continue
            rows.append(data)
    return rows


def run(rows: Sequence[Dict[str, Any]], reviewed_only: bool) -> List[Dict[str, Any]]:
    config = load_config(root=ROOT, use_dotenv=False)  # 评测必须离线可复现
    kb = KnowledgeBase(config)
    kb.build()
    agent = Agent(config=config, kb=kb, llm=MockLLM(), registry=build_registry(kb))

    results: List[Dict[str, Any]] = []
    for row in rows:
        if reviewed_only and not row.get("reviewed", False):
            continue
        case = EvalCase.from_dict(row)
        answer = agent.ask(case.question, role=case.role, history=case.history or None, reset=True)
        score = score_case(case, answer)
        results.append({"row": row, "case": case, "score": score, "answer": answer})
    return results


def pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def summarise(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    negatives = [item for item in results if not item["case"].expect_answer]
    positives = [item for item in results if item["case"].expect_answer]
    answered_negatives = [item for item in negatives if not item["score"].refused]
    refused_positives = [item for item in positives if item["score"].refused]
    composite = sum(item["score"].composite for item in results) / len(results) if results else 0.0
    return {
        "n": len(results),
        "negatives": len(negatives),
        "positives": len(positives),
        "answered_negatives": len(answered_negatives),
        "refused_positives": len(refused_positives),
        "hallucination_rate": len(answered_negatives) / len(negatives) if negatives else 0.0,
        "over_refusal_rate": len(refused_positives) / len(positives) if positives else 0.0,
        "composite": composite,
        "answered_examples": [item["case"].question for item in answered_negatives],
        "refused_examples": [item["case"].question for item in refused_positives],
    }


def main() -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="按题型分层评测拒答/误拒")
    parser.add_argument("--cases", nargs="+", help="评测集 JSONL（默认为 hard negative 的 dev+test）")
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="只统计 reviewed=true 的用例（人工复核后才算数）",
    )
    parser.add_argument("--show", type=int, default=5, help="每组最多列出几条误答示例")
    args = parser.parse_args()

    paths = [Path(p) for p in args.cases] if args.cases else DEFAULT_CASES
    rows = load_rows(paths)
    reviewed = sum(1 for row in rows if row.get("reviewed", False))
    print(f"评测集：{', '.join(p.name for p in paths)}")
    print(f"用例 {len(rows)} 条（已复核 {reviewed} 条）")
    if args.reviewed_only:
        print("只统计 reviewed=true 的用例")
    elif reviewed < len(rows):
        print("⚠ 还有未复核的用例，结果仅供参考 —— 复核后加 --reviewed-only 再跑一次")
    print()

    results = run(rows, reviewed_only=args.reviewed_only)
    if not results:
        print("没有可统计的用例（是否全部未复核？）")
        return 1

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        groups.setdefault(item["row"]["kind"], []).append(item)

    header = f"{'类型':<18}{'n':>4}{'负例':>5}{'误答':>10}{'95%CI':>18}{'正例':>5}{'误拒':>8}{'复合分':>9}"
    print(header)
    print("-" * len(header))
    for kind, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        summary = summarise(items)
        low, high = wilson(summary["answered_negatives"], summary["negatives"])
        ci = f"[{pct(low)}, {pct(high)}]" if summary["negatives"] else "—"
        print(
            f"{kind:<18}{summary['n']:>4}{summary['negatives']:>5}"
            f"{pct(summary['hallucination_rate']):>10}{ci:>18}"
            f"{summary['positives']:>5}{pct(summary['over_refusal_rate']):>8}"
            f"{summary['composite']:>9.4f}"
        )

    overall = summarise(results)
    low, high = wilson(overall["answered_negatives"], overall["negatives"])
    print("-" * len(header))
    print(
        f"{'总体':<18}{overall['n']:>4}{overall['negatives']:>5}"
        f"{pct(overall['hallucination_rate']):>10}{f'[{pct(low)}, {pct(high)}]':>18}"
        f"{overall['positives']:>5}{pct(overall['over_refusal_rate']):>8}{overall['composite']:>9.4f}"
    )

    print("\n按 hardness 再看一次（当前 gate 判定的难度分层，非人工标注）：")
    by_hardness: Dict[str, List[Dict[str, Any]]] = {}
    for item in results:
        by_hardness.setdefault(item["row"]["hardness"], []).append(item)
    for hardness, items in sorted(by_hardness.items(), key=lambda kv: -len(kv[1])):
        summary = summarise(items)
        print(
            f"  {hardness:<24}n={summary['n']:<4}误答 {summary['answered_negatives']}/{summary['negatives']}"
            f"  误拒 {summary['refused_positives']}/{summary['positives']}"
        )

    if overall["answered_examples"]:
        print(f"\n负例误答示例（前 {args.show} 条，说明 gate 把什么当成了依据）：")
        for question in overall["answered_examples"][: args.show]:
            print(f"  · {question}")
    if overall["refused_examples"]:
        print("\n正例被误拒（说明 gate 太保守，或标注的角色不对）：")
        for question in overall["refused_examples"][: args.show]:
            print(f"  · {question}")

    print("\n对照：README 上报告的 hallucination_rate=0.143 是在**原 21 条**（其中 6 条负例、")
    print("大多为 easy negative）上算出来的。同一套逻辑在这批 hard negative 上明显更高 ——")
    print("这不是系统变差了，而是**原来的题太容易**、把问题掩盖了。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
