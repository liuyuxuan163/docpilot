"""用真实模型跑一遍 hard negative 全集，量出"真实模型下的误答率/误拒率"。

用法::

    python scripts/measure_real_llm_hard.py                  # 主集（77 条已复核）
    python scripts/measure_real_llm_hard.py --split holdout  # 第三份（31 条）
    python scripts/measure_real_llm_hard.py --workers 8

为什么必须单独做这个测量
------------------------
`eval_stratified.py` 用的是 Mock 基线：Mock 把 `has_evidence` 当**唯一**依据，
gate 说"有依据"它就把 top-1 片段抄进答案 —— 所以 89.8% 的误答率本质是
**「词面 gate 误判 × 无判断力的摘要器」**，不是产品行为。

真实模型多了一道判断：读到片段就能发现"这没回答我的问题"。所以必须实测，
不能用 Mock 的数字代替。

两个刻意的设计
--------------
1. **同时用两套"拒答"判定**：
   * 项目自带的 ``_looks_like_refusal``（关键词表，是为 Mock 的话术写的）；
   * 一个**更宽**的检测器（"没有提到 / 未包含 / 暂无 / 无法提供 …"）。
   两者不一致的条目会单独列出来人工看 —— 因为"拒答被误判成作答"
   会直接虚高误答率，这种偏差必须暴露而不是掩盖。
2. **原始答案全部落盘**（``data/eval/real_llm_answers.json``）：
   数字可以复算，但"它到底怎么答的"只有原文能说明。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from docpilot.agent import Agent  # noqa: E402
from docpilot.agent import _looks_like_refusal  # noqa: E402
from docpilot.core import ensure_utf8_stdio, load_config  # noqa: E402
from docpilot.eval import EvalCase, score_case  # noqa: E402
from docpilot.llm import build_llm  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402
from docpilot.tools import build_registry  # noqa: E402

MAIN = ROOT / "data" / "eval" / "hard_negatives.candidates.jsonl"
HOLDOUT = ROOT / "data" / "eval" / "hard_negatives.holdout.jsonl"
# 按 split 分文件：否则跑完 holdout 会把主集的原始答案覆盖掉（踩过一次）
OUT_DIR = ROOT / "data" / "eval"

# 比项目自带检测器更宽的一套：真实模型的说法比 Mock 多得多
WIDE_REFUSAL = (
    "无法回答", "无法提供", "无法给出", "没有找到", "未找到", "没有检索到",
    "没有相关", "没有关于", "没有提及", "未提及", "没有提到", "未包含", "不包含",
    "未涉及", "没有涉及", "暂无", "缺乏依据", "不足以", "不做推测", "不能确定",
    "没有访问权限", "无权访问", "涉及受限文档", "建议咨询", "建议向",
)


def looks_like_refusal_wide(text: str) -> bool:
    return any(phrase in (text or "") for phrase in WIDE_REFUSAL)


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_rows(split: str, reviewed_only: bool, cases: str | None = None) -> List[Dict[str, Any]]:
    path = Path(cases) if cases else (HOLDOUT if split == "holdout" else MAIN)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("//")]
    if reviewed_only and any("reviewed" in row for row in rows):
        # 老评测集（cases.jsonl）没有 reviewed 字段，不能因为"没复核"就被清空
        rows = [row for row in rows if row.get("reviewed") and row.get("verdict", "keep") != "drop"]
    else:
        rows = [row for row in rows if row.get("verdict", "keep") != "drop"]
    return rows


def run_one(config, kb, row: Dict[str, Any], retries: int = 2) -> Dict[str, Any]:
    """一条用例跑一次真实模型（失败重试，避免并发限流把结果搞成 0）。"""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            agent = Agent(config=config, kb=kb, llm=build_llm(config), registry=build_registry(kb))
            # 老评测集在 role=public 时**省略该字段**，直接取会 KeyError（踩过一次）；
            # history 也必须传，否则多轮追问题会被模型合理地判成"没有上文"
            answer = agent.ask(
                row["question"],
                role=row.get("role", "public"),
                history=row.get("history") or None,
                reset=True,
            )
            case = EvalCase.from_dict(row)
            score = score_case(case, answer)
            return {
                "case_id": row["case_id"],
                "kind": row.get("kind", "legacy"),
                "expect_answer": bool(row.get("expect_answer", True)),
                "question": row["question"],
                "text": answer.text,
                "refused": answer.refused,
                "refused_wide": looks_like_refusal_wide(answer.text),
                "looks_like_refusal": _looks_like_refusal(answer.text),
                "citations": [c.label for c in answer.citations],
                "tokens": answer.tokens,
                "latency_ms": round(answer.latency_ms, 1),
                "steps": answer.steps,
                "tool_calls": answer.tool_calls,
                "composite": score.composite,
                # 存全量分数：真实模型下"零伪造引用"这类卖点必须能核验，
                # 只留 composite 就等于把答案扔了
                "score": score.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 + attempt * 2)
    return {
        "case_id": row["case_id"],
        "kind": row.get("kind", "legacy"),
        "expect_answer": bool(row.get("expect_answer", True)),
        "question": row["question"],
        "error": last_error,
    }


def main() -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="真实模型跑 hard negative 全集")
    parser.add_argument("--split", choices=["main", "holdout"], default="main")
    parser.add_argument("--cases", help="直接指定评测集文件（例：data/eval/cases.jsonl）")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--include-unreviewed", action="store_true")
    args = parser.parse_args()

    config = load_config(root=ROOT, use_dotenv=True)
    if config.provider == "mock":
        raise SystemExit("需要真实模型：请在 .env 或 --provider 中指定 deepseek")

    rows = load_rows(args.split, reviewed_only=not args.include_unreviewed, cases=args.cases)
    label = Path(args.cases).stem if args.cases else args.split
    print(f"split={label} · 用例 {len(rows)} 条 · 模型 {config.model} · 并发 {args.workers}")
    kb = KnowledgeBase(config)
    kb.build()

    started = time.perf_counter()
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, config, kb, row): row for row in rows}
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 10 == 0 or done == len(rows):
                print(f"  …{done}/{len(rows)}（{time.perf_counter() - started:.0f}s）")

    errors = [item for item in results if "error" in item]
    if errors:
        print(f"\n⚠ {len(errors)} 条失败，前 3 条错误：")
        for item in errors[:3]:
            print(f"  · {item['case_id']}: {item['error'][:220]}")
    ok = [item for item in results if "error" not in item]
    out = OUT_DIR / f"real_llm_answers.{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # 错误也落盘，否则"18 条失败"会随控制台一起消失
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    results = ok

    neg = [item for item in results if not item["expect_answer"]]
    pos = [item for item in results if item["expect_answer"]]

    def rate(rows_: List[Dict[str, Any]], field: str) -> Tuple[int, int]:
        return sum(1 for item in rows_ if item[field]), len(rows_)

    print("\n" + "=" * 92)
    print(f"真实模型结果（{len(results)} 条成功、{len(errors)} 条失败）· 总耗时 {time.perf_counter()-started:.0f}s")
    print("=" * 92)
    print(f"{'判定器':<22}{'负例误答（该拒却答）':<34}{'正例误拒（该答却拒）'}")
    print("-" * 92)
    for label, field, invert in (
        ("项目检测器", "refused", False),
        ("宽检测器", "refused_wide", False),
    ):
        answered_neg = sum(1 for item in neg if not item[field])
        refused_pos = sum(1 for item in pos if item[field])
        low, high = wilson(answered_neg, len(neg))
        print(
            f"{label:<22}{f'{answered_neg}/{len(neg)} ({answered_neg/len(neg):.1%})':<34}"
            f"{f'{refused_pos}/{len(pos)} ({refused_pos/len(pos):.1%})'}"
        )
        print(f"{'  95% CI':<22}[{low:.1%}, {high:.1%}]")

    answered_neg = [item for item in neg if not item["refused"]]
    print(f"\n负例里仍被判为『作答』的 {len(answered_neg)} 条（这些是真的误答，要逐条看）：")
    for item in answered_neg:
        flag = "" if item["refused_wide"] else "  ⚠宽检测器也说没拒答"
        print(f"\n  · {item['case_id']}  [{item['kind']}]  {item['question']}{flag}")
        body = " ".join(item["text"].split())[:220]
        print(f"    {body}…")

    mismatched = [item for item in neg if item["refused"] and not item["refused_wide"]]
    if mismatched:
        print(f"\n⚠ 项目检测器判为拒答、但宽检测器没命中的 {len(mismatched)} 条（可能是措辞没覆盖）：")
        for item in mismatched[:5]:
            print(f"  · {item['question']} → {' '.join(item['text'].split())[:100]}")

    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for item in neg:
        by_kind.setdefault(item["kind"], []).append(item)
    print("\n按类型（负例误答率）：")
    for kind, items in sorted(by_kind.items(), key=lambda kv: -len(kv[1])):
        answered = sum(1 for item in items if not item["refused"])
        print(f"  {kind:<18}{answered}/{len(items)} ({answered/len(items):.0%})")

    total_tokens = sum(item.get("tokens", 0) for item in results)
    print(f"\n总 token：{total_tokens:,}（约 {total_tokens/1000:.0f}k）")
    print(f"原始答案已写入：{out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
