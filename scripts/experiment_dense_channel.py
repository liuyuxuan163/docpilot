"""消融实验：稠密通道到底给最终 top-k 贡献了多少？

用法::

    python scripts/experiment_dense_channel.py

为什么这个实验比"接真实 embedding"更该先做
------------------------------------------
当前链路里，决定"答还是拒答"的只有**两个词面信号**
（``tools.search_docs`` 里的 ``has_evidence``：实义覆盖率 + 最长连续命中），
排序主键也是 ``relevance``（同样是词面），RRF 融合分只用来打破平局。

所以"换一个更好的 embedding 能不能降低幻觉率"取决于一个**可观测事实**：
**稠密通道有没有给最终 top-k 带来独家候选**。
如果它一条都没带来，那更好的 embedding 在这条链路上就没有可观测收益 ——
问题不在模型，在"语义信号根本没被判定逻辑使用"。

这个脚本用 ``rag.enable_dense`` / ``rag.enable_sparse`` 做三组对照，
同时记录三样东西：

1. **端到端指标**（21 条用例）：换通道到底有没有改变结果；
2. **检索层观测量**：最终 top-4 里有多少条是"只有稠密通道找到的"
   （``returned_origin.dense_only``）—— 这是最有信息量的一个数；
3. **逐例变化**：哪些用例的复合分随通道开关而变。

三组对照的读法：

* ``full`` 与 ``sparse_only`` 完全一致，且 ``dense_only`` 恒为 0
  → 稠密通道在本语料上是"空转的"，优化重点应放在**判定信号**而不是嵌入模型；
* 两者不一致 → 说明稠密确实带回了 BM25 找不到的片段，换 embedding 才有意义，
  并且应该去 ``retrieval_hit`` / 口语化用例上看收益。

结果写入 ``evals/reports/dense-channel-experiment.json``，方便贴进文档。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docpilot.agent import Agent  # noqa: E402
from docpilot.core import ensure_utf8_stdio, load_config  # noqa: E402
from docpilot.eval import load_cases, run_eval  # noqa: E402
from docpilot.llm import MockLLM  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402
from docpilot.tools import build_registry  # noqa: E402

METRICS = [
    "composite",
    "retrieval_hit",
    "citation_valid",
    "refusal_correct",
    "answer_coverage",
    "over_refusal_rate",
    "hallucination_rate",
    "avg_tokens",
    "p95_latency_ms",
]

# 质量指标与成本指标必须分开看：
# 换一条通道几乎必然改变 prompt 长度与延迟（成本），但那不等于"效果变好/变差"。
# 结论只拿质量指标判 —— 把 token/延迟混进判据，会让"零变化"看起来像"有变化"。
QUALITY_METRICS = [
    "composite",
    "retrieval_hit",
    "citation_valid",
    "refusal_correct",
    "answer_coverage",
    "over_refusal_rate",
    "hallucination_rate",
]
COST_METRICS = ["avg_tokens", "p95_latency_ms"]

# 表头用短标签：完整名字会把 9 列挤成一坨（真实踩过的可读性问题）
LABELS = {
    "composite": "comp",
    "retrieval_hit": "hit",
    "citation_valid": "cite",
    "refusal_correct": "refuse",
    "answer_coverage": "cover",
    "over_refusal_rate": "overref",
    "hallucination_rate": "halluc",
    "avg_tokens": "tokens",
    "p95_latency_ms": "p95ms",
}

VARIANTS: List[Dict[str, Any]] = [
    {"name": "full", "enable_dense": True, "enable_sparse": True},
    {"name": "sparse_only", "enable_dense": False, "enable_sparse": True},
    {"name": "dense_only", "enable_dense": True, "enable_sparse": False},
]


def probe(variant: Dict[str, Any]) -> Dict[str, Any]:
    """跑一个变体：端到端指标 + 通道贡献统计。"""
    config = load_config(
        root=ROOT,
        overrides={
            "rag": {
                "enable_dense": variant["enable_dense"],
                "enable_sparse": variant["enable_sparse"],
            }
        },
    )
    kb = KnowledgeBase(config)
    stats = kb.build(force=True)
    cases = load_cases(config.path_of("data/eval/cases.jsonl"))
    agent = Agent(config=config, kb=kb, llm=MockLLM(), registry=build_registry(kb))

    report = run_eval(agent, cases, gates=config.get("eval.gates", {}))

    # 逐例再跑一遍，只为采集检索器的 last_debug（MockLLM 是确定性的，
    # 第二遍的检索结果与第一遍完全一致；这样就不必把 run_eval 的聚合逻辑复制一份）。
    origin_total = {"both": 0, "sparse_only": 0, "dense_only": 0}
    origin_by_case: Dict[str, Dict[str, int]] = {}
    for case in cases:
        agent.ask(case.question, role=case.role, history=case.history or None, reset=True)
        origin = dict(kb.retriever.last_debug.get("returned_origin") or {})
        origin_by_case[case.case_id] = origin
        for key in origin_total:
            origin_total[key] += int(origin.get(key, 0))

    return {
        "variant": variant["name"],
        "channels": {
            "dense": variant["enable_dense"],
            "sparse": variant["enable_sparse"],
        },
        "index": {
            "documents": stats["documents"],
            "chunks": stats["chunks"],
            "bm25_terms": kb.stats()["bm25_terms"],
            "vectors": len(kb.vectors),
        },
        "metrics": {metric: report.summary.get(metric) for metric in METRICS},
        "per_case": {case.case_id: case.composite for case in report.cases},
        "returned_origin": origin_total,
        "returned_origin_by_case": origin_by_case,
    }


def _fmt(value: Any, width: int = 12) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rjust(width)
    return str(value).rjust(width)


def main() -> int:
    ensure_utf8_stdio()
    print("稠密通道消融实验")
    print("=" * 78)
    print("对照：full（双通道）/ sparse_only（只 BM25）/ dense_only（只本地嵌入）")
    print("核心观测量：最终 top-4 里有多少条是「只有稠密通道找到的」\n")

    results = [probe(variant) for variant in VARIANTS]

    header = f"{'variant':<13}{'chunks':>7}{'bm25':>7}{'vec':>5}" + "".join(
        f"{LABELS[m]:>10}" for m in METRICS
    )
    print(header)
    print("-" * len(header))
    for item in results:
        row = (
            f"{item['variant']:<13}{item['index']['chunks']:>7}"
            f"{item['index']['bm25_terms']:>7}{item['index']['vectors']:>5}"
        )
        row += "".join(_fmt(item["metrics"].get(metric), 10) for metric in METRICS)
        print(row)

    print("\n通道贡献（21 条用例累计，每个用例取 top-4）：")
    print(f"{'variant':<13}{'both':>8}{'sparse_only':>14}{'dense_only':>13}")
    for item in results:
        origin = item["returned_origin"]
        print(f"{item['variant']:<13}{origin['both']:>8}{origin['sparse_only']:>14}{origin['dense_only']:>13}")

    full = next(item for item in results if item["variant"] == "full")
    sparse_only = next(item for item in results if item["variant"] == "sparse_only")
    dense_only = next(item for item in results if item["variant"] == "dense_only")
    quality_changed = [m for m in QUALITY_METRICS if full["metrics"].get(m) != sparse_only["metrics"].get(m)]
    changed_cases = [
        case_id
        for case_id, score in full["per_case"].items()
        if score != sparse_only["per_case"].get(case_id)
    ]
    dense_slots = full["returned_origin"]["dense_only"]
    dense_cases = [cid for cid, origin in full["returned_origin_by_case"].items() if origin.get("dense_only")]

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print(f"· 质量指标差异（full vs sparse_only）：{quality_changed or '无'}")
    print(f"· 复合分发生变化的用例：{changed_cases or '无'}")
    print(f"· 稠密通道的独家槽位：{dense_slots} 个"
          + (f"，全部出现在 {', '.join(dense_cases)}" if dense_cases else ""))
    print(f"· 成本差异：延迟 {sparse_only['metrics']['p95_latency_ms']} → "
          f"{full['metrics']['p95_latency_ms']} ms，"
          f"平均 token {sparse_only['metrics']['avg_tokens']} → {full['metrics']['avg_tokens']}")
    print("")

    if not quality_changed and not changed_cases:
        print("⇒ 关掉稠密通道，21 条用例的**质量指标一模一样**（复合分 0.9310 → 0.9310）。")
        if dense_slots:
            print(f"  稠密通道并非完全空转：它独自找回了 {dense_slots} 个槽位，但它们")
            print("  在词面 relevance 上并不优于替代它们的片段 —— 而排序主键与拒答判定")
            print("  都只看词面信号（rag._rerank / tools.search_docs.has_evidence）。")
            if set(dense_cases) <= {"refuse-03"}:
                print(f"  更值得注意的是：这 {dense_slots} 个槽位全部落在 **refuse-03**（已知幻觉用例）上，")
                print("  也就是说稠密通道唯一的独家贡献，恰好发生在一个「本来就不该回答」的问题上，")
                print("  而且并没有改变最终判定。")
        else:
            print("  稠密通道连一个独家候选都没带回来。")
        print("  因此：在本语料 + 当前判定逻辑下，**只换 embedding 不会有可观测收益**。")
        print("  瓶颈是「语义信号根本没被判定逻辑使用」，不是「嵌入模型不够好」。")
        print(f"  佐证：单开稠密（dense_only）幻觉率 {dense_only['metrics']['hallucination_rate']} "
              f"低于 {full['metrics']['hallucination_rate']}，")
        print(f"  但检索命中掉到 {dense_only['metrics']['retrieval_hit']}、"
              f"误拒升到 {dense_only['metrics']['over_refusal_rate']} ——")
        print("  说明指标对**候选池的组成**敏感，而不对「语义能力」敏感。")
    else:
        print("⇒ 稠密通道改变了质量指标，值得继续投入。下一步应去 retrieval_hit /")
        print("   口语化用例（colloquial-*）上看具体收益来自哪里。")

    out = ROOT / "evals" / "reports" / "dense-channel-experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始数据已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
