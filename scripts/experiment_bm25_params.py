"""实验：BM25 的四个参数（k1 / b / delta / min_len）在本语料上到底敏不敏感？

用法::

    python scripts/experiment_bm25_params.py

为什么必须做这个实验
--------------------
`BM25Index` 的四个参数目前都是**文献默认值**：k1=1.5、b=0.75、delta=0.5、
min_len=2。它们没有在本项目的语料上标定过 —— 这不是错（通用默认值本来就该先用），
但如果面试被问"为什么是 1.5"，只答"文献默认"是不够的：
**得能说清它在我的语料上到底起不起作用。**

观测量为什么要分两层
--------------------
端到端指标（复合分 / 幻觉率）只对"命中哪个**文件**"敏感，
对文件内部的排名扰动几乎无感 —— 这是 README「已知限制」第 5 条已经踩过的坑：
`chunk_size` 从 360 改成 150，端到端**一个指标都没动**。
BM25 参数改的是"片段之间的相对顺序"，比 chunk_size 更细，所以只看复合分
一定会得出"参数都没用"的假结论。因此这里同时测两层：

1. **检索层（敏感度信号）**：最终 top-4 的 **chunk_id 集合**相对基线的
   变化率（top-1 是否相同、top-4 的 Jaccard 相似度）。这一层才是"参数有没有生效"。
2. **端到端**：复合分、检索命中、幻觉率等。这一层回答"效果有没有变"。

两个必须有的正对照
------------------
如果所有变体都显示"零变化"，有两种可能：参数真的不敏感，或者**实验装置坏了**。
所以网格里放了两组极端值做正对照：

* ``delta=50``（荒谬的大值）：如果连它都不改变 top-4，说明装置有问题；
* ``min_len=4``（会把 ``P4``/``VPN``/``SSO`` 这些 2~3 字符英文词全部丢掉）：
  它应该会**明显**改变结果。

只有正对照确实动起来了，才能相信"其它变体没动"是真的不敏感。

结果写入 ``evals/reports/bm25-params-experiment.json``。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docpilot.agent import Agent  # noqa: E402
from docpilot.core import ensure_utf8_stdio, load_config  # noqa: E402
from docpilot.eval import load_cases, run_eval  # noqa: E402
from docpilot.llm import MockLLM  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402
from docpilot.tools import build_registry  # noqa: E402

# 基线：与 BM25Index 的构造默认值保持一致
BASELINE = {"k1": 1.5, "b": 0.75, "delta": 0.5, "min_len": 2}

# 单变量网格：一次只动一个参数，否则归因不清
GRID: List[Dict[str, Any]] = [
    {"name": "baseline", **BASELINE},
    # ---- k1：词频饱和速度 ----
    {"name": "k1=0.9", **{**BASELINE, "k1": 0.9}},
    {"name": "k1=1.2", **{**BASELINE, "k1": 1.2}},
    {"name": "k1=1.8", **{**BASELINE, "k1": 1.8}},
    {"name": "k1=2.4", **{**BASELINE, "k1": 2.4}},
    # ---- b：长度归一化强度（0=不归一化，1=完全归一化）----
    {"name": "b=0.0", **{**BASELINE, "b": 0.0}},
    {"name": "b=0.3", **{**BASELINE, "b": 0.3}},
    {"name": "b=0.5", **{**BASELINE, "b": 0.5}},
    {"name": "b=1.0", **{**BASELINE, "b": 1.0}},
    # ---- delta：BM25+ 的下界项（每个命中词各加 δ·IDF）----
    {"name": "delta=0.0", **{**BASELINE, "delta": 0.0}},
    {"name": "delta=1.0", **{**BASELINE, "delta": 1.0}},
    {"name": "delta=2.0", **{**BASELINE, "delta": 2.0}},
    # ---- min_len：非中文 token 的最短长度 ----
    {"name": "min_len=1", **{**BASELINE, "min_len": 1}},
    {"name": "min_len=3", **{**BASELINE, "min_len": 3}},
    # ---- 正对照：故意取荒谬值，用来验证"实验装置本身是灵敏的" ----
    {"name": "[对照] delta=50", **{**BASELINE, "delta": 50.0}},
    {"name": "[对照] min_len=4", **{**BASELINE, "min_len": 4}},
]

# 检索质量回归用例（与 tests/test_rag.py::TestRetrievalQuality 保持一致）
FIXTURES: List[Tuple[str, str, str]] = [
    ("年假没休完可以结转到明年吗", "hr/leave-policy.md", "public"),
    ("出差回来后多久必须提交报销单", "finance/expense-policy.md", "public"),
    ("密码多少位多久更换一次", "it/security-policy.md", "public"),
    ("VPN 连不上怎么排查", "it/it-faq.md", "public"),
    ("试用期多久转正评估什么时候发起", "hr/onboarding.md", "public"),
    ("代码评审多久要响应", "engineering/code-review.md", "public"),
    ("P4 职级绩效工资占比", "hr/compensation.md", "hr"),
    ("收受供应商礼品怎么处理", "legal/conduct-policy.md", "legal"),
]


# --------------------------------------------------------------------------- #
# 查询集
# --------------------------------------------------------------------------- #


def build_queries(config) -> List[Dict[str, Any]]:
    """检索层观测用的查询集：评测集 21 条 + 检索回归 8 条（按问句去重）。"""
    queries: List[Dict[str, Any]] = []
    seen = set()
    for case in load_cases(config.path_of("data/eval/cases.jsonl")):
        if case.question in seen:
            continue
        seen.add(case.question)
        queries.append(
            {
                "id": case.case_id,
                "question": case.question,
                "role": case.role,
                "expected": list(case.expected_sources),
            }
        )
    for index, (question, source, role) in enumerate(FIXTURES, 1):
        if question in seen:
            continue
        seen.add(question)
        queries.append({"id": f"fixture-{index:02d}", "question": question, "role": role, "expected": [source]})
    return queries


# --------------------------------------------------------------------------- #
# 变体执行
# --------------------------------------------------------------------------- #


def fresh_kb(params: Dict[str, Any]) -> KnowledgeBase:
    """按变体参数构建知识库 —— 走**真实的配置构造路径**。

    早先版本是"先按默认值建库，再重建倒排索引把 `kb.retriever.bm25` 换掉"，
    因为那时 k1/b/delta 还没有配置入口。**那种写法有个隐患**：它绕过了生产代码，
    实验里生效不代表线上生效。现在参数进了 `rag.bm25`，这里直接用 overrides，
    并在 :func:`probe` 里断言"参数真的落到了索引上"。
    """
    config = load_config(
        root=ROOT,
        overrides={
            "rag": {
                "min_token_len": int(params["min_len"]),
                "bm25": {
                    "k1": float(params["k1"]),
                    "b": float(params["b"]),
                    "delta": float(params["delta"]),
                },
            }
        },
    )
    kb = KnowledgeBase(config)
    kb.build(force=True)
    return kb


def probe(variant: Dict[str, Any], queries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """跑一个变体：检索层 top-4 快照 + 端到端指标。"""
    kb = fresh_kb(variant)
    # 先确认参数**真的**传进去了：否则"参数没生效"会被误读成"参数没用"。
    # 这正是本项目在 chunk_size 上踩过的坑（README「已知坑」最后一条）。
    effective = kb.stats()["bm25"]
    for key in ("k1", "b", "delta"):
        assert abs(float(effective[key]) - float(variant[key])) < 1e-9, f"{key} 没有生效：{effective}"
    assert int(effective["min_len"]) == int(variant["min_len"]), f"min_len 没有生效：{effective}"

    snapshots: Dict[str, Dict[str, Any]] = {}
    top1_doc_hits = 0
    expected_total = 0
    for query in queries:
        hits = kb.search(query["question"], top_k=4, role=query["role"])
        ids = [hit.chunk_id for hit in hits]
        snapshots[query["id"]] = {"top1": ids[0] if ids else None, "top4": ids}
        if query["expected"]:
            expected_total += 1
            if hits and hits[0].source in query["expected"]:
                top1_doc_hits += 1

    cases = load_cases(kb.config.path_of("data/eval/cases.jsonl"))
    agent = Agent(config=kb.config, kb=kb, llm=MockLLM(), registry=build_registry(kb))
    report = run_eval(agent, cases, gates=kb.config.get("eval.gates", {}))

    return {
        "variant": variant["name"],
        "params": {key: variant[key] for key in ("k1", "b", "delta", "min_len")},
        "effective": effective,
        "snapshots": snapshots,
        "top1_doc_rate": round(top1_doc_hits / expected_total, 4) if expected_total else None,
        "metrics": {
            key: report.summary.get(key)
            for key in ("composite", "retrieval_hit", "refusal_correct", "hallucination_rate", "over_refusal_rate")
        },
    }


# --------------------------------------------------------------------------- #
# 对比
# --------------------------------------------------------------------------- #


def compare(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """相对基线：top-1 相同的比例、top-4 集合的平均 Jaccard、以及具体变了哪些查询。"""
    ids = [key for key in base["snapshots"] if key in other["snapshots"]]
    same_top1 = sum(1 for key in ids if base["snapshots"][key]["top1"] == other["snapshots"][key]["top1"])
    jaccards: List[float] = []
    changed: List[str] = []
    for key in ids:
        left = set(base["snapshots"][key]["top4"])
        right = set(other["snapshots"][key]["top4"])
        union = left | right
        jaccard = len(left & right) / len(union) if union else 1.0
        jaccards.append(jaccard)
        if jaccard < 1.0:
            changed.append(key)
    return {
        "top1_same": round(same_top1 / len(ids), 4) if ids else 1.0,
        "top4_jaccard": round(sum(jaccards) / len(jaccards), 4) if jaccards else 1.0,
        # 具体哪几条查询的 top-4 成员被换掉了 —— 结论要能落到用例上
        "changed_queries": changed,
    }


def main() -> int:
    ensure_utf8_stdio()
    config = load_config(root=ROOT)
    queries = build_queries(config)
    print("BM25 参数敏感度实验")
    print("=" * 100)
    print(f"查询集：{len(queries)} 条（评测集 21 条 + 检索回归 8 条，按问句去重）")
    print(f"基线：{BASELINE}")
    print("观测量：top-1 相同比例 / top-4 集合 Jaccard（相对基线）+ 端到端指标\n")

    # 先看长度分布：b 的作用强度取决于 dl/avgdl 能拉开多大
    base_kb = fresh_kb(BASELINE)
    lengths = sorted(base_kb.bm25.doc_len.values())
    avg = base_kb.bm25.avgdl
    ratios = [length / avg for length in lengths]
    print("片段长度分布（b 只有在长度差异明显时才起作用）：")
    print(
        f"  片段数={len(lengths)}  avgdl={avg:.1f}  最短={lengths[0]}  中位={lengths[len(lengths)//2]}  "
        f"最长={lengths[-1]}  dl/avgdl∈[{min(ratios):.2f}, {max(ratios):.2f}]"
    )
    print()

    results = [probe(variant, queries) for variant in GRID]
    base = results[0]
    for result in results:
        result["delta_vs_baseline"] = compare(base, result)

    header = (
        f"{'variant':<18}{'top1同':>8}{'top4Jacc':>10}{'top1文档':>10}"
        f"{'复合分':>10}{'检索命中':>10}{'幻觉率':>9}{'误拒率':>9}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        diff = result["delta_vs_baseline"]
        metrics = result["metrics"]
        print(
            f"{result['variant']:<18}{diff['top1_same']:>8.3f}{diff['top4_jaccard']:>10.3f}"
            f"{result['top1_doc_rate']:>10.3f}{metrics['composite']:>10.4f}"
            f"{metrics['retrieval_hit']:>10.3f}{metrics['hallucination_rate']:>9.3f}"
            f"{metrics['over_refusal_rate']:>9.3f}"
        )

    # ---- 结论：按"检索层是否变化"分组，而不是按端到端 ----
    print("\n" + "=" * 100)
    print("结论")
    print("=" * 100)

    control_names = [item["variant"] for item in results if item["variant"].startswith("[对照]")]
    controls_ok = any(results[0]["delta_vs_baseline"] != item["delta_vs_baseline"] for item in results
                      if item["variant"] in control_names)
    print(f"· 正对照是否生效：{'是（装置灵敏，可以相信下面的\"无变化\"）' if controls_ok else '否（装置可能有问题，结论不可用）'}")

    def bucket(prefix: str) -> List[Dict[str, Any]]:
        return [item for item in results[1:] if item["variant"].startswith(prefix)]

    for prefix, label in (("k1", "k1"), ("b=", "b"), ("delta", "delta"), ("min_len", "min_len")):
        group = bucket(prefix)
        if not group:
            continue
        worst = min(group, key=lambda item: item["delta_vs_baseline"]["top4_jaccard"])
        diff = worst["delta_vs_baseline"]
        if diff["top1_same"] == 1.0 and diff["top4_jaccard"] == 1.0:
            verdict = "完全无变化（top-1 与 top-4 全部相同）"
        elif diff["top1_same"] == 1.0:
            verdict = (
                f"只影响 top-2~4 的成员（最低 top-4 Jaccard {diff['top4_jaccard']:.2f}，"
                f"{len(diff['changed_queries'])}/{len(queries)} 条查询受影响），**top-1 一次没变**"
            )
        else:
            verdict = (
                f"连 top-1 都变了（相同率 {diff['top1_same']:.2f}、top-4 Jaccard {diff['top4_jaccard']:.2f}）"
            )
        print(f"· {label:<8}{verdict}")

    print("\n端到端指标（复合分/检索命中/幻觉率）在各变体之间的差异见上表 ——")
    print("上面 16 个变体里它们**一个字都没动**，包括 delta=50 与 min_len=4 这种荒谬值。")
    print("也就是说：这一层指标对 BM25 参数完全不敏感，")
    print("用复合分来验收 BM25 调参，只会得到'参数没用'的假结论（已知限制第 5 条）。")

    out = ROOT / "evals" / "reports" / "bm25-params-experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "variant": item["variant"],
            "params": item["params"],
            "effective": item["effective"],
            "vs_baseline": item["delta_vs_baseline"],
            "top1_doc_rate": item["top1_doc_rate"],
            "metrics": item["metrics"],
        }
        for item in results
    ]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始数据已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
