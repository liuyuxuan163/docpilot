"""检查拒答判定所依赖的两个信号（对应上手计划"实验 9"）。

这个脚本把原先要用 ``python -c`` 一行写完的逻辑固化成文件 ——
原因是那段代码需要三层引号（PowerShell 双引号 + f-string + 字典 key），
在命令行里必然被转义搞乱。

用法::

    python scripts/check_relevance_signals.py

它会对比"应回答"与"应拒答"两组问题上的两个信号：

* ``relevance``：IDF 加权的实义字符覆盖率（管"确实对上了"）
* ``longest_span_chars``：最长连续命中字符数（管"不是零散撞字"）

**要观察的核心现象**：单看 ``relevance`` 时两组是**重叠**的
（这就是当初"用覆盖率做判据"失败的原因），
而加上 ``longest_span_chars`` 后两组才被分开。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docpilot.core import load_config
from docpilot.rag import KnowledgeBase

# 应该回答的问题（文档里有明确依据）
IN_SCOPE = [
    ("public", "公司密码要求多少位？多久必须更换一次？"),
    ("public", "年假没休完可以结转到明年吗？"),
    ("public", "出差住酒店超预算了怎么办？"),
    ("public", "出差回来后多久必须提交报销单？"),
]

# 应该拒答的问题（知识库里没有依据）
OUT_OF_SCOPE = [
    ("public", "公司的股票代码是多少？"),
    ("public", "今天天气怎么样？"),
    ("public", "公司明年会不会裁员？"),
    ("public", "午饭吃什么好？"),
]


def probe(kb: KnowledgeBase, role: str, question: str) -> tuple[float, int]:
    """返回 (relevance, 最长连续命中字符数)。"""
    hits = kb.search(question, top_k=1, role=role)
    if not hits:
        return 0.0, 0
    top = hits[0]
    return (
        float(top.metadata.get("relevance", 0.0)),
        int(top.metadata.get("longest_span_chars", 0)),
    )


def main() -> int:
    config = load_config(root=ROOT)
    kb = KnowledgeBase(config)
    kb.build()

    threshold_relevance = float(config.get("agent.min_evidence_relevance", 0.28))
    threshold_span = int(config.get("agent.min_evidence_span", 2))

    rows: list[tuple[str, float, int, str]] = []

    print("=" * 82)
    print("应回答的问题（in-scope）")
    print("=" * 82)
    print(f"{'relevance':>10} {'span':>6}  {'判定':<6} 问题")
    for role, question in IN_SCOPE:
        relevance, span = probe(kb, role, question)
        decided = "作答" if (relevance >= threshold_relevance and span >= threshold_span) else "拒答"
        rows.append(("in", relevance, span, question))
        print(f"{relevance:>10.3f} {span:>6}  {decided:<6} {question}")

    print()
    print("=" * 82)
    print("应拒答的问题（out-of-scope）")
    print("=" * 82)
    print(f"{'relevance':>10} {'span':>6}  {'判定':<6} 问题")
    for role, question in OUT_OF_SCOPE:
        relevance, span = probe(kb, role, question)
        decided = "作答" if (relevance >= threshold_relevance and span >= threshold_span) else "拒答"
        rows.append(("out", relevance, span, question))
        print(f"{relevance:>10.3f} {span:>6}  {decided:<6} {question}")

    in_values = [r for kind, r, _, _ in rows if kind == "in"]
    out_values = [r for kind, r, _, _ in rows if kind == "out"]
    in_spans = [s for kind, _, s, _ in rows if kind == "in"]
    out_spans = [s for kind, _, s, _ in rows if kind == "out"]

    print()
    print("=" * 82)
    print("观察要点")
    print("=" * 82)
    print(f"  阈值：relevance >= {threshold_relevance} 且 span >= {threshold_span}")
    print()
    print(f"  单看 relevance：")
    print(f"      应回答组 min = {min(in_values):.3f}")
    print(f"      应拒答组 max = {max(out_values):.3f}")
    overlap = min(in_values) <= max(out_values)
    print(f"      → {'❌ 两组重叠，无论阈值定在哪都会错一批' if overlap else '✅ 可用单一阈值分开'}")
    print()
    print(f"  加上 longest_span_chars：")
    print(f"      应回答组 span = {sorted(in_spans)}")
    print(f"      应拒答组 span = {sorted(out_spans)}")
    separated = min(in_spans) > max(out_spans)
    print(f"      → {'✅ 两组被分开（这正是 AND 规则的价值）' if separated else '⚠️ 仍有重叠，需要更多信号'}")
    print()
    print("  结论：单看覆盖率区分不开（'今天天气怎么样'这类字面撞车），")
    print("        加上'最长连续命中'才能挡住'只共用两三个字'的提问。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
