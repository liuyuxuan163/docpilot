"""最终阈值验证：确认 (rel >= 0.28, span >= 2) 在标注集上零误拒。

这个脚本是**可复现的标定证据**，不是临时调试代码：
任何人改动了检索或匹配逻辑后，都应该重跑它，确认误拒没有回归。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docpilot.core import ensure_utf8_stdio, load_config
from docpilot.rag import KnowledgeBase

# 应回答（in-scope）
IN_SCOPE = [
    ("public", "新员工入职第一年年假有几天？怎么折算的？"),
    ("public", "年假没休完可以结转到明年吗？最多结转多少天？"),
    ("public", "出差回来后多久必须提交报销单？"),
    ("public", "公司密码要求多少位？多久必须更换一次？"),
    ("hr", "P4 职级的绩效工资占比是多少？"),
    ("public", "出差住酒店超预算了怎么办？"),
    ("public", "住宿超标了要自己承担吗？"),
    ("hr", "试用期是多久？"),
    ("public", "公司的VPN连不上，我自己能怎么排查？"),
    ("public", "可以把公司代码贴到外面的AI工具上问吗？"),
    ("public", "那最多能结转到下一年多少天？"),
    ("public", "年假没休完可以结转到明年吗？"),
    ("public", "密码有什么复杂度要求？"),
    ("public", "出差住宿一线城市 P4 职级的标准是多少？"),
    ("public", "报销单多久内要提交？"),
    ("public", "软件安装申请要多久审批？"),
]
# 应拒答（out-of-scope / 无权限）
OUT_OF_SCOPE = [
    ("public", "公司的股票代码是多少？"),
    ("public", "公司附近哪家餐厅比较好吃？"),
    ("public", "公司明年会不会裁员？"),
    ("public", "竞争对手的核心算法怎么实现的？"),
    ("public", "今天天气怎么样？"),
    ("public", "午饭吃什么好？"),
    ("public", "比特币现在多少钱？"),
    ("public", "P4 职级的绩效工资占比是多少？"),
    ("public", "收受供应商价值500元的礼品要怎么处理？"),
]


def main() -> int:
    # 这个脚本会在末尾打印 ✅/❌。Windows 的 GBK 控制台一旦接管道，
    # 那两个符号会抛 UnicodeEncodeError —— 一条**已经验证完零误拒**的命令
    # 会以退出码 1 结束，被误读成"出现回归"。所以先切 UTF-8。
    ensure_utf8_stdio()
    config = load_config()
    kb = KnowledgeBase(config)
    kb.build()
    rel_th = float(config.get("agent.min_evidence_relevance", 0.28))
    span_th = int(config.get("agent.min_evidence_span", 2))
    print(f"阈值：relevance >= {rel_th} 且 longest_span >= {span_th} 字符\n")

    false_refusals = []
    hallucinations = []
    print(f"{'rel':>7} {'span':>5} {'判定':<7} 问题")
    for role, query in IN_SCOPE + OUT_OF_SCOPE:
        expected_answer = (role, query) in IN_SCOPE
        hits = kb.search(query, top_k=4, role=role)
        rel = float(hits[0].metadata.get("relevance", 0.0)) if hits else 0.0
        span = int(hits[0].metadata.get("longest_span_chars", 0)) if hits else 0
        restricted = kb.restricted_matches(query, role=role)
        decided = bool(hits) and not restricted and rel >= rel_th and span >= span_th
        mark = "作答" if decided else "拒答"
        flag = ""
        if expected_answer and not decided:
            false_refusals.append(query)
            flag = "  ← 误拒（该答却拒）"
        if not expected_answer and decided:
            hallucinations.append(query)
            flag = "  ← 幻觉（该拒却答）"
        print(f"{rel:>7.3f} {span:>5} {mark:<7} {query[:32]}{flag}")

    total = len(IN_SCOPE) + len(OUT_OF_SCOPE)
    print(f"\n用例 {total} 条：误拒 {len(false_refusals)} 条，幻觉 {len(hallucinations)} 条")
    if hallucinations:
        print("残余幻觉（词面启发式的已知边界，需要真实 embedding 才能解决）：")
        for query in hallucinations:
            print(f"  · {query}")
    if false_refusals:
        print("❌ 出现误拒，说明阈值需要重新标定 —— 这是回归，必须修复")
        return 1
    print("✅ 无误拒：用户问得到答案时一定会回答")
    return 0


if __name__ == "__main__":
    sys.exit(main())
