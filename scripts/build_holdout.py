"""生成第三份数据（holdout）：专门覆盖"规则 + 具体情形"这类推理题。

用法::

    python scripts/build_holdout.py

为什么需要这第三份
------------------
前两份（dev 45 / test 32）已经被用过：dev 上调过 prompt，test 上看过一次结果。
在同一份数据上继续改 prompt，就是在污染它 —— 所以任何"修好了"的结论都必须
在**没见过的数据**上验证。

这份 holdout 的构成刻意偏斜：

* **大部分是正例**，且集中在判定器栽跟头的那一类：**文档给判定规则，
  问题问某个具体情形**。例：文档写「P1-P6 可乘坐二等座；P7 及以上一等座」，
  问「P3 能坐高铁一等座吗」—— 正确答案要靠规则推出来，文档里没有"不能"两个字。
* **少量负例**（约 1/3）保留幻觉维度，否则只看误拒率会得出"判定器越宽越好"的错误结论。

⚠ 所有记录 ``reviewed: false``：这份同样需要人工确认（约 10 分钟），
在确认前跑出来的数字只能当**方向性**参考。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_hard_negatives import (  # noqa: E402
    build_candidates,
    ensure_utf8_stdio,
    load_docs,
    render_review,
)
from docpilot.core import load_config  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402
from docpilot.tools import build_registry  # noqa: E402

OUT_DIR = ROOT / "data" / "eval"

# kind 只用一个新值，避免和主集混在一起统计：holdout_rule = 规则推理
HOLDOUT_TOPICS: List[Dict[str, Any]] = [
    # ---- 正例：规则 + 具体情形（判定器最容易误拒的一类）----
    {
        "topic": "高铁座席-P3",
        "kind": "holdout_rule",
        "anchor": "高铁/动车：P1-P6 可乘坐二等座；P7 及以上可乘坐一等座",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档给了规则，问题问具体职级；正确答案要由规则推出（不能）",
        "expected_keywords": ["二等座"],
        "negatives": [],
        "positives": ["P3 出差可以坐高铁一等座吗？"],
    },
    {
        "topic": "公务舱-P7",
        "kind": "holdout_rule",
        "anchor": "飞机：经济舱；P7 及以上可乘坐公务舱",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档给了规则；P7 可直接推出（可以）",
        "expected_keywords": ["公务舱"],
        "negatives": [],
        "positives": ["P7 出差可以坐公务舱吗？"],
    },
    {
        "topic": "住宿-P2一线",
        "kind": "holdout_rule",
        "anchor": "住宿表：P1-P3 一线城市 500 元/晚",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档给了表格，问题问具体职级+城市，查表可得",
        "expected_keywords": ["500"],
        "negatives": [],
        "positives": ["P2 在一线城市出差，住宿上限是多少？"],
    },
    {
        "topic": "住宿-P6二线",
        "kind": "holdout_rule",
        "anchor": "住宿表：P4-P6 二线城市 550 元/晚",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "查表可得 550",
        "expected_keywords": ["550"],
        "negatives": [],
        "positives": ["P6 去二线城市出差，住宿标准是多少？"],
    },
    {
        "topic": "实习生出差审批",
        "kind": "holdout_rule",
        "anchor": "实习生出差需由导师发起并附说明",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确写了实习生出差的发起人",
        "expected_keywords": ["导师"],
        "negatives": [],
        "positives": ["实习生出差需要谁来发起？"],
    },
    {
        "topic": "密码连续数字",
        "kind": "holdout_rule",
        "anchor": "密码不得包含姓名拼音、工号、连续数字（如 123456）",
        "docs": ["it/security-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确禁止连续数字",
        "expected_keywords": ["不得"],
        "negatives": [],
        "positives": ["公司密码可以使用 123456 这种连续数字吗？"],
    },
    {
        "topic": "账号锁定次数",
        "kind": "holdout_rule",
        "anchor": "连续输错 5 次密码会锁定 30 分钟",
        "docs": ["it/it-faq.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了次数与时长",
        "expected_keywords": ["5"],
        "negatives": [],
        "positives": ["密码连续输错几次会被锁定？"],
    },
    {
        "topic": "抓包工具审批时长",
        "kind": "holdout_rule",
        "anchor": "涉及网络代理/抓包类工具：IT 安全负责人，3 个工作日",
        "docs": ["it/it-faq.md"],
        "absent_keywords": [],
        "absent_reason": "查表可得 3 个工作日",
        "expected_keywords": ["3 个工作日"],
        "negatives": [],
        "positives": ["申请安装抓包工具要多久审批？"],
    },
    {
        "topic": "核心模块合并人数",
        "kind": "holdout_rule",
        "anchor": "核心模块（支付、权限、数据管道）需 2 位评审人 Approve",
        "docs": ["engineering/code-review.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 2 位",
        "expected_keywords": ["2"],
        "negatives": [],
        "positives": ["核心模块的 PR 需要几个人 Approve？"],
    },
    {
        "topic": "试用期薪资比例",
        "kind": "holdout_rule",
        "anchor": "试用期薪资为转正后的 90%",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 90%",
        "expected_keywords": ["90"],
        "negatives": [],
        "positives": ["试用期薪资是转正后的多少？"],
    },
    {
        "topic": "试用期延长幅度",
        "kind": "holdout_rule",
        "anchor": "延长试用期一次，不超过原试用期的 50%",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 50% 上限",
        "expected_keywords": ["50"],
        "negatives": [],
        "positives": ["试用期最多可以延长多久？"],
    },
    {
        "topic": "事假月度上限",
        "kind": "holdout_rule",
        "anchor": "事假每月不超过 3 天，每年不超过 15 天",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 3 天",
        "expected_keywords": ["3 天"],
        "negatives": [],
        "positives": ["事假每个月最多几天？"],
    },
    {
        "topic": "病假工资标准",
        "kind": "holdout_rule",
        "anchor": "病假工资按当地最低工资标准的 80% 计发",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 80%",
        "expected_keywords": ["80"],
        "negatives": [],
        "positives": ["病假期间工资按什么标准发？"],
    },
    {
        "topic": "年假连休规则",
        "kind": "holdout_rule",
        "anchor": "年假可与法定节假日连休；总天数超 10 个工作日需 HR 总监会签",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：可以，但有会签条件",
        "expected_keywords": ["可以"],
        "negatives": [],
        "positives": ["年假可以和法定节假日连着休吗？"],
    },
    {
        "topic": "保密期-离职后",
        "kind": "holdout_rule",
        "anchor": "保密义务离职后 2 年（核心技术秘密 3 年）",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 2 年/3 年",
        "expected_keywords": ["2 年"],
        "negatives": [],
        "positives": ["离职后保密义务还要持续多久？"],
    },
    {
        "topic": "匿名举报",
        "kind": "holdout_rule",
        "anchor": "合规举报邮箱 compliance@example.com 可匿名",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确写了可匿名",
        "expected_keywords": ["匿名"],
        "negatives": [],
        "positives": ["举报职场骚扰可以匿名吗？"],
    },
    {
        "topic": "礼品-800元",
        "kind": "holdout_rule",
        "anchor": "礼品界限：200 元以内可收；200-1000 需申报，原则上上交或退回；超 1000 不得收",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": [],
        "absent_reason": "800 元落在 200-1000 区间，规则明确",
        "expected_keywords": ["申报"],
        "negatives": [],
        "positives": ["收受供应商价值 800 元的礼品要怎么处理？"],
    },
    {
        "topic": "年终奖系数-C",
        "kind": "holdout_rule",
        "anchor": "年终奖个人系数：S=2.0、A=1.5、B=1.0、C=0.5、D=0",
        "docs": ["hr/compensation.md"],
        "absent_keywords": [],
        "absent_reason": "查表可得 0.5",
        "expected_keywords": ["0.5"],
        "negatives": [],
        "positives": ["绩效 C 级的年终奖个人系数是多少？"],
    },
    {
        "topic": "连续两次D",
        "kind": "holdout_rule",
        "anchor": "连续两个考核周期为 D，公司可依法启动岗位调整或解除劳动合同",
        "docs": ["hr/compensation.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确写了后果",
        "expected_keywords": ["调整", "解除"],
        "negatives": [],
        "positives": ["连续两个考核周期都是 D 会怎么样？"],
    },
    {
        "topic": "调薪窗口月份",
        "kind": "holdout_rule",
        "anchor": "年度调薪窗口：每年 3 月",
        "docs": ["hr/compensation.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 3 月",
        "expected_keywords": ["3 月"],
        "negatives": [],
        "positives": ["年度调薪一般几月进行？"],
    },
    {
        "topic": "绩效申诉时限",
        "kind": "holdout_rule",
        "anchor": "收到考核结果后 5 个工作日内可提交书面申诉",
        "docs": ["hr/compensation.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给了 5 个工作日",
        "expected_keywords": ["5 个工作日"],
        "negatives": [],
        "positives": ["对绩效结果申诉要在几天内提交？"],
    },
    # ---- 负例：同域缺口（保留幻觉维度）----
    {
        "topic": "年假抵病假",
        "kind": "sibling_slot",
        "anchor": "病假需二级以上医院证明；病假工资按最低工资 80%",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["抵扣", "抵用", "冲抵"],
        "absent_reason": "文档没有任何「用年假抵扣病假」的规定",
        "negatives": ["可以用年假抵扣病假吗？"],
    },
    {
        "topic": "加班审批",
        "kind": "synonym_only",
        "anchor": "SYNONYMS 里有「加班→延时工作/overtime」",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["加班申请", "加班审批", "延时工作申请"],
        "absent_reason": "文档没有加班申请/审批流程",
        "negatives": ["加班需要提前审批吗？"],
    },
    {
        "topic": "出差轮船",
        "kind": "sibling_slot",
        "anchor": "交通标准：市内交通、高铁/动车、飞机、自驾",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["轮船", "船票", "轮渡"],
        "absent_reason": "交通标准只列了四种方式，没有轮船",
        "negatives": ["出差可以乘坐轮船吗？船票能报销吗？"],
    },
    {
        "topic": "打印机彩色",
        "kind": "sibling_slot",
        "anchor": "打印机：确认 OFFICE-PRINT 网络、驱动下载地址",
        "docs": ["it/it-faq.md"],
        "absent_keywords": ["彩色", "彩打", "双面"],
        "absent_reason": "打印机章节只讲连接排查，没有彩色/双面等能力说明",
        "negatives": ["打印机支持彩色打印吗？"],
    },
    {
        "topic": "会议室预订",
        "kind": "sibling_slot",
        "anchor": "入职指引、IT FAQ 均无会议室相关流程",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["会议室", "预订", "预约会议"],
        "absent_reason": "全库没有会议室预订流程",
        "negatives": ["会议室怎么预订？"],
    },
    {
        "topic": "报销分次提交",
        "kind": "half_answer",
        "anchor": "报销单需在 15 个工作日内提交（单次流程）",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["分次", "分批", "多次提交"],
        "absent_reason": "只规定了提交时限与材料，没写能否分次提交",
        "negatives": ["一次出差的费用可以分几次提交报销吗？"],
    },
    {
        "topic": "转正再评估",
        # 原本标成负例（"文档没说能否再评估一次"），经人工判定改为**正例**：
        # 文档给出了失败后的两种后果，属于"有依据可依"，正确答案是复述这两种后果
        # 并说明"再评估一次"文档未规定 —— 这属于"可回答但要说边界"。
        "kind": "holdout_rule",
        "anchor": "转正评估未通过：可延长试用期一次（不超过原试用期 50%），或依法解除劳动合同",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["再次评估", "二次评估", "重新评估"],
        "rationale": "文档明确给出失败后的两种处置（延长一次≤50% / 解除劳动合同），属可回答；"
                     "「再评估一次」文档未规定，作答时应说明这条边界",
        "absent_reason": "文档没有「再评估一次」的直接规定，但给出了失败的处置方式，按口径算可回答",
        "expected_keywords": ["延长", "解除"],
        "negatives": [],
        "positives": ["转正评估没通过，可以再申请一次评估吗？"],
    },
    {
        "topic": "绩效影响社保",
        "kind": "sibling_slot",
        "anchor": "绩效系数影响绩效工资；薪酬=基本+绩效+补贴",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["社保基数", "缴费基数"],
        "absent_reason": "文档没有任何社保基数相关说明",
        "negatives": ["绩效结果会影响社保缴费基数吗？"],
    },
    {
        "topic": "回扣法律责任",
        "kind": "half_answer",
        "anchor": "收受供应商回扣或贿赂属严重违纪（可解除、无补偿）",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": ["法律责任", "刑事责任", "判刑", "刑法"],
        "absent_reason": "只规定了公司处分，没有法律责任的说明",
        "negatives": ["收受供应商回扣要承担什么法律责任？"],
    },
    {
        "topic": "病假远程办公",
        "kind": "sibling_slot",
        "anchor": "远程办公须经 VPN 接入；病假需医院证明",
        "docs": ["it/security-policy.md"],
        "absent_keywords": ["病假远程", "带病", "居家办公申请"],
        "absent_reason": "远程办公章节只讲安全要求，没有「病假期间能否远程」的规定",
        "negatives": ["病假期间可以在家远程办公吗？"],
    },
]


def main() -> int:
    ensure_utf8_stdio()
    config = load_config(root=ROOT, use_dotenv=False)
    kb = KnowledgeBase(config)
    kb.build()
    registry = build_registry(kb)
    docs = load_docs()

    visibility: Dict[str, List[str]] = {}
    for chunk in kb.chunks.values():
        value = chunk.metadata.get("visibility", ["public"])
        visibility.setdefault(chunk.source, value if isinstance(value, list) else [value])

    rows = build_candidates(HOLDOUT_TOPICS, registry, docs, visibility)

    ids = [row["case_id"] for row in rows]
    duplicated = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicated:
        raise SystemExit(f"case_id 重复：{duplicated[:5]}")

    neg = sum(1 for row in rows if not row["expect_answer"])
    print(f"holdout 生成完成：{len(rows)} 条（负例 {neg}、正例 {len(rows) - neg}）")
    print(f"文件：data/eval/hard_negatives.holdout.jsonl")

    (OUT_DIR / "hard_negatives.holdout.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "hard_negatives.holdout.review.md").write_text(render_review(rows), encoding="utf-8")
    print("\n⚠ reviewed=false：这份需要人工确认后才能当作正式结论（约 10 分钟）")
    print("   重点确认：正例的正确答案是否真能从文档推出来（尤其是「规则 + 具体情形」那些）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
