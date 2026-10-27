"""生成 hard negative 候选（**只产候选，不写最终标签**）。

用法::

    python scripts/build_hard_negatives.py

为什么需要这个脚本
------------------
现有的 21 条评测集里只有 **6 条**负例（应拒答），而且已经被
`calibrate_thresholds.py` 用来网格搜索阈值了 —— 在同一批数据上调参再报"改善"，
就是过拟合。要验证"换判定信号能不能降幻觉"，必须有一批**没被用过、且足够难**的题。

"难"的定义不是"问题奇怪"，而是：
**与知识库里真实存在的事实同构，但文档里恰好没有这个答案。**
这种题词面 gate 会误判成"有依据"（实测 8 条候选里 5 条被误答），
才是真正能区分"词面启发式"与"语义判定"的测试集。

四类 hard negative（见 ``TOPICS`` 里的 ``kind``）
------------------------------------------------
* ``sibling_slot``：**同胞槽位缺失** —— 文档有"绩效工资占比"，没有"期权授予比例"；
* ``synonym_only``：**概念只在同义词表里** —— ``SYNONYMS`` 有"加班/调休"，语料里没有这些条款；
* ``half_answer``：**只答了一半** —— 年终奖给了公式与发放条件，但没写几月发；
* ``wrong_premise``：**带错误前提的确认题** —— "结转上限是 10 天吧？"（实际 5 天）；
* ``permission``：**权限维度** —— 内容确实存在，但当前角色无权看，正确行为是"明确无权限"。

三个刻意的设计
--------------
1. **脚本不判断"该不该拒"，只记录证据。** 每条候选会附上：
   * ``absence``：对主题关键词做**全库 grep** 的命中数（``期权=0, 股权=0``）；
   * ``probe``：当前 gate 的真实判定、``rel`` / ``span``、以及它**认为**的证据片段。
   复核的人看这两个就能在 5 秒内判断，而不是重新通读文档。
2. **hardness 由现有 gate 算出来，不是手标的。** 只有 ``trap``（当前会误答）
   和 ``adjacent``（差一点就误答）才算 hard；``easy`` 只作对照。
3. **dev/test 按 case_id 的哈希对半分**，天然可复现，避免"分完就忘"。

产出
----
* ``data/eval/hard_negatives.candidates.jsonl`` —— 全部候选
* ``data/eval/hard_negatives.review.md`` —— 人工复核清单（按类型分组）
* ``data/eval/hard_negatives.dev.jsonl`` / ``.test.jsonl`` —— 对半切分，可直接喂给
  ``python -m docpilot eval --cases ...``

⚠ 所有记录带 ``"reviewed": false``。**确认无答案后把它改成 true**，
这才是"人工标注"的凭据；``scripts/eval_stratified.py`` 默认只统计 ``reviewed`` 的。
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docpilot.core import ensure_utf8_stdio, load_config  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402
from docpilot.tools import build_registry  # noqa: E402

DOCS_DIR = ROOT / "data" / "docs"
OUT_DIR = ROOT / "data" / "eval"

# --------------------------------------------------------------------------- #
# 主题库：每一主题锚定一条**文档里真实存在**的事实，然后问它**相邻但不存在的槽位**
#
# 写这个库时反复犯的一个错，是把"我没找到"当成"文档里没有" ——
# 所以每条都带 `absent_keywords`，脚本会重新 grep 一遍把命中数写进证据里，
# 复核的人能立刻看出"这个词其实出现过 1 次"。
# --------------------------------------------------------------------------- #
TOPICS: List[Dict[str, Any]] = [
    # ---------------- HR：休假 ----------------
    {
        "topic": "丧假",
        "kind": "sibling_slot",
        "anchor": "婚假 10 天 / 产假 158 天 / 陪产假 15 天（同一张表）",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["丧假", "直系亲属", "奔丧"],
        "absent_reason": "《休假办法》的假期表只有婚假/产假/陪产假/育儿假，没有丧假",
        "negatives": ["直系亲属去世有丧假吗？几天？", "丧假期间工资怎么算？"],
    },
    {
        "topic": "产检哺乳假",
        "kind": "sibling_slot",
        "anchor": "产假 158 天、育儿假每年 10 天",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["产检", "哺乳"],
        "absent_reason": "只有产假与育儿假，没有产检假、哺乳假的天数规定",
        "negatives": ["产检假有几天？产检算病假吗？", "哺乳期每天有几小时哺乳假？"],
    },
    {
        "topic": "护理假",
        "kind": "sibling_slot",
        "anchor": "育儿假（子女满 3 周岁前每年 10 天）",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["护理假", "独生子女", "父母住院"],
        "absent_reason": "假期表没有独生子女护理假/老人护理假",
        "negatives": ["独生子女父母住院有护理假吗？"],
    },
    {
        "topic": "婚假分次",
        "kind": "half_answer",
        "anchor": "婚假 10 天，领证后 1 年内有效",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["分次", "分段", "一次休完"],
        "absent_reason": "写了天数与有效期，但没说能否分次休",
        "negatives": ["婚假可以分两次休吗？"],
    },
    {
        "topic": "产假起始",
        "kind": "half_answer",
        "anchor": "产假 158 天，按当地政策执行",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["提前休", "预产期", "产前"],
        "absent_reason": "给了总天数，没写可以提前多少天开始休",
        "negatives": ["产假可以提前多少天开始休？"],
    },
    # ---------------- HR：加班 / 调休（同义词表里有，语料里没有）----------------
    {
        "topic": "加班费与调休",
        "kind": "synonym_only",
        "anchor": "SYNONYMS 里有「加班→延时工作/overtime」「调休→加班换休」",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["加班费", "调休", "延时工作", "overtime", "加班"],
        "absent_reason": "同义词表登记了加班/调休，但《休假办法》没有加班费与调休条款",
        "negatives": [
            "加班有加班费吗？按几倍计算？",
            "周末加班可以调休吗？调休有效期多久？",
        ],
    },
    {
        "topic": "加班餐补",
        "kind": "sibling_slot",
        "anchor": "出差期间餐补 150 元/天",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["加班餐", "夜宵", "加班"],
        "absent_reason": "餐补只写了出差场景，没有平时加班餐补",
        "negatives": ["平时加班有餐补吗？"],
    },
    {
        "topic": "加班打车",
        "kind": "sibling_slot",
        "anchor": "市内交通实报实销（差旅场景）",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["打车", "出租车", "加班"],
        "absent_reason": "交通报销只覆盖差旅，没有加班打车回家的规定",
        "negatives": ["加班到晚上十点打车回家能报销吗？"],
    },
    # ---------------- HR：薪酬（受限文档）----------------
    {
        "topic": "期权股权",
        "kind": "sibling_slot",
        "anchor": "薪酬结构 = 基本工资 + 绩效工资（20%~35%）+ 补贴",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["期权", "股权", "授予", "行权"],
        "absent_reason": "薪酬只定义三部分，完全没有股权/期权类激励",
        "negatives": ["P6 的期权授予比例是多少？", "员工的股权激励什么时候行权？"],
    },
    {
        "topic": "公积金社保",
        "kind": "sibling_slot",
        "anchor": "月度总薪酬 = 基本工资 + 绩效工资 + 补贴",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["公积金", "社保", "五险一金", "缴存"],
        "absent_reason": "薪酬结构不含社保公积金；社保仅在病假条款里作为「以当地政策为准」出现",
        "negatives": ["公积金缴存基数和比例是多少？", "社保是按什么基数缴纳的？"],
    },
    {
        "topic": "个税",
        "kind": "sibling_slot",
        "anchor": "薪酬构成与绩效系数",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["个税", "所得税", "专项附加", "税前", "税后"],
        "absent_reason": "全文没有个税与专项附加扣除的说明",
        "negatives": ["个税专项附加扣除怎么申报？"],
    },
    {
        "topic": "13薪",
        "kind": "sibling_slot",
        "anchor": "年终奖 = 月基本工资 × 个人系数 × 部门系数 × 公司系数",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["13", "十三薪", "双薪"],
        "absent_reason": "只有年终奖公式，没有 13 薪/年底双薪",
        "negatives": ["公司有 13 薪吗？什么时候发？"],
    },
    {
        "topic": "年终奖发放时间",
        "kind": "half_answer",
        "anchor": "年终奖给了公式与发放条件（在职满 3 个月）",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["发放时间", "几月", "发放日"],
        "absent_reason": "有公式、有发放条件，但没有发放月份",
        "negatives": ["年终奖一般几月发放？"],
    },
    {
        "topic": "住房补贴",
        "kind": "half_answer",
        "anchor": "补贴（作为薪酬第三项出现，但没给标准）",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["住房补贴", "房补", "租房"],
        "absent_reason": "薪酬公式里有「补贴」但没有任何补贴项目与金额",
        "negatives": ["住房补贴每月多少？", "租房有补贴吗？怎么申请？"],
    },
    # ---------------- 差旅报销 ----------------
    {
        "topic": "境外住宿标准",
        "kind": "half_answer",
        "anchor": "住宿标准按 P1-P3/P4-P6/P7+ × 一线/二线/其他城市",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["境外住宿", "海外", "美元", "汇率"],
        "absent_reason": "有境外出差审批人，但住宿表只覆盖国内三档城市",
        "negatives": ["境外出差的住宿标准是多少一晚？"],
    },
    {
        "topic": "境外餐补",
        "kind": "half_answer",
        "anchor": "餐补 150 元/天（不分地区）",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["境外", "海外", "美元"],
        "absent_reason": "餐补单一标准，未区分境内外",
        "negatives": ["境外出差餐补一天多少？"],
    },
    {
        "topic": "发票丢失",
        "kind": "sibling_slot",
        "anchor": "发票要求（抬头、税号、电子发票 PDF、1000 元以上签字）",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["丢失", "遗失", "补开", "遗失说明"],
        "absent_reason": "发票章节只讲要求，没有发票丢失/补开的处理办法",
        "negatives": ["发票丢了还能报销吗？怎么处理？"],
    },
    {
        "topic": "借款上限",
        "kind": "sibling_slot",
        "anchor": "差旅借款需注明预计冲销日期",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["借款上限", "额度", "最多"],
        "absent_reason": "借款章节只有冲销时限，没有借款金额上限",
        "negatives": ["差旅借款最多能借多少？"],
    },
    {
        "topic": "出差保险",
        "kind": "sibling_slot",
        "anchor": "差旅审批、住宿、交通、餐补、发票",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["保险", "意外险", "出行险"],
        "absent_reason": "全文没有出差期间的保险安排",
        "negatives": ["出差期间的意外险是公司买的吗？"],
    },
    {
        "topic": "报销加急",
        "kind": "half_answer",
        "anchor": "打款周期为审批通过后 7 个工作日内",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["加急", "催", "快一点"],
        "absent_reason": "给了标准打款周期，没有加急通道",
        "negatives": ["报销可以加急打款吗？最快多久？"],
    },
    {
        "topic": "快递费",
        "kind": "sibling_slot",
        "anchor": "市内交通实报实销；不予报销清单",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["快递", "邮寄", "顺丰"],
        "absent_reason": "交通与不予报销清单都没提快递/邮寄费",
        "negatives": ["出差寄快递的费用能报销吗？"],
    },
    {
        "topic": "差旅改期审批",
        "kind": "half_answer",
        "anchor": "出差前必须在 OA 提交出差申请",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["改期", "变更", "行程调整"],
        "absent_reason": "只规定出差前审批，没写出差行程变更是否需要重新审批",
        "negatives": ["出差行程临时变更需要重新审批吗？"],
    },
    # ---------------- IT ----------------
    {
        "topic": "VPN 客户端系统",
        "kind": "half_answer",
        "anchor": "VPN 连不上排查五步（含 Windows/macOS 命令）",
        "docs": ["it/it-faq.md"],
        "absent_keywords": ["Linux", "Ubuntu", "Mac", "Windows"],
        "absent_reason": "排查步骤提到 Windows/macOS，但没说支持哪些客户端系统",
        "negatives": ["VPN 支持 Linux 系统吗？"],
    },
    {
        "topic": "密码管理器",
        "kind": "sibling_slot",
        "anchor": "密码复杂度、每 90 天更换、新密码不得与前 5 次相同",
        "docs": ["it/security-policy.md"],
        "absent_keywords": ["密码管理器", "Password", "1Password", "保存密码"],
        "absent_reason": "密码章节没有任何关于保管方式的允许/禁止规定",
        "negatives": ["可以用 1Password 这类工具保存公司密码吗？"],
    },
    {
        "topic": "硬件令牌申请",
        "kind": "half_answer",
        "anchor": "MFA 方式：企业认证 App 优先，硬件令牌其次，短信仅备用",
        "docs": ["it/security-policy.md"],
        "absent_keywords": ["硬件令牌", "令牌申请", "申请流程"],
        "absent_reason": "提到硬件令牌是次选，但没写怎么申请、找谁领",
        "negatives": ["MFA 硬件令牌去哪里申请？"],
    },
    {
        "topic": "工单响应时限",
        "kind": "sibling_slot",
        "anchor": "工单提交规范（现象/影响/时间/已尝试/紧急程度）",
        "docs": ["it/it-faq.md"],
        "absent_keywords": ["响应时间", "SLA", "多久响应"],
        "absent_reason": "工单章节只规定怎么写，没有响应时限承诺",
        "negatives": ["提交 IT 工单后多久必须响应？"],
    },
    {
        "topic": "软件申请申诉",
        "kind": "sibling_slot",
        "anchor": "软件安装申请审批人与时长（含「一律不批」）",
        "docs": ["it/it-faq.md"],
        "absent_keywords": ["申诉", "复议", "被拒"],
        "absent_reason": "审批表里没有「被拒绝后如何申诉」",
        "negatives": ["软件安装申请被拒绝可以申诉吗？"],
    },
    # ---------------- 入职 / 试用期 ----------------
    {
        "topic": "入职体检",
        "kind": "sibling_slot",
        "anchor": "必带材料：身份证、学历学位证、离职证明、银行卡、照片",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["体检", "健康证"],
        "absent_reason": "必带材料清单里没有体检报告",
        "negatives": ["入职需要提交体检报告吗？"],
    },
    {
        "topic": "延长试用期薪资",
        "kind": "half_answer",
        "anchor": "试用期薪资为转正后的 90%；可延长一次（≤50%）",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["延长", "薪资"],
        "absent_reason": "写了试用期薪资与延长规则，但没说延长期间薪资怎么算",
        "negatives": ["试用期延长后薪资还是 90% 吗？"],
    },
    {
        "topic": "缩短试用期",
        "kind": "half_answer",
        "anchor": "试用期 3 个月（P7 及以上 6 个月）；可延长一次",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["缩短", "提前转正"],
        "absent_reason": "只规定延长，没说能否缩短/提前转正",
        "negatives": ["P7 的 6 个月试用期可以提前转正吗？"],
    },
    {
        "topic": "实习生转正",
        "kind": "sibling_slot",
        "anchor": "试用期与转正评估流程",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["实习生", "实习期"],
        "absent_reason": "入职指引只写正式员工试用期，实习生转正未提（休假办法里也只说按协议执行）",
        "negatives": ["实习生实习期满可以直接转正吗？"],
    },
    # ---------------- 工程 ----------------
    {
        "topic": "覆盖率阈值",
        "kind": "half_answer",
        "anchor": "新增/修改逻辑要有单测，覆盖率不得低于改动前的水平",
        "docs": ["engineering/code-review.md"],
        "absent_keywords": ["覆盖率", "百分比", "80%", "90%"],
        "absent_reason": "只要求「不低于改动前」，没有绝对阈值",
        "negatives": ["代码覆盖率要求达到多少百分比？"],
    },
    {
        "topic": "PR 评审轮次",
        "kind": "sibling_slot",
        "anchor": "评审人 1 个工作日内响应；单个 PR 建议不超过 400 行",
        "docs": ["engineering/code-review.md"],
        "absent_keywords": ["轮次", "几轮", "最多改"],
        "absent_reason": "规定了响应时效与体量，没有评审轮次上限",
        "negatives": ["一个 PR 最多可以评审几轮？"],
    },
    {
        "topic": "评审激励",
        "kind": "sibling_slot",
        "anchor": "评审关注点与反模式",
        "docs": ["engineering/code-review.md"],
        "absent_keywords": ["积分", "奖励", "考核评审"],
        "absent_reason": "全文没有评审激励/积分机制",
        "negatives": ["代码评审有积分或奖励吗？"],
    },
    {
        "topic": "单测耗时",
        "kind": "sibling_slot",
        "anchor": "CI 必须全绿：单测、lint、构建",
        "docs": ["engineering/code-review.md"],
        "absent_keywords": ["耗时", "分钟", "跑完"],
        "absent_reason": "只要求 CI 全绿，没有单测耗时要求",
        "negatives": ["单元测试必须在几分钟内跑完？"],
    },
    # ---------------- 合规（受限文档）----------------
    {
        "topic": "宴请金额",
        "kind": "half_answer",
        "anchor": "礼品界限：200 元以内可收 / 200-1000 申报 / 超 1000 不得收",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": ["宴请", "人均", "招待"],
        "absent_reason": "礼品有金额界限，宴请只说「接受宴请必须申报」，没有金额标准",
        "negatives": ["接受供应商宴请的人均标准是多少元？"],
    },
    {
        "topic": "竞业补偿金额",
        "kind": "half_answer",
        "anchor": "竞业限制：按月支付补偿金，期限通常 1 年",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": ["补偿金", "多少", "百分比"],
        "absent_reason": "说了按月支付补偿金，但没有金额或比例",
        "negatives": ["竞业限制的补偿金每月多少钱？"],
    },
    {
        "topic": "利益冲突审批时长",
        "kind": "sibling_slot",
        "anchor": "利益冲突申报流程：提交 HR 与法务，合规委员会评估",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": ["审批时长", "工作日", "多久"],
        "absent_reason": "有申报流程，没有评估时限",
        "negatives": ["利益冲突申报多久能审批完？"],
    },
    # ---------------- 带错误前提的确认题 ----------------
    {
        "topic": "结转上限错误前提",
        "kind": "wrong_premise",
        "anchor": "年假最多结转 5 天（文档明确）",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档是 5 天；问题故意说 10 天，正确行为是纠正而不是附和",
        "expect_answer": True,  # 有依据可依（5 天），但前提被写错
        "expected_keywords": ["5 天"],
        "note_extra": "关键词只能查「有没有提 5 天」，查不出「是不是纠正了 10 天」；"
                      "Mock 会因摘抄原文而误通过，真实验收要读句子或用判定器",
        "negatives": [],
        "positives": ["年假结转上限是 10 天吧？"],
    },
    {
        "topic": "密码位数错误前提",
        "kind": "wrong_premise",
        "anchor": "密码不少于 12 位（文档明确）",
        "docs": ["it/security-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档是 12 位；问题说 8 位，正确行为是纠正",
        "expect_answer": True,
        "expected_keywords": ["12 位"],
        "note_extra": "关键词只能查「有没有提 12 位」，查不出「是不是附和了 8 位」",
        "negatives": [],
        "positives": ["公司密码要求 8 位以上就行吧？"],
    },
    {
        "topic": "报销时效错误前提",
        "kind": "wrong_premise",
        "anchor": "出差结束后 15 个工作日内提交（文档明确）",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档是 15 个工作日；问题说一个月，正确行为是纠正",
        "expect_answer": True,
        "expected_keywords": ["15 个工作日"],
        "note_extra": "关键词只能查「有没有提 15 个工作日」，查不出「是不是默认了一个月」",
        "negatives": [],
        "positives": ["报销单是一个月内提交就行吧？"],
    },
    # ---------------- 权限维度：内容存在，但角色看不到 ----------------
    {
        "topic": "薪酬受限-P4年终奖",
        "kind": "permission",
        "anchor": "年终奖个人系数表（S=2.0 … D=0）",
        "docs": ["hr/compensation.md"],
        "absent_keywords": [],
        "absent_reason": "内容存在，但 public 角色无权看，正确行为是明确无权限",
        "force_role": "public",
        "negatives": ["P4 的年终奖个人系数是多少？"],
    },
    {
        "topic": "薪酬受限-S级调薪",
        "kind": "permission",
        "anchor": "调薪幅度：S 级 10%~15%",
        "docs": ["hr/compensation.md"],
        "absent_keywords": [],
        "absent_reason": "内容存在，但 public 角色无权看",
        "force_role": "public",
        "negatives": ["S 级绩效的年度调薪幅度是多少？"],
    },
    {
        "topic": "行为规范受限-购物卡",
        "kind": "permission",
        "anchor": "现金/购物卡/有价证券一律不得收受",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": [],
        "absent_reason": "内容存在，但 public 角色无权看",
        "force_role": "public",
        "negatives": ["收受供应商 300 元购物卡该怎么处理？"],
    },
    # ---------------- 相邻但**可回答**的正例（防"变成什么都不敢答"）----------------
    {
        "topic": "正例-年假折现",
        "kind": "positive_control",
        "anchor": "超出 5 天的部分自动作废，公司不另行折现",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：不折现",
        "expect_answer": True,
        "negatives": [],
        "positives": ["年假没休完可以折现吗？"],
    },
    {
        "topic": "正例-代码外发",
        "kind": "positive_control",
        "anchor": "禁止将机密及以上数据上传至公有云 AI 服务",
        "docs": ["it/security-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：不可以",
        "expect_answer": True,
        "negatives": [],
        "positives": ["能把公司代码贴到公开的 AI 工具里问问题吗？"],
    },
    {
        "topic": "正例-个人电脑",
        "kind": "positive_control",
        "anchor": "确需使用个人设备时须安装 MDM 并通过合规检查",
        "docs": ["it/security-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：原则上用公司设备，特殊情况需 MDM",
        "expect_answer": True,
        "negatives": [],
        "positives": ["可以用自己的电脑办公吗？"],
    },
    {
        "topic": "正例-服务台周末",
        "kind": "positive_control",
        "anchor": "服务台电话工作日 9:00-18:00；紧急故障 7×24",
        "docs": ["it/it-faq.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：普通工单工作时间，紧急故障 7×24",
        "expect_answer": True,
        "negatives": [],
        "positives": ["服务台周末上班吗？"],
    },
    {
        "topic": "正例-公务舱",
        "kind": "positive_control",
        "anchor": "飞机经济舱；P7 及以上可乘坐公务舱",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：P6 不可坐公务舱",
        "expect_answer": True,
        "negatives": [],
        "positives": ["P6 出差飞行 6 小时可以坐公务舱吗？"],
    },
    {
        "topic": "正例-自己批PR",
        "kind": "positive_control",
        "anchor": "作者不得自己 Approve 自己的 PR",
        "docs": ["engineering/code-review.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：不可以",
        "expect_answer": True,
        "negatives": [],
        "positives": ["可以自己 approve 自己的 PR 吗？"],
    },
    {
        "topic": "正例-骚扰调查时长",
        "kind": "positive_control",
        "anchor": "5 个工作日内启动调查，30 个工作日内给出结论",
        "docs": ["legal/conduct-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：30 个工作日内",
        "expect_answer": True,
        "negatives": [],
        "positives": ["职场骚扰举报后多久出结论？"],
    },
    # ---------------- 第二批：补足样本量（负例 ≥60、正例 ≥18）----------------
    {
        "topic": "丧假证明",
        "kind": "sibling_slot",
        "anchor": "婚假需提供结婚证；产假需配偶生育证明",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["丧假", "死亡证明", "亲属关系证明"],
        "absent_reason": "丧假本身不存在，更没有证明材料要求",
        "negatives": ["丧假需要提供什么证明材料？"],
    },
    {
        "topic": "护理假工资",
        "kind": "sibling_slot",
        "anchor": "病假工资按最低工资 80% 计发（唯一给了工资口径的假）",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["护理假", "护理假工资"],
        "absent_reason": "护理假不存在；各类假期的工资口径也只覆盖病假",
        "negatives": ["护理假期间工资怎么算？"],
    },
    {
        "topic": "产假工资",
        "kind": "half_answer",
        "anchor": "产假 158 天；病假工资按最低工资 80%",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": ["生育津贴", "产假工资", "待遇"],
        "absent_reason": "给了产假天数，但没写产假期间工资/生育津贴怎么发",
        "negatives": ["产假期间的工资怎么发？"],
    },
    {
        "topic": "年终奖个税",
        "kind": "sibling_slot",
        "anchor": "年终奖 = 月基本工资 × 个人系数 × 部门系数 × 公司系数",
        "docs": ["hr/compensation.md"],
        "absent_keywords": ["个税", "所得税", "税"],
        "absent_reason": "年终奖只给计算公式，没有任何计税口径",
        "negatives": ["年终奖要交多少个税？"],
    },
    {
        "topic": "境外住宿上限",
        "kind": "half_answer",
        "anchor": "境外出差需事业部总经理审批；住宿表只有国内三档",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["美元", "境外", "汇率"],
        "absent_reason": "境外出差只有审批要求，没有住宿上限（本币或外币都没有）",
        "negatives": ["境外出差住宿的报销上限是多少？"],
    },
    {
        "topic": "电子发票丢失",
        "kind": "half_answer",
        "anchor": "电子发票需下载 PDF 原件，截图不予受理",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["重新下载", "发票丢失", "补开"],
        "absent_reason": "要求提供 PDF 原件，但没写原件丢失/未下载怎么补救",
        "negatives": ["电子发票原件丢了可以重新下载吗？"],
    },
    {
        "topic": "借款审批人",
        "kind": "half_answer",
        "anchor": "差旅借款需在借款单注明预计冲销日期，15 个工作日内冲销",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["借款审批", "谁审批"],
        "absent_reason": "借款章节只写冲销规则，没写借款由谁审批、上限多少",
        "negatives": ["差旅借款需要谁审批？"],
    },
    {
        "topic": "出差生病",
        "kind": "half_answer",
        "anchor": "不予报销情形清单（个人消费/改签费/宴请/罚款/缺发票）",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": ["医疗", "生病", "住院", "医药费"],
        "absent_reason": "不予报销清单没有提医疗费，全文也没有出差期间就医的规定",
        "negatives": ["出差期间生病住院的费用能报销吗？"],
    },
    {
        "topic": "VPN客户端下载",
        "kind": "sibling_slot",
        "anchor": "打印机驱动下载地址：内网 http://intranet.it/printers",
        "docs": ["it/it-faq.md"],
        "absent_keywords": ["VPN 客户端", "下载地址", "安装包"],
        "absent_reason": "有打印机驱动地址，但没有 VPN 客户端的下载方式",
        "negatives": ["VPN 客户端在哪里下载？"],
    },
    {
        "topic": "工单超时",
        "kind": "sibling_slot",
        "anchor": "工单系统可跟踪进度；服务台电话工作日 9:00-18:00",
        "docs": ["it/it-faq.md"],
        "absent_keywords": ["超时", "投诉", "催单"],
        "absent_reason": "没有工单超时后的升级/投诉路径",
        "negatives": ["工单提交后一直没人处理该怎么办？"],
    },
    {
        "topic": "体检费用",
        "kind": "sibling_slot",
        "anchor": "入职前准备：HR 发送入职通知与材料清单",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["体检", "健康检查"],
        "absent_reason": "入职流程没有体检环节，也没有体检费用报销",
        "negatives": ["入职体检的费用公司报销吗？"],
    },
    {
        "topic": "实习生试用期",
        "kind": "sibling_slot",
        "anchor": "试用期 3 个月（P7 及以上 6 个月），适用于正式员工",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": ["实习生", "实习期"],
        "absent_reason": "试用期规则只针对正式员工，实习生未提及",
        "negatives": ["实习生有试用期吗？多久？"],
    },
    {
        "topic": "权限审批时长",
        "kind": "half_answer",
        "anchor": "权限申请审批人与有效期（一般业务系统/生产只读/读写/敏感字段）",
        "docs": ["it/security-policy.md"],
        "absent_keywords": ["审批时长", "多久批", "几个工作日"],
        "absent_reason": "权限表给了审批人与有效期，但没有审批时长",
        "negatives": ["生产环境只读权限申请要多久审批完？"],
    },
    # ---------------- 第二批正例：防"新 gate 变成什么都不敢答"----------------
    {
        "topic": "正例-短信备用",
        "kind": "positive_control",
        "anchor": "短信验证码仅作为备用方式，不作为首选",
        "docs": ["it/security-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：可以，但仅作备用",
        "expect_answer": True,
        "negatives": [],
        "positives": ["MFA 可以用短信验证码吗？"],
    },
    {
        "topic": "正例-高铁一等座",
        "kind": "positive_control",
        "anchor": "高铁/动车：P1-P6 二等座；P7 及以上一等座",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：P1 只能二等座",
        "expect_answer": True,
        "negatives": [],
        "positives": ["P1 出差可以坐高铁一等座吗？"],
    },
    {
        "topic": "正例-试用期年假",
        "kind": "positive_control",
        "anchor": "试用期员工同样享有年假，但需转正后方可申请使用",
        "docs": ["hr/leave-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：享有，但转正后才能用",
        "expect_answer": True,
        "negatives": [],
        "positives": ["试用期员工有年假吗？"],
    },
    {
        "topic": "正例-报销审批链",
        "kind": "positive_control",
        "anchor": "审批链：直属主管 → 部门负责人 → 财务复核 → 打款",
        "docs": ["finance/expense-policy.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给出了完整审批链",
        "expect_answer": True,
        "negatives": [],
        "positives": ["报销单需要经过哪些人审批？"],
    },
    {
        "topic": "正例-打印机驱动",
        "kind": "positive_control",
        "anchor": "驱动下载地址：内网 http://intranet.it/printers",
        "docs": ["it/it-faq.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确给出了下载地址",
        "expect_answer": True,
        "negatives": [],
        "positives": ["打印机驱动在哪里下载？"],
    },
    {
        "topic": "正例-忘记密码",
        "kind": "positive_control",
        "anchor": "在 SSO 登录页点击「忘记密码」，通过企业邮箱重置",
        "docs": ["it/it-faq.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：SSO 页自助重置",
        "expect_answer": True,
        "negatives": [],
        "positives": ["忘记密码了怎么重置？"],
    },
    {
        "topic": "正例-MFA重置",
        "kind": "positive_control",
        "anchor": "MFA 设备丢失需本人携带工牌到 IT 服务台重置，不接受线上申请",
        "docs": ["it/it-faq.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：不可以线上，需本人到服务台",
        "expect_answer": True,
        "negatives": [],
        "positives": ["MFA 设备丢了可以线上申请重置吗？"],
    },
    {
        "topic": "正例-转正评估",
        "kind": "positive_control",
        "anchor": "转正评估在试用期结束前 10 个工作日发起",
        "docs": ["hr/onboarding.md"],
        "absent_keywords": [],
        "absent_reason": "文档明确回答：试用期结束前 10 个工作日",
        "expect_answer": True,
        "negatives": [],
        "positives": ["转正评估什么时候发起？"],
    },
]

# 难度分层阈值：与 agent.min_evidence_span 保持一致的"差一点"口径
ADJACENT_RELEVANCE = 0.20


# --------------------------------------------------------------------------- #
# 语料与证据
# --------------------------------------------------------------------------- #


def load_docs() -> Dict[str, str]:
    """读取全部文档正文（去掉 front matter），用于 absence 取证。"""
    docs: Dict[str, str] = {}
    for path in sorted(DOCS_DIR.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        raw = re.sub(r"^---\s*\n.*?\n---\s*\n", "", raw, flags=re.S)
        docs[path.relative_to(DOCS_DIR).as_posix()] = raw
    return docs


def absence_evidence(keywords: Sequence[str], docs: Dict[str, str]) -> Dict[str, Any]:
    """对主题关键词做全库 grep，返回可审计的证据。

    **这不是判定依据，只是证据。** "关键词命中 0 次"不等于"文档没答这个问题"
    （可能用别的词说了），所以复核的人仍要读 ``probe.quote``。
    """
    hits: Dict[str, List[str]] = {}
    for keyword in keywords:
        where = [source for source, text in docs.items() if keyword and keyword in text]
        hits[keyword] = where
    return hits


def summary_of_hits(hits: Dict[str, List[str]]) -> str:
    parts = []
    for keyword, where in hits.items():
        parts.append(f"{keyword}={len(where)}")
    return "、".join(parts) if parts else "（无需 grep 的关键词）"


# --------------------------------------------------------------------------- #
# 语料指纹与复核状态覆盖层
# --------------------------------------------------------------------------- #
#
# 为什么复核状态要单独存一个文件
# ------------------------------
# 生成器每次都会**无条件重写** candidates / dev / test 三个文件。如果人直接去改
# JSONL 里的 ``reviewed``，那么下一次重建（哪怕只是为了补 fingerprint）就会把
# 一小时的复核结果**静默清空**。
# 所以复核结论放在独立的 ``hard_negatives.review_state.jsonl`` 里，
# 生成器只**读**它，从来不写掉它的人为决定。
#
# 为什么每行带一个 ``question``
# ------------------------------
# 复核时要反复对照"这一条问的是什么"。把问题抄进状态文件，就能在一个文件里
# 完成全部复核，不用在 review.md 和 jsonl 之间来回跳。


def corpus_fingerprint(docs: Dict[str, str], chunks: int) -> Dict[str, Any]:
    """语料指纹：文件集合 + 正文 + 片段数。

    存进状态文件的 ``_meta``，用来回答一个很实际的问题：
    **"这次复核是针对哪一版语料做的？"** 语料一改，指纹就变，生成器会警告
    "复核结论可能已过期"（真正的差异清单由后续的 ``--recheck`` 给出）。
    """
    blob = json.dumps({"docs": docs, "chunks": chunks}, ensure_ascii=False, sort_keys=True)
    return {
        "fingerprint": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12],
        "documents": len(docs),
        "chunks": chunks,
    }


def load_review_state(path: Path) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """读复核状态（JSONL，首行 ``_meta``）。文件不存在时返回空状态。"""
    meta: Dict[str, Any] = {}
    decisions: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return meta, decisions
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # 状态文件被手改坏一行，不该让整个流程挂掉
        if "_meta" in data:
            meta = data["_meta"] or {}
            continue
        case_id = data.get("case_id")
        if case_id:
            decisions[case_id] = {
                "reviewed": bool(data.get("reviewed", False)),
                "verdict": str(data.get("verdict", "keep")),
                "reason": str(data.get("reason", "")),
            }
    return meta, decisions


def write_review_state(
    path: Path,
    meta: Dict[str, Any],
    decisions: Dict[str, Dict[str, Any]],
    questions: Dict[str, str],
) -> None:
    """写回复核状态：已有决定**原样保留**，新用例追加为 ``reviewed=false``。"""
    lines = [json.dumps({"_meta": meta}, ensure_ascii=False)]
    for case_id in sorted(decisions):
        entry = decisions[case_id]
        lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "question": questions.get(case_id, ""),
                    "reviewed": bool(entry.get("reviewed", False)),
                    "verdict": entry.get("verdict", "keep"),
                    "reason": entry.get("reason", ""),
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_review(
    items: Sequence[Dict[str, Any]], decisions: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """把复核状态套到候选上。``verdict=drop`` 的条目保留在 candidates 里（可审计），
    但不进入 dev/test（免得评测继续跑已判定标错的题）。"""
    for item in items:
        state = decisions.get(item["case_id"], {})
        item["reviewed"] = bool(state.get("reviewed", False))
        item["verdict"] = str(state.get("verdict", "keep"))
        item["review_reason"] = str(state.get("reason", ""))
    return list(items)


# --------------------------------------------------------------------------- #
# 用现有 gate 计算 hardness
# --------------------------------------------------------------------------- #


def probe_gate(registry, question: str, role: str) -> Dict[str, Any]:
    """跑一次真实检索 + 证据判定，记录"系统认为的证据"。"""
    payload = registry.execute("search_docs", {"query": question, "top_k": 4}, role=role).payload or {}
    results = payload.get("results", []) or []
    return {
        "decision": "作答" if payload.get("has_evidence") else "拒答",
        "has_evidence": bool(payload.get("has_evidence")),
        "relevance": float(payload.get("best_relevance", 0.0)),
        "span": int(payload.get("best_span_chars", 0)),
        "restricted": list(payload.get("restricted_sources", []) or []),
        "top": [
            {
                "citation": item.get("citation", ""),
                "relevance": item.get("relevance"),
                "quote": re.sub(r"\s+", " ", str(item.get("text", "")))[:90],
            }
            for item in results[:3]
        ],
    }


def classify(kind: str, expect_answer: bool, probe: Dict[str, Any]) -> str:
    """难度分层：由**当前 gate 的真实行为**决定，不是手标。

    * 负例：``trap``（会误答）> ``adjacent``（差一点）> ``easy``（明显该拒）
    * 正例：``answerable``（正常）> ``would_over_refuse``（**当前会被误拒**，重点看）
    * 权限类：``handled_permission``（已正确识别为无权限）> ``unhandled_permission``
    """
    if expect_answer:
        if probe["has_evidence"]:
            return "answerable"
        return "blocked_by_permission" if probe["restricted"] else "would_over_refuse"
    if kind == "permission":
        return "handled_permission" if probe["restricted"] else "unhandled_permission"
    if probe["has_evidence"]:
        return "trap"
    if probe["relevance"] >= ADJACENT_RELEVANCE or probe["span"] >= 2:
        return "adjacent"
    return "easy"


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #


def slug(text: str) -> str:
    """生成稳定、可读、**唯一**的短 id。

    这里踩过一个坑：最初写成"只保留 [a-z0-9]"，而主题名几乎全是中文，
    于是**所有主题都被清洗成空串 → case_id 全部撞成 ``hard-case-n1``**。
    症状很隐蔽：文件看着有 57 条，实际只有少数几个不同的 id，
    dev/test 切分也变成 47/10（哈希只落在几个值上）。
    修法：ASCII 部分为空时，用主题名的哈希前 6 位兜底，并在写入前断言唯一。
    """
    ascii_part = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if ascii_part:
        return ascii_part
    return "t" + hashlib.md5(text.encode("utf-8")).hexdigest()[:6]


# dev/test 文件只保留评测必需字段：证据片段（probe.top 的 quote）留在
# candidates.jsonl 与 review.md 里，避免三份文件重复存同样的文本（125KB → 几 KB）。
COMPACT_FIELDS = (
    "case_id",
    "question",
    "role",
    "expect_answer",
    "expected_sources",
    "expected_keywords",
    "kind",
    "topic",
    "hardness",
    "note",
    "reviewed",
    "verdict",
    "review_reason",
)


def compact(item: Dict[str, Any]) -> Dict[str, Any]:
    """给评测用的精简记录（``EvalCase.from_dict`` 只认其中一部分字段）。"""
    return {key: item[key] for key in COMPACT_FIELDS if key in item}


def note_of(topic: Dict[str, Any]) -> str:
    """统一生成 note，附带主题级的补充说明（``note_extra``）。"""
    note = f"[{topic['kind']}] 锚定：{topic['anchor']}"
    extra = topic.get("note_extra")
    return f"{note}；{extra}" if extra else note


def split_of(case_id: str) -> str:
    """稳定对半切分：同一 case_id 永远落在同一侧（换机器也一样）。

    用哈希的**高半区**做阈值，而不是 ``% 2``：后者只让哈希最后一位参与运算，
    对"前缀相同、只在结尾变化"的 id 容易失衡；阈值判断把 32 位都用上了。
    """
    digest = hashlib.md5(case_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) < 0x80000000 else "test"


def build_candidates(
    topics: Sequence[Dict[str, Any]],
    registry,
    docs: Dict[str, str],
    visibility: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """把主题表展开成候选记录。

    抽成函数是为了**复用**：holdout 那份数据用同一套规则生成
    （`scripts/build_holdout.py`），避免两处逻辑漂移 ——
    一旦漂移，两份数据的 hardness / 证据口径就不可比了。
    """
    candidates: List[Dict[str, Any]] = []
    for topic in topics:
        keywords = topic.get("absent_keywords", [])
        hits = absence_evidence(keywords, docs)
        # 主题 key = 可读 slug + 主题名哈希前 4 位。
        # 只加哈希是因为撞过一次：`VPN 客户端系统` 与 `VPN客户端下载` 都会被
        # 清洗成 "vpn"，case_id 直接重复（唯一性断言当场抓住了）。
        # 带哈希后既稳定（不随主题顺序变化）又唯一。
        topic_key = f"{slug(topic['topic'])}-{hashlib.md5(topic['topic'].encode('utf-8')).hexdigest()[:4]}"
        default_role = "public"
        for source in topic["docs"]:
            visible = [str(v).lower() for v in visibility.get(source, ["public"])]
            if "public" not in visible:
                default_role = visible[0]
                break
        role = topic.get("force_role") or default_role

        for index, question in enumerate(topic.get("negatives", []), 1):
            probe = probe_gate(registry, question, role)
            case_id = f"hard-{topic_key}-n{index}"
            candidates.append(
                {
                    "case_id": case_id,
                    "question": question,
                    "role": role,
                    "expect_answer": False,
                    "expected_sources": [],
                    "expected_keywords": [],
                    "kind": topic["kind"],
                    "topic": topic["topic"],
                    "anchor": topic["anchor"],
                    # rationale = 中性字段：负例写"为什么没有答案"，正例写"依据在哪"。
                    # 早先两处都塞在 absent_reason 里，导致正例在复核清单上
                    # 也印着"为什么没有答案："——标签和内容自相矛盾（展示 bug）。
                    "rationale": topic.get("rationale") or topic["absent_reason"],
                    "absent_reason": topic["absent_reason"],
                    "absence_check": summary_of_hits(hits),
                    "absence_detail": hits,
                    "hardness": classify(topic["kind"], False, probe),
                    "probe": probe,
                    "note": note_of(topic),
                    "reviewed": False,
                }
            )
        for index, question in enumerate(topic.get("positives", []), 1):
            probe = probe_gate(registry, question, role)
            case_id = f"hard-{topic_key}-p{index}"
            candidates.append(
                {
                    "case_id": case_id,
                    "question": question,
                    "role": role,
                    "expect_answer": True,
                    "expected_sources": topic["docs"],
                    "expected_keywords": list(topic.get("expected_keywords", [])),
                    "kind": topic["kind"],
                    "topic": topic["topic"],
                    "anchor": topic["anchor"],
                    "rationale": topic.get("rationale") or topic["absent_reason"],
                    "absent_reason": topic["absent_reason"],
                    "absence_check": summary_of_hits(hits),
                    "absence_detail": hits,
                    "hardness": classify(topic["kind"], True, probe),
                    "probe": probe,
                    "note": note_of(topic),
                    "reviewed": False,
                }
            )
    return candidates


def main() -> int:
    ensure_utf8_stdio()
    config = load_config(root=ROOT, use_dotenv=False)  # 不读 .env：本脚本不碰 LLM
    kb = KnowledgeBase(config)
    kb.build()
    registry = build_registry(kb)
    docs = load_docs()

    visibility: Dict[str, List[str]] = {}
    for chunk in kb.chunks.values():
        value = chunk.metadata.get("visibility", ["public"])
        visibility.setdefault(chunk.source, value if isinstance(value, list) else [value])

    candidates = build_candidates(TOPICS, registry, docs, visibility)

    # 去重（同一问题可能被两条主题命中）
    seen = set()
    unique: List[Dict[str, Any]] = []
    for item in candidates:
        if item["question"] in seen:
            continue
        seen.add(item["question"])
        unique.append(item)

    # 断言 case_id 唯一 —— 中文主题名撞车那次，就是靠这条才该被当场发现的
    ids = [item["case_id"] for item in unique]
    duplicated = sorted({cid for cid in ids if ids.count(cid) > 1})
    if duplicated:
        raise SystemExit(f"case_id 重复：{duplicated[:5]}（共 {len(duplicated)} 个）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 复核状态：读旧状态 → 套用 → 合并新用例 → 写回（不覆盖人为决定）----
    fingerprint = corpus_fingerprint(docs, len(kb.chunks))
    state_path = OUT_DIR / "hard_negatives.review_state.jsonl"
    old_meta, decisions = load_review_state(state_path)
    if old_meta and old_meta.get("fingerprint") != fingerprint["fingerprint"]:
        print(
            "⚠ 语料已在复核之后发生变化：\n"
            f"   复核时 {old_meta.get('fingerprint')}（{old_meta.get('documents')} 篇 / "
            f"{old_meta.get('chunks')} 片段）\n"
            f"   现在   {fingerprint['fingerprint']}（{fingerprint['documents']} 篇 / "
            f"{fingerprint['chunks']} 片段）\n"
            "   → 已有 reviewed 标记可能已过期，需要重新确认受影响的条目。\n"
        )
    apply_review(unique, decisions)
    for item in unique:
        decisions.setdefault(item["case_id"], {"reviewed": False, "verdict": "keep", "reason": ""})
    write_review_state(
        state_path,
        {
            **fingerprint,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cases": len(unique),
        },
        decisions,
        {item["case_id"]: item["question"] for item in unique},
    )

    dropped = [item for item in unique if item["verdict"] == "drop"]
    kept = [item for item in unique if item["verdict"] != "drop"]

    # candidates 保留**全部**（含 drop），便于审计；dev/test 只放 keep
    (OUT_DIR / "hard_negatives.candidates.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in unique) + "\n", encoding="utf-8"
    )

    for side in ("dev", "test"):
        subset = [item for item in kept if split_of(item["case_id"]) == side]
        (OUT_DIR / f"hard_negatives.{side}.jsonl").write_text(
            "\n".join(json.dumps(compact(item), ensure_ascii=False) for item in subset) + "\n", encoding="utf-8"
        )

    (OUT_DIR / "hard_negatives.review.md").write_text(render_review(unique), encoding="utf-8")

    report(unique)
    return 0


def report(items: Sequence[Dict[str, Any]]) -> None:
    negatives = [item for item in items if not item["expect_answer"]]
    positives = [item for item in items if item["expect_answer"]]

    def count(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in rows:
            out[row[key]] = out.get(row[key], 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    print("hard negative 候选生成完成")
    print("=" * 78)
    print(f"共 {len(items)} 条：负例 {len(negatives)}、正例 {len(positives)}")
    reviewed = [item for item in items if item.get("reviewed")]
    dropped = [item for item in items if item.get("verdict") == "drop"]
    print(
        f"复核状态：已复核 {len(reviewed)}/{len(items)}"
        + (f"，其中判为「文档其实有答案」{len(dropped)} 条（不进入 dev/test）" if dropped else "")
    )
    print(f"切分：dev {sum(1 for i in items if split_of(i['case_id']) == 'dev')} / "
          f"test {sum(1 for i in items if split_of(i['case_id']) == 'test')}")
    print("\n按类型：")
    for kind, n in count(items, "kind").items():
        print(f"  {kind:<18}{n:>4}")
    print("\n按当前 gate 的难度：")
    for hardness, n in count(items, "hardness").items():
        print(f"  {hardness:<24}{n:>4}")

    traps = [i for i in negatives if i["hardness"] == "trap"]
    adjacent = [i for i in negatives if i["hardness"] == "adjacent"]
    easy = [i for i in negatives if i["hardness"] in ("easy",)]
    print("\n负例难度分布（这是「这批题到底难不难」的直接证据）：")
    if negatives:
        print(f"  trap（当前会误答）    {len(traps):>3} 条 = {len(traps)/len(negatives):.0%}")
        print(f"  adjacent（差一点）    {len(adjacent):>3} 条 = {len(adjacent)/len(negatives):.0%}")
        print(f"  easy（明显该拒）      {len(easy):>3} 条 = {len(easy)/len(negatives):.0%}")
    over = [i for i in positives if i["hardness"] == "would_over_refuse"]
    blocked = [i for i in positives if i["hardness"] == "blocked_by_permission"]
    print("\n正例上的现状（用来看「新 gate 会不会变成什么都不敢答」）：")
    print(f"  answerable {sum(1 for i in positives if i['hardness'] == 'answerable')} / "
          f"would_over_refuse {len(over)} / blocked_by_permission {len(blocked)}")
    for item in over:
        print(f"    ⚠ 当前会被误拒：{item['question']}（rel={item['probe']['relevance']:.2f}）")
    for item in blocked:
        print(f"    ⚠ 角色可能标错：{item['question']}（restricted={item['probe']['restricted']}）")

    print("\n产出：")
    for name in ("hard_negatives.candidates.jsonl", "hard_negatives.review.md",
                 "hard_negatives.review_state.jsonl",
                 "hard_negatives.dev.jsonl", "hard_negatives.test.jsonl"):
        print(f"  data/eval/{name}")
    print("\n复核方式：编辑 hard_negatives.review_state.jsonl ——")
    print("  确认「文档确实没有答案」→ 该行 reviewed 改成 true")
    print("  发现「文档其实答了」    → verdict 改成 \"drop\"，并写一句 reason")
    print("  复核结论存在独立文件里，重跑本脚本不会清空它。")


def render_review(items: Sequence[Dict[str, Any]]) -> str:
    """人工复核清单。每条把"系统认为的证据"摊开，确认成本从分钟级降到秒级。

    复核结论请写到 ``hard_negatives.review_state.jsonl``（不是这个 md，也不是
    candidates.jsonl）—— 前两个文件每次生成都会被重写，状态文件不会。
    """
    reviewed = sum(1 for item in items if item.get("reviewed"))
    dropped = sum(1 for item in items if item.get("verdict") == "drop")
    lines = [
        "# hard negative 复核清单",
        "",
        "对每条只需回答一个问题：**知识库里真的有这个答案吗？**",
        "",
        "复核结论写进 **`hard_negatives.review_state.jsonl`**（本文件每次生成都会被覆盖）：",
        "",
        "- 确认「文档确实没有答案」→ 该行 `reviewed` 改成 `true`；",
        "- 发现「文档其实答了」→ `verdict` 改成 `\"drop\"`，并写一句 `reason`；",
        "- 判为 drop 的条目仍留在 candidates 里（可审计），但**不进入 dev/test**。",
        "",
        "- `absence_check` 是关键词的全库命中数，只是**证据**不是判定依据；",
        "- `probe.quote` 是当前 gate **误认为**的证据片段，重点看它为什么不算答案。",
        "",
        f"共 {len(items)} 条（负例 {sum(1 for i in items if not i['expect_answer'])}、"
        f"正例 {sum(1 for i in items if i['expect_answer'])}）· "
        f"已复核 {reviewed} 条 · 判为 drop {dropped} 条。",
        "",
    ]
    by_kind: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        by_kind.setdefault(item["kind"], []).append(item)

    for kind, rows in sorted(by_kind.items()):
        lines += [f"## {kind}（{len(rows)} 条）", ""]
        for item in rows:
            if item.get("verdict") == "drop":
                mark = "❌"
            elif item.get("reviewed"):
                mark = "✅"
            else:
                mark = "🟥" if item["hardness"] == "trap" else ("🟨" if item["hardness"] == "adjacent" else "⬜")
            box = "x" if item.get("reviewed") else " "
            expect = "应回答" if item["expect_answer"] else "应拒答"
            lines += [
                f"- [{box}] {mark} **{item['question']}**",
                f"  - `{item['case_id']}` · 角色 `{item['role']}` · {expect} · "
                f"hardness=`{item['hardness']}` · rel={item['probe']['relevance']:.2f} span={item['probe']['span']}",
                f"  - 锚定事实：{item['anchor']}",
                # 标签必须和 expect_answer 一致：正例不能印"为什么没有答案"
                f"  - {'为什么应当能回答（依据）' if item['expect_answer'] else '为什么没有答案'}："
                f"{item.get('rationale') or item.get('absent_reason', '')}",
                f"  - 关键词命中：{item['absence_check']}",
            ]
            if item.get("review_reason"):
                lines.append(f"  - 复核备注：{item['review_reason']}")
            for top in item["probe"]["top"][:2]:
                lines.append(
                    f"  - 系统认为的证据：`{top['citation']}`（rel={top['relevance']}）「{top['quote']}…」"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
