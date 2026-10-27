"""评测"拒答识别"本身：在已保存的真实模型答案上对比几种判定器。

用法::

    python scripts/eval_refusal_detector.py

为什么需要这个脚本
------------------
`agent._looks_like_refusal` 的关键词表是照着 **Mock 的话术**写的
（"无法回答/没有找到/没有访问权限"）。接真实模型后，它的说法变成
"知识库中没有 X 这个说法 / 未检索到 / 没有直接规定 / 没有给出"——
表里都没有，于是**两个方向都错**：

* 正确拒答被漏判成"作答"（虚高误答率）；
* 正常作答被误判成"拒答"（虚高误拒率）。

这直接决定"真实模型误答率"这个数字可不可信，所以必须单独评测**判定器本身**。

验证集：已经落盘的真实模型答案（主集 77 + holdout 31 + 原评测集 21 = 129 条），
标签用 ``expect_answer``（负例=应当拒答，正例=应当作答）。
**零 API 成本**：只读 JSON。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docpilot.agent import _looks_like_refusal  # noqa: E402
from docpilot.core import ensure_utf8_stdio  # noqa: E402

DATA = ROOT / "data" / "eval"
FILES = ["real_llm_answers.main.json", "real_llm_answers.holdout.json", "real_llm_answers.cases.json"]

# 真实模型常见的拒答说法（比项目内置表宽得多）
WIDE = (
    "无法回答", "无法提供", "无法给出", "无法确认", "无法判断", "不确定",
    "没有找到", "未找到", "没有检索到", "未检索到", "检索不到",
    "没有相关", "没有关于", "没有提及", "未提及", "没有提到", "未包含", "不包含",
    "未涉及", "没有涉及", "没有直接规定", "没有给出", "没有写明", "没有规定",
    "暂无", "缺乏依据", "不足以", "不做推测", "没有访问权限", "无权访问",
    "涉及受限文档", "受限来源", "建议咨询", "建议向", "请补充", "无法据此",
)


# 真实模型常见的拒答句式：不靠死词表，而是"否定词 + 缺什么"的组合
NEGATION_RE = re.compile(
    r"(没有|未|无法|不能|暂无|缺乏)[^。；\n]{0,14}"
    r"(规定|条款|依据|说明|标准|时限|比例|数据|信息|说法|内容|记录|办法|流程)"
)
DENY_RE = re.compile(r"(不可以|不能|无法|不得|并非|不是|不予|无需)")


def load() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name in FILES:
        path = DATA / name
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            if "error" in row or "text" not in row:
                continue
            row["_source"] = name
            rows.append(row)
    return rows


def first_n(text: str, limit: int) -> str:
    """只看前 N 个字符：拒答的结论几乎总在第一段，往后看反而会误命中
    （正常作答里也会出现"文档未包含 X"这种补充说明）。"""
    return re.sub(r"\s+", "", (text or ""))[:limit]


def detector_current(row: Dict[str, Any]) -> bool:
    """当前项目里的实现（结论区 + 正则），直接调线上函数。"""
    return _looks_like_refusal(row.get("text", ""))


def detector_legacy(row: Dict[str, Any]) -> bool:
    """改动前的实现：全文匹配一份 Mock 话术词表。留作对照。"""
    text = row.get("text", "") or ""
    legacy = ("无法回答", "没有找到", "未找到", "没有相关依据", "知识库中没有任何",
              "不足以支撑", "不做推测", "没有访问权限", "无权访问", "涉及受限文档")
    return any(phrase in text for phrase in legacy)


def detector_wide(row: Dict[str, Any]) -> bool:
    head = first_n(row.get("text", ""), 400)
    return any(phrase in head for phrase in WIDE)


def first_line(text: str) -> str:
    """只取第一个"结论型"片段：跳过 "结论：" 前缀，到第一个句号/换行/分号为止。

    踩过两个坑：
    1. 把 ``：`` 也当结束符 → ``"结论：知识库中没有…"`` 直接被截成 ``"结论"``；
    2. 不跳过 ``结论`` 前缀 → 正则匹配到的否定词被前缀挤到窗口外。
    """
    head = re.sub(r"\s+", " ", (text or "").strip()).lstrip("#* ")
    head = re.sub(r"^(结论|答案|回答|简答)\s*[：:]?\s*", "", head)
    for sep in ("。", "；", "\n"):
        index = head.find(sep)
        if 0 < index <= 80:
            head = head[:index]
            break
    return head[:90]


def detector_first_line(row: Dict[str, Any]) -> bool:
    line = first_line(row.get("text", ""))
    if any(phrase in line for phrase in WIDE):
        return True
    return bool(NEGATION_RE.search(line))


def detector_first_line_or_current(row: Dict[str, Any]) -> bool:
    return detector_first_line(row) or detector_current(row)


def detector_wide_or_nocite(row: Dict[str, Any]) -> bool:
    """宽表命中 **或** 没有引用且正文里有否定词。多一条冗余信号。"""
    if detector_wide(row):
        return True
    head = first_n(row.get("text", ""), 200)
    return not row.get("citations") and any(w in head for w in ("没有", "未", "不"))


VARIANTS: List[tuple[str, Callable[[Dict[str, Any]], bool]]] = [
    ("项目内置（旧：全文词表）", detector_legacy),
    ("项目内置（新：结论区+正则）", detector_current),
    ("宽关键词表（全文 400 字）", detector_wide),
    ("宽关键词表（只看结论行）", detector_first_line),
    ("宽表 + 无引用兜底", detector_wide_or_nocite),
]


def main() -> int:
    ensure_utf8_stdio()
    rows = load()
    neg = [row for row in rows if not row.get("expect_answer", True)]
    pos = [row for row in rows if row.get("expect_answer", True)]
    print(f"验证集：{len(rows)} 条真实模型答案（负例 {len(neg)} / 正例 {len(pos)}）")
    print(f"来源：{', '.join(sorted({row['_source'] for row in rows}))}\n")

    header = f"{'判定器':<28}{'负例召回（认出拒答）':>22}{'正例误判（当成拒答）':>24}"
    print(header)
    print("-" * len(header))
    for label, fn in VARIANTS:
        hit = sum(1 for row in neg if fn(row))
        false_pos = sum(1 for row in pos if fn(row))
        recall = hit / len(neg) if neg else 0.0
        fpr = false_pos / len(pos) if pos else 0.0
        print(f"{label:<28}{f'{hit}/{len(neg)} ({recall:.1%})':>22}{f'{false_pos}/{len(pos)} ({fpr:.1%})':>24}")

    # 项目内置表漏掉的真实说法（这些就是要补进词表的）
    missed = [row for row in neg if not detector_current(row) and detector_wide(row)]
    print(f"\n项目内置表漏判、但宽表能认出的 {len(missed)} 条 —— 这些短语应当补进 `_looks_like_refusal`：")
    phrases: Dict[str, int] = {}
    for row in missed:
        head = first_n(row["text"], 120)
        for phrase in WIDE:
            if phrase in head:
                phrases[phrase] = phrases.get(phrase, 0) + 1
    for phrase, count in sorted(phrases.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {phrase:<12}{count} 次")

    print("\n结论：接真实模型后，**判定器本身也要被评测** ——")
    print("否则'误答率'这个数字反映的是词表覆盖率，而不是模型行为。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
