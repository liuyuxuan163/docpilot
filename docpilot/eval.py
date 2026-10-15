"""评测层：把"答得好不好"变成可量化的数字，并接成 CI 门禁。

为什么知识库问答必须带评测
--------------------------
改了切块大小、调了 RRF 权重、换了一版提示词之后，**靠肉眼看两三个问题是不可能判断好坏的**。
没有评测，这些改动就是"感觉好像好了一点"，而这在项目上线后会变成真实事故。

本项目评四件互相不可替代的事：

===================  ================================================
指标                  衡量什么 / 为什么必要
===================  ================================================
``retrieval_hit``     期望文档是否被检索到 —— 检索层出了问题，生成层再强也没用
``citation_valid``    引用是否真实存在（防"编造来源"，企业场景最危险的失败）
``refusal_correct``   **该拒答时是否拒答** —— 防一本正经地胡说（幻觉）
``answer_coverage``   参考答案要点是否覆盖到 —— 最终业务效果
===================  ================================================

另外单独统计 ``over_refusal``（该答却拒答），因为**过度拒答和幻觉一样是失败**：
一个什么都不肯回答的助手，用户会直接弃用。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .agent import Agent
from .core import Answer, Config

__all__ = ["EvalCase", "CaseScore", "load_cases", "run_eval", "EvalReport"]


@dataclass
class EvalCase:
    case_id: str
    question: str
    role: str = "public"
    expect_answer: bool = True              # True=应该回答；False=应该拒答
    expected_sources: List[str] = field(default_factory=list)   # 期望命中的文档
    expected_keywords: List[str] = field(default_factory=list)  # 答案里应出现的关键点
    history: List[str] = field(default_factory=list)            # 多轮上下文
    note: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any], index: int = 0) -> "EvalCase":
        return cls(
            case_id=str(data.get("case_id") or f"case-{index:03d}"),
            question=str(data.get("question", "")),
            role=str(data.get("role", "public")),
            expect_answer=bool(data.get("expect_answer", True)),
            expected_sources=list(data.get("expected_sources", []) or []),
            expected_keywords=list(data.get("expected_keywords", []) or []),
            history=list(data.get("history", []) or []),
            note=str(data.get("note", "")),
        )


@dataclass
class CaseScore:
    case_id: str
    retrieval_hit: float = 0.0
    citation_valid: float = 0.0
    refusal_correct: float = 0.0
    answer_coverage: float = 0.0
    composite: float = 0.0
    refused: bool = False
    over_refusal: bool = False
    missing_sources: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "case_id": self.case_id,
            "retrieval_hit": round(self.retrieval_hit, 4),
            "citation_valid": round(self.citation_valid, 4),
            "refusal_correct": round(self.refusal_correct, 4),
            "answer_coverage": round(self.answer_coverage, 4),
            "composite": round(self.composite, 4),
            "refused": self.refused,
            "over_refusal": self.over_refusal,
            "tokens": self.tokens,
            "latency_ms": round(self.latency_ms, 1),
        }
        if self.missing_sources:
            data["missing_sources"] = self.missing_sources
        if self.missing_keywords:
            data["missing_keywords"] = self.missing_keywords
        if self.invalid_citations:
            data["invalid_citations"] = self.invalid_citations
        if self.error:
            data["error"] = self.error
        return data


# 权重：检索与引用占比高，因为它们是"不能错"的环节
WEIGHTS = {
    "retrieval_hit": 0.30,
    "citation_valid": 0.25,
    "refusal_correct": 0.25,
    "answer_coverage": 0.20,
}


def _normalize(text: str) -> str:
    """归一化：去空白与标点，便于关键词匹配。"""
    if not text:
        return ""
    lowered = text.lower()
    lowered = re.sub(r"[\s\u3000]+", "", lowered)
    return re.sub(r"[，。、；：！？（）【】《》,.;:!?()\[\]{}<>\"'`~\-_/\\|+=*&^%$#@]+", "", lowered)


def load_cases(path: str | Path) -> List[EvalCase]:
    """加载 JSONL 评测集（``//`` 开头为注释）。"""
    file = Path(path)
    if not file.exists():
        return []
    cases: List[EvalCase] = []
    with open(file, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                cases.append(EvalCase.from_dict(json.loads(stripped), lineno))
            except json.JSONDecodeError:
                continue
    return cases


def score_case(case: EvalCase, answer: Answer) -> CaseScore:
    """给单条用例打分。"""
    score = CaseScore(case_id=case.case_id)
    score.refused = answer.refused
    score.tokens = answer.tokens
    score.latency_ms = answer.latency_ms

    # ---- 1. 检索命中：期望文档是否出现在引用或检索结果里 ----
    seen_sources = {c.source for c in answer.citations} | {c.source for c in answer.retrieved}
    if case.expected_sources:
        hit = len(set(case.expected_sources) & seen_sources)
        score.retrieval_hit = hit / len(case.expected_sources)
        score.missing_sources = [s for s in case.expected_sources if s not in seen_sources]
    else:
        score.retrieval_hit = 1.0

    # ---- 2. 引用有效性：引用必须能对应到检索结果 ----
    if not answer.citations:
        # 该拒答时没有引用是正常的；该回答却完全没有引用要扣分
        score.citation_valid = 1.0 if not case.expect_answer else 0.0
    else:
        valid_ids = {c.chunk_id for c in answer.retrieved}
        bad = [c.marker for c in answer.citations if c.chunk_id not in valid_ids]
        score.invalid_citations = bad
        score.citation_valid = 1.0 - len(bad) / len(answer.citations)

    # ---- 3. 拒答是否恰当（双向考核） ----
    if case.expect_answer:
        # 应该回答：拒答了 = 过度拒答（扣分）；正常回答 = 满分
        if answer.refused:
            score.refusal_correct = 0.0
            score.over_refusal = True
        else:
            score.refusal_correct = 1.0
    else:
        # 应该拒答：拒答 = 满分；硬答 = 0 分（幻觉）
        score.refusal_correct = 1.0 if answer.refused else 0.0

    # ---- 4. 答案要点覆盖 ----
    if case.expected_keywords:
        haystack = _normalize(answer.text)
        matched = [k for k in case.expected_keywords if _normalize(k) in haystack]
        score.answer_coverage = len(matched) / len(case.expected_keywords)
        score.missing_keywords = [k for k in case.expected_keywords if k not in matched]
    else:
        score.answer_coverage = 1.0 if answer.refused == (not case.expect_answer) or case.expect_answer else 1.0

    if not case.expect_answer:
        # 拒答用例不考核"要点覆盖"（本来就该没有内容），把权重让给拒答判定
        score.answer_coverage = 1.0

    score.composite = round(
        WEIGHTS["retrieval_hit"] * score.retrieval_hit
        + WEIGHTS["citation_valid"] * score.citation_valid
        + WEIGHTS["refusal_correct"] * score.refusal_correct
        + WEIGHTS["answer_coverage"] * score.answer_coverage,
        4,
    )
    return score


@dataclass
class EvalReport:
    summary: Dict[str, Any] = field(default_factory=dict)
    cases: List[CaseScore] = field(default_factory=list)
    gates: Dict[str, Any] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return bool(self.gates.get("passed", True))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "summary": self.summary,
            "gates": self.gates,
            "cases": [c.to_dict() for c in self.cases],
        }

    def to_markdown(self) -> str:
        summary = self.summary
        lines = [
            "# 评测报告 · DocPilot",
            "",
            f"- 用例数：{summary.get('cases')}",
            f"- 复合分：**{summary.get('composite')}**",
            "",
            "## 汇总指标",
            "",
            "| 指标 | 值 | 说明 |",
            "|---|---|---|",
            f"| retrieval_hit | {summary.get('retrieval_hit')} | 期望文档是否被检索到 |",
            f"| citation_valid | {summary.get('citation_valid')} | 引用是否真实存在 |",
            f"| refusal_correct | {summary.get('refusal_correct')} | 该拒答时是否拒答 |",
            f"| answer_coverage | {summary.get('answer_coverage')} | 答案要点覆盖率 |",
            f"| over_refusal_rate | {summary.get('over_refusal_rate')} | 该答却拒答的比例 |",
            f"| 平均 token | {summary.get('avg_tokens')} | — |",
            f"| P95 延迟 | {summary.get('p95_latency_ms')} ms | — |",
            "",
            f"## 质量门禁：{'✅ 通过' if self.passed else '❌ 未通过'}",
            "",
            "| 指标 | 实际 | 门槛 | 判定 |",
            "|---|---|---|---|",
        ]
        for check in self.gates.get("checks", []):
            lines.append(f"| {check['metric']} | {check['actual']} | {check['threshold']} | {'✅' if check['ok'] else '❌'} |")
        lines += ["", "## 逐例结果", "", "| 用例 | 复合分 | 检索 | 引用 | 拒答 | 覆盖 | 拒答态 |", "|---|---|---|---|---|---|---|"]
        for case in self.cases:
            lines.append(
                f"| {case.case_id} | {case.composite} | {case.retrieval_hit} | {case.citation_valid} | "
                f"{case.refusal_correct} | {case.answer_coverage} | {'拒答' if case.refused else '作答'} |"
            )
        weak = [c for c in self.cases if c.composite < 0.7]
        if weak:
            lines += ["", "## 待改进用例（复合分 < 0.7）", ""]
            for case in weak:
                detail = []
                if case.missing_sources:
                    detail.append(f"漏检索 {case.missing_sources}")
                if case.missing_keywords:
                    detail.append(f"漏要点 {case.missing_keywords}")
                if case.invalid_citations:
                    detail.append(f"无效引用 {case.invalid_citations}")
                if case.over_refusal:
                    detail.append("该答却拒答")
                if not case.refused and case.refusal_correct == 0:
                    detail.append("该拒答却硬答（疑似幻觉）")
                lines.append(f"- **{case.case_id}**：{'；'.join(detail) or '见逐例结果'}")
        return "\n".join(lines)


def check_gates(summary: Dict[str, Any], gates: Dict[str, float]) -> Dict[str, Any]:
    """按门槛检查指标。``*_ms`` / ``avg_tokens`` 为上限，其余为下限。"""
    checks: List[Dict[str, Any]] = []
    passed = True
    for metric, threshold in (gates or {}).items():
        actual = summary.get(metric)
        if actual is None:
            checks.append({"metric": metric, "actual": None, "threshold": threshold, "ok": False})
            passed = False
            continue
        upper_is_better = metric.endswith("_ms") or metric in ("avg_tokens", "over_refusal_rate")
        ok = float(actual) <= float(threshold) if upper_is_better else float(actual) >= float(threshold)
        checks.append({"metric": metric, "actual": actual, "threshold": threshold, "ok": ok})
        passed = passed and ok
    return {"passed": passed, "checks": checks}


def run_eval(
    agent: Agent,
    cases: Sequence[EvalCase],
    gates: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> EvalReport:
    """跑完整评测并聚合。"""
    scores: List[CaseScore] = []
    for case in cases:
        try:
            answer = agent.ask(case.question, role=case.role, history=case.history or None, reset=True)
            score = score_case(case, answer)
        except Exception as exc:  # noqa: BLE001 - 单例失败记 0 分，不中断整体
            score = CaseScore(case_id=case.case_id, error=f"{type(exc).__name__}: {exc}")
        scores.append(score)
        if verbose:
            mark = "✓" if score.composite >= 0.7 else "✗"
            print(f"  {mark} {case.case_id} composite={score.composite:.3f} refused={score.refused}")

    n = max(1, len(scores))
    def mean(attr: str) -> float:
        return round(sum(float(getattr(s, attr)) for s in scores) / n, 4)

    latencies = sorted(s.latency_ms for s in scores)
    p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))] if latencies else 0.0
    summary = {
        "cases": len(scores),
        "composite": mean("composite"),
        "retrieval_hit": mean("retrieval_hit"),
        "citation_valid": mean("citation_valid"),
        "refusal_correct": mean("refusal_correct"),
        "answer_coverage": mean("answer_coverage"),
        "over_refusal_rate": round(sum(1 for s in scores if s.over_refusal) / n, 4),
        "hallucination_rate": round(
            sum(1 for s in scores if not s.refused and s.refusal_correct == 0) / n, 4
        ),
        "avg_tokens": round(sum(s.tokens for s in scores) / n, 1),
        "p95_latency_ms": round(p95, 1),
    }
    gate_config = gates if gates is not None else {
        "retrieval_hit": 0.85,
        "citation_valid": 0.95,
        "refusal_correct": 0.85,
        "composite": 0.75,
    }
    return EvalReport(summary=summary, cases=scores, gates=check_gates(summary, gate_config))
