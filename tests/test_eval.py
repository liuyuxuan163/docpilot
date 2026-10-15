"""测试：评测层（指标计算、拒答双向考核、门禁）。

评测是**这个项目的核心资产之一**：没有它，改检索/改提示词就只能靠肉眼判断好坏。
所以评测本身的正确性必须被测 —— 而且这里踩过一个很典型的坑：
**指标本身算错，比效果不好更危险**，它会让你朝错误的方向优化。
"""

from __future__ import annotations

import json

import pytest

from docpilot.agent import Agent
from docpilot.core import Answer, Citation, Config, RetrievedChunk
from docpilot.eval import (
    EvalCase,
    WEIGHTS,
    check_gates,
    load_cases,
    run_eval,
    score_case,
)
from docpilot.llm import MockLLM
from docpilot.rag import KnowledgeBase
from docpilot.tools import build_registry


def _answer_with(text: str, citations=None, refused=False, retrieved=None) -> Answer:
    return Answer(
        question="q",
        text=text,
        citations=list(citations or []),
        refused=refused,
        retrieved=list(retrieved or []),
        tokens=100,
        latency_ms=10.0,
    )


def _cite(marker: str, chunk_id: str, source: str = "a.md", section: str = "节") -> Citation:
    return Citation(marker=marker, chunk_id=chunk_id, source=source, section=section)


def _chunk(chunk_id: str, source: str = "a.md") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text="内容", source=source)


# --------------------------------------------------------------------------- #
# 用例加载
# --------------------------------------------------------------------------- #


class TestLoadCases:
    def test_loads_project_benchmark(self, config: Config):
        cases = load_cases(config.path_of("data/eval/cases.jsonl"))
        assert len(cases) >= 20
        assert all(case.question for case in cases)
        assert all(case.case_id for case in cases)

    def test_skips_comments_and_blank_lines(self, workdir):
        path = workdir / "cases.jsonl"
        path.write_text(
            '// 这是注释\n\n{"case_id": "a", "question": "q1"}\n\n{"case_id": "b", "question": "q2"}\n',
            encoding="utf-8",
        )
        cases = load_cases(path)
        assert [case.case_id for case in cases] == ["a", "b"]

    def test_missing_file_returns_empty(self, workdir):
        assert load_cases(workdir / "nope.jsonl") == []

    def test_bad_json_line_is_skipped(self, workdir):
        path = workdir / "cases.jsonl"
        path.write_text('{"case_id": "a", "question": "q"}\n{坏行\n', encoding="utf-8")
        assert len(load_cases(path)) == 1

    def test_defaults_are_sane(self):
        case = EvalCase.from_dict({"question": "q"}, index=3)
        assert case.case_id == "case-003"
        assert case.role == "public"
        assert case.expect_answer is True


# --------------------------------------------------------------------------- #
# 单例打分
# --------------------------------------------------------------------------- #


class TestScoreCase:
    def test_perfect_answer_scores_one(self):
        case = EvalCase(
            case_id="c",
            question="q",
            expected_sources=["a.md"],
            expected_keywords=["五天"],
        )
        answer = _answer_with(
            "最多结转五天 [1]",
            citations=[_cite("[1]", "c1")],
            retrieved=[_chunk("c1")],
        )
        score = score_case(case, answer)
        assert score.composite == 1.0
        assert score.retrieval_hit == 1.0
        assert score.citation_valid == 1.0
        assert score.refusal_correct == 1.0
        assert score.answer_coverage == 1.0

    def test_missing_source_lowers_retrieval_hit(self):
        case = EvalCase(case_id="c", question="q", expected_sources=["a.md", "b.md"])
        answer = _answer_with("答案 [1]", citations=[_cite("[1]", "c1", "a.md")], retrieved=[_chunk("c1", "a.md")])
        score = score_case(case, answer)
        assert score.retrieval_hit == 0.5
        assert score.missing_sources == ["b.md"]

    def test_fabricated_citation_is_penalized(self):
        """引用了不存在的 chunk → citation_valid 必须下降。

        伪造来源是企业场景最危险的失败，指标必须能抓到它。
        """
        case = EvalCase(case_id="c", question="q")
        answer = _answer_with(
            "答案 [1][2]",
            citations=[_cite("[1]", "c1"), _cite("[2]", "不存在")],
            retrieved=[_chunk("c1")],
        )
        score = score_case(case, answer)
        assert score.citation_valid == 0.5
        assert score.invalid_citations == ["[2]"]

    def test_answer_without_any_citation_scores_zero_on_citation(self):
        case = EvalCase(case_id="c", question="q", expect_answer=True)
        score = score_case(case, _answer_with("没有标号的回答"))
        assert score.citation_valid == 0.0

    def test_correct_refusal_is_full_marks(self):
        case = EvalCase(case_id="c", question="超纲问题", expect_answer=False)
        score = score_case(case, _answer_with("知识库中没有相关依据", refused=True))
        assert score.refusal_correct == 1.0
        assert score.over_refusal is False

    def test_hallucinating_on_unanswerable_is_zero(self):
        """该拒答却硬答 → 拒答分 0（幻觉）。"""
        case = EvalCase(case_id="c", question="超纲问题", expect_answer=False)
        score = score_case(case, _answer_with("我觉得是 42", refused=False))
        assert score.refusal_correct == 0.0

    def test_over_refusal_is_detected(self):
        """该答却拒答同样是失败 —— 一个什么都不答的助手会被用户弃用。"""
        case = EvalCase(case_id="c", question="正常问题", expect_answer=True)
        score = score_case(case, _answer_with("无法回答", refused=True))
        assert score.refusal_correct == 0.0
        assert score.over_refusal is True

    def test_keyword_coverage_is_case_insensitive_and_punctuation_free(self):
        case = EvalCase(case_id="c", question="q", expected_keywords=["15个工作日"])
        answer = _answer_with("需在 15 个工作日 内提交 [1]", citations=[_cite("[1]", "c1")], retrieved=[_chunk("c1")])
        assert score_case(case, answer).answer_coverage == 1.0

    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_score_is_bounded(self):
        case = EvalCase(case_id="c", question="q")
        score = score_case(case, _answer_with("任意回答"))
        assert 0.0 <= score.composite <= 1.0


# --------------------------------------------------------------------------- #
# 门禁
# --------------------------------------------------------------------------- #


class TestGates:
    def test_lower_bound_metric(self):
        result = check_gates({"retrieval_hit": 0.9}, {"retrieval_hit": 0.8})
        assert result["passed"] is True
        result = check_gates({"retrieval_hit": 0.5}, {"retrieval_hit": 0.8})
        assert result["passed"] is False

    def test_upper_bound_metric(self):
        # avg_tokens / *_ms / over_refusal_rate 是**上限**指标
        assert check_gates({"avg_tokens": 100}, {"avg_tokens": 200})["passed"] is True
        assert check_gates({"avg_tokens": 300}, {"avg_tokens": 200})["passed"] is False
        assert check_gates({"p95_latency_ms": 5000}, {"p95_latency_ms": 1000})["passed"] is False
        assert check_gates({"over_refusal_rate": 0.05}, {"over_refusal_rate": 0.1})["passed"] is True

    def test_missing_metric_fails_closed(self):
        assert check_gates({}, {"retrieval_hit": 0.8})["passed"] is False

    def test_empty_gates_pass(self):
        assert check_gates({"any": 0.0}, {})["passed"] is True


# --------------------------------------------------------------------------- #
# 端到端评测
# --------------------------------------------------------------------------- #


class TestRunEval:
    def test_runs_real_benchmark_and_passes_gates(self, kb: KnowledgeBase, config: Config):
        """跑真实评测集 + 真实门禁。这是最有价值的一条测试：它是回归防线。"""
        cases = load_cases(config.path_of("data/eval/cases.jsonl"))
        agent = Agent(config=config, kb=kb, llm=MockLLM(), registry=build_registry(kb))
        report = run_eval(agent, cases, gates=config.get("eval.gates", {}))

        assert report.summary["cases"] == len(cases)
        assert report.summary["citation_valid"] >= 0.95
        assert report.summary["over_refusal_rate"] <= 0.15
        assert report.passed, report.to_markdown()

    def test_report_renders_markdown(self, kb: KnowledgeBase, config: Config):
        cases = load_cases(config.path_of("data/eval/cases.jsonl"))[:3]
        agent = Agent(config=config, kb=kb, llm=MockLLM(), registry=build_registry(kb))
        report = run_eval(agent, cases)
        markdown = report.to_markdown()
        assert "评测报告" in markdown
        assert "retrieval_hit" in markdown
        assert "质量门禁" in markdown

    def test_case_exception_is_captured_not_raised(self, kb: KnowledgeBase, config: Config):
        class BoomLLM(MockLLM):
            def complete(self, *args, **kwargs):
                raise RuntimeError("炸了")

        agent = Agent(config=config, kb=kb, llm=BoomLLM(), registry=build_registry(kb))
        report = run_eval(agent, [EvalCase(case_id="c", question="q")])
        assert report.cases[0].composite == 0.0

    def test_report_serializes_to_json(self, kb: KnowledgeBase, config: Config):
        agent = Agent(config=config, kb=kb, llm=MockLLM(), registry=build_registry(kb))
        report = run_eval(agent, [EvalCase(case_id="c", question="年假结转到明年")])
        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["summary"]["cases"] == 1
        assert "gates" in payload

    def test_empty_case_list(self, kb: KnowledgeBase, config: Config):
        agent = Agent(config=config, kb=kb, llm=MockLLM(), registry=build_registry(kb))
        report = run_eval(agent, [])
        assert report.summary["cases"] == 0
