"""测试：Agent 主循环、引用映射、多轮会话、LLM 双模式、评测指标。

重点覆盖**编排层**的逻辑：预算、引用校验、拒答识别、历史管理。
这些都是"不报错但结果错"的高发区。
"""

from __future__ import annotations

import json

import pytest

from docpilot.agent import SYSTEM_PROMPT, Agent, Conversation
from docpilot.core import Config, LLMError, Message, ToolCall
from docpilot.llm import BaseLLM, LLMResponse, MockLLM, _extract_json, _repair_json, build_llm, estimate_tokens
from docpilot.rag import KnowledgeBase
from docpilot.tools import build_registry


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class ScriptedLLM(BaseLLM):
    """按脚本依次返回响应，用来精确控制 ReAct 循环的走向。"""

    provider = "scripted"
    model = "scripted"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        if not self.responses:
            return LLMResponse(content="（脚本用尽）", prompt_tokens=1, completion_tokens=1)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _agent(config: Config, kb: KnowledgeBase, llm: BaseLLM) -> Agent:
    return Agent(config=config, kb=kb, llm=llm, registry=build_registry(kb))


# --------------------------------------------------------------------------- #
# LLM 层
# --------------------------------------------------------------------------- #


class TestEstimateTokens:
    def test_scales_with_length(self):
        assert estimate_tokens("a" * 220) > estimate_tokens("a" * 22)

    def test_never_zero(self):
        assert estimate_tokens("") >= 1


class TestJsonExtraction:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_nested_json(self):
        data = _extract_json('前面 {"a": {"b": [1, 2]}} 后面')
        assert data == {"a": {"b": [1, 2]}}

    def test_truncated_json_is_repaired(self):
        """截断的 JSON 必须能修复。

        工具结果会被按字符数截断后才喂给模型，截断点很可能落在 JSON 中间。
        不做修复会让下游误判成"检索没有结果"——本项目真实踩过这个坑。
        """
        truncated = '{"tool": "search_docs", "results": [{"source": "a.md", "text": "内容'
        data = _extract_json(truncated)
        assert data is not None
        assert data["tool"] == "search_docs"

    def test_truncated_nested_structure_is_repaired(self):
        truncated = '{"tool": "search_docs", "results": [{"source": "a.md"}, {"source": "b.md"'
        data = _extract_json(truncated)
        assert data is not None
        assert data["results"][0]["source"] == "a.md"

    def test_garbage_returns_none(self):
        assert _extract_json("这里没有 JSON") is None
        assert _extract_json("") is None
        assert _extract_json(None) is None

    def test_repair_handles_unclosed_string(self):
        repaired = _repair_json('{"a": "未闭合')
        assert json.loads(repaired)["a"] == "未闭合"


class TestBuildLLM:
    def test_mock_by_default(self, config: Config):
        assert isinstance(build_llm(config), MockLLM)

    def test_falls_back_to_mock_without_key(self, config: Config, monkeypatch):
        """没配 Key 时必须**降级**而不是抛错。

        "新同学 clone 完三分钟能跑起来"是关键体验：
        环境不全时系统应该可用并说明降级，而不是直接起不来。
        """
        monkeypatch.setenv("DOCPILOT_PROVIDER", "deepseek")
        for name in ("DOCPILOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert isinstance(build_llm(config), MockLLM)

    def test_builds_openai_compat_with_key(self, config: Config, monkeypatch):
        from docpilot.llm import OpenAICompatLLM

        monkeypatch.setenv("DOCPILOT_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        llm = build_llm(config)
        assert isinstance(llm, OpenAICompatLLM)
        assert llm.base_url == "https://api.deepseek.com/v1"


class TestMockLLM:
    def test_calls_rewrite_first_for_coreference(self, kb: KnowledgeBase, config: Config):
        llm = MockLLM()
        messages = [
            Message(role="user", content="## 历史\n- 用户：年假能结转到明年吗"),
            Message(role="user", content="问题：那最多多少天"),
        ]
        response = llm.complete(messages, tools=build_registry(kb).schemas())
        assert [call.name for call in response.tool_calls] == ["rewrite_question"]

    def test_calls_search_for_fresh_question(self, kb: KnowledgeBase):
        llm = MockLLM()
        messages = [Message(role="user", content="问题：年假能结转到明年吗")]
        response = llm.complete(messages, tools=build_registry(kb).schemas())
        assert [call.name for call in response.tool_calls] == ["search_docs"]

    def test_answers_after_search(self, kb: KnowledgeBase):
        llm = MockLLM()
        payload = {
            "tool": "search_docs",
            "has_evidence": True,
            "best_relevance": 0.8,
            "margin": 0.1,
            "role": "public",
            "results": [{"text": "年假最多结转五天", "citation": "hr/leave-policy.md#年假", "source": "hr/leave-policy.md"}],
        }
        messages = [
            Message(role="user", content="问题：年假结转"),
            Message(role="assistant", content="检索", tool_calls=[ToolCall(name="search_docs")]),
            Message(role="tool", content=json.dumps(payload, ensure_ascii=False), name="search_docs"),
        ]
        response = llm.complete(messages, tools=build_registry(kb).schemas())
        assert response.tool_calls == []
        assert "5" in response.content or "五" in response.content

    def test_is_deterministic(self, kb: KnowledgeBase):
        messages = [Message(role="user", content="问题：年假能结转到明年吗")]
        tools = build_registry(kb).schemas()
        first = MockLLM().complete(messages, tools)
        second = MockLLM().complete(messages, tools)
        assert first.content == second.content
        assert [c.name for c in first.tool_calls] == [c.name for c in second.tool_calls]


# --------------------------------------------------------------------------- #
# 主循环
# --------------------------------------------------------------------------- #


class TestAgentLoop:
    def test_full_loop_produces_answer_with_citations(self, kb: KnowledgeBase, config: Config):
        agent = _agent(config, kb, MockLLM())
        answer = agent.ask("年假可以结转到明年吗", role="public", reset=True)
        assert answer.text
        assert answer.refused is False
        assert answer.citations
        assert answer.steps >= 2
        assert answer.tokens > 0

    def test_stops_at_max_steps(self, kb: KnowledgeBase, config: Config):
        """步数用尽时必须给出**部分结论**，而不是空手而归。"""
        config._data["agent"]["max_steps"] = 1
        llm = ScriptedLLM([
            LLMResponse(
                content="检索一下",
                tool_calls=[ToolCall(name="search_docs", arguments={"query": "年假"})],
                prompt_tokens=10,
                completion_tokens=5,
            ),
        ])
        answer = _agent(config, kb, llm).ask("年假可以结转到明年吗", reset=True)
        assert answer.degraded is True
        assert answer.text  # 不能是空字符串

    def test_respects_max_tool_calls(self, kb: KnowledgeBase, config: Config):
        config._data["agent"]["max_tool_calls"] = 1
        call = ToolCall(name="search_docs", arguments={"query": "年假"})
        llm = ScriptedLLM([LLMResponse(content="查", tool_calls=[call], prompt_tokens=1, completion_tokens=1)] * 5)
        answer = _agent(config, kb, llm).ask("年假可以结转到明年吗", reset=True)
        assert len(answer.tool_calls) <= 1

    def test_llm_error_degrades_gracefully(self, kb: KnowledgeBase, config: Config):
        """模型挂了要给出可读信息，而不是抛异常把调用方炸掉。"""
        answer = _agent(config, kb, ScriptedLLM([LLMError("模拟网络故障")])).ask("年假结转", reset=True)
        assert answer.degraded is True
        assert "模型调用失败" in answer.text

    def test_trace_is_recorded(self, kb: KnowledgeBase, config: Config):
        answer = _agent(config, kb, MockLLM()).ask("年假可以结转到明年吗", reset=True)
        events = [step.get("event") for step in answer.trace]
        assert "llm" in events and "tool" in events

    def test_refusal_is_detected(self, kb: KnowledgeBase, config: Config):
        # 用"完全无字面重合"的提问：这类词面启发式能可靠拒答
        answer = _agent(config, kb, MockLLM()).ask("今天天气怎么样？", reset=True)
        assert answer.refused is True
        assert answer.refusal_reason

    def test_out_of_scope_question_with_partial_overlap_is_known_gap(self, kb: KnowledgeBase, config: Config):
        """记录一个**已知的、被如实公开的**能力边界。

        "公司的股票代码是多少？" 与《信息安全规定》里讲"代码"的段落有字面重合
        （覆盖率 0.58、连续命中 2 字），词面启发式无法分辨
        "文档提到了这个话题" 与 "文档回答了这个问题"，于是会作答。

        这类失败在评测里被计入 ``hallucination_rate``，并写进 README 的已知限制。
        **我们选择让测试如实记录它，而不是把阈值调到刚好把它挡住** ——
        那样做等于对着评测集过拟合，换一份文档就会失效。

        如果要真正修掉它，需要真实 embedding 或一次模型判定。
        """
        answer = _agent(config, kb, MockLLM()).ask("公司的股票代码是多少？", reset=True)
        # 不做"必须拒答"的强断言，只断言：如果作答了，必须带引用（不许凭空编）
        if not answer.refused:
            assert answer.citations, "作答但没有任何引用，说明可能凭空编造"
            assert any("security-policy" in label or "it/" in label for label in answer.citation_labels)

    def test_permission_refusal_names_the_document(self, kb: KnowledgeBase, config: Config):
        """无权限时要明确告知，而不是拿别的文档含糊作答。"""
        answer = _agent(config, kb, MockLLM()).ask("P4 职级的绩效工资占比是多少？", role="public", reset=True)
        assert answer.refused is True
        assert "compensation" in answer.text
        assert "权限" in answer.text

    def test_authorized_role_gets_answer(self, kb: KnowledgeBase, config: Config):
        answer = _agent(config, kb, MockLLM()).ask("P4 职级的绩效工资占比是多少？", role="hr", reset=True)
        assert answer.refused is False
        assert any("compensation" in label for label in answer.citation_labels)


class TestCitationMapping:
    def test_invalid_marker_is_recorded_not_fabricated(self, kb: KnowledgeBase, config: Config):
        """模型写出超出检索范围的 ``[9]`` 时，框架不能臆造来源。

        引用的 ``[n] → 文档`` 映射由框架生成，并且要**校验**标号范围。
        """
        payload = {
            "tool": "search_docs",
            "has_evidence": True,
            "best_relevance": 0.8,
            "role": "public",
            "results": [{"text": "内容", "citation": "a.md#节", "source": "a.md", "chunk_id": "c1"}],
        }
        model = ScriptedLLM([
            LLMResponse(content="检索", tool_calls=[ToolCall(name="search_docs", arguments={"query": "x"})], prompt_tokens=1, completion_tokens=1),
            LLMResponse(content="答案是 [9] 这样。", prompt_tokens=1, completion_tokens=1),
        ])
        agent = _agent(config, kb, model)
        # 用真实的检索结果替换脚本里的 payload
        agent.registry._tools["search_docs"].handler = lambda query, top_k=4, role="public": payload
        answer = agent.ask("任意问题", reset=True)
        assert any(step.get("event") == "citation_check" for step in answer.trace)
        assert all(cite.marker == "[9]" or cite.chunk_id == "c1" for cite in answer.citations)
        assert not any(cite.chunk_id not in {"c1"} for cite in answer.citations)

    def test_missing_markers_get_fallback_citations(self, kb: KnowledgeBase, config: Config):
        """模型忘了标引用时，要附上检索来源，而不是留空。"""
        payload = {
            "tool": "search_docs",
            "has_evidence": True,
            "best_relevance": 0.8,
            "role": "public",
            "results": [
                {"text": "内容一", "citation": "a.md#节", "source": "a.md", "chunk_id": "c1"},
                {"text": "内容二", "citation": "b.md#节", "source": "b.md", "chunk_id": "c2"},
            ],
        }
        model = ScriptedLLM([
            LLMResponse(content="检索", tool_calls=[ToolCall(name="search_docs", arguments={"query": "x"})], prompt_tokens=1, completion_tokens=1),
            LLMResponse(content="这是没有任何标号的回答。", prompt_tokens=1, completion_tokens=1),
        ])
        agent = _agent(config, kb, model)
        agent.registry._tools["search_docs"].handler = lambda query, top_k=4, role="public": payload
        answer = agent.ask("任意问题", reset=True)
        assert len(answer.citations) == 2
        assert {cite.chunk_id for cite in answer.citations} == {"c1", "c2"}


class TestConversation:
    def test_remembers_and_trims(self):
        conversation = Conversation(max_turns=2)
        for i in range(3):
            conversation.remember(f"q{i}", f"a{i}")
        assert conversation.recent_questions == ["q1", "q2"]

    def test_transcript_renders_history(self):
        conversation = Conversation()
        conversation.remember("年假能结转吗", "可以，最多五天")
        transcript = conversation.transcript()
        assert "年假能结转吗" in transcript
        assert "对话历史" in transcript

    def test_reset_clears(self):
        conversation = Conversation()
        conversation.remember("q", "a")
        conversation.reset()
        assert conversation.recent_questions == []
        assert conversation.transcript() == ""


class TestMultiTurn:
    def test_followup_uses_history(self, kb: KnowledgeBase, config: Config):
        agent = _agent(config, kb, MockLLM())
        agent.ask("年假可以结转到明年吗？", role="public", reset=True)
        answer = agent.ask("那最多能结转多少天？", role="public")
        assert answer.refused is False
        assert answer.rewritten_question
        assert "年假" in answer.rewritten_question

    def test_reset_starts_new_session(self, kb: KnowledgeBase, config: Config):
        agent = _agent(config, kb, MockLLM())
        agent.ask("年假可以结转到明年吗？", reset=True)
        assert agent.conversation.recent_questions
        agent.ask("密码多少位？", reset=True)
        assert len(agent.conversation.recent_questions) == 1

    def test_explicit_history_overrides_conversation(self, kb: KnowledgeBase, config: Config):
        agent = _agent(config, kb, MockLLM())
        answer = agent.ask("那最多多少天？", history=["年假可以结转到明年吗？"])
        assert "年假" in (answer.rewritten_question or "")


class TestSystemPrompt:
    def test_contains_the_three_disciplines(self):
        """提示词的"纪律"是这个项目最核心的产品决策，值得被测试锁住。"""
        assert "只用检索到的内容" in SYSTEM_PROMPT
        assert "必须标注引用" in SYSTEM_PROMPT
        assert "检索不到就说不知道" in SYSTEM_PROMPT

    def test_forbids_guessing(self):
        assert "不要推测" in SYSTEM_PROMPT
