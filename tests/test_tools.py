"""测试：工具层（schema 校验、权限注入、证据判定、注册表语义）。"""

from __future__ import annotations

import pytest

from docpilot.core import ToolError
from docpilot.rag import KnowledgeBase
from docpilot.tools import (
    ToolRegistry,
    ToolResult,
    _rewrite,
    build_registry,
    validate_args,
)


# --------------------------------------------------------------------------- #
# 入参校验
# --------------------------------------------------------------------------- #


class TestValidateArgs:
    SCHEMA = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 4, "minimum": 1, "maximum": 10},
            "mode": {"type": "string", "enum": ["fast", "full"]},
        },
        "required": ["query"],
    }

    def test_accepts_valid_args(self):
        out = validate_args(self.SCHEMA, {"query": "年假", "top_k": 3})
        assert out["query"] == "年假"
        assert out["top_k"] == 3

    def test_applies_default(self):
        assert validate_args(self.SCHEMA, {"query": "x"})["top_k"] == 4

    def test_missing_required_raises(self):
        with pytest.raises(ToolError, match="缺少必填参数"):
            validate_args(self.SCHEMA, {})

    def test_type_mismatch_raises(self):
        with pytest.raises(ToolError, match="期望 integer"):
            validate_args(self.SCHEMA, {"query": "x", "top_k": "3"})

    def test_bool_is_not_integer(self):
        """``bool`` 是 ``int`` 的子类，必须显式排除。

        不排除的话 ``True`` 会被当成合法的 integer 1，
        这类"看着能跑但语义错了"的参数错误很难排查。
        """
        with pytest.raises(ToolError, match="期望 integer"):
            validate_args(self.SCHEMA, {"query": "x", "top_k": True})

    def test_below_minimum_raises(self):
        with pytest.raises(ToolError, match="小于下限"):
            validate_args(self.SCHEMA, {"query": "x", "top_k": 0})

    def test_above_maximum_raises(self):
        with pytest.raises(ToolError, match="大于上限"):
            validate_args(self.SCHEMA, {"query": "x", "top_k": 99})

    def test_enum_violation_raises(self):
        with pytest.raises(ToolError, match="只能是"):
            validate_args(self.SCHEMA, {"query": "x", "mode": "other"})

    def test_unknown_keys_are_passed_through(self):
        # 未知参数不报错（向前兼容），但也不会被 schema 校验
        out = validate_args(self.SCHEMA, {"query": "x", "extra": 1})
        assert out["extra"] == 1

    def test_empty_schema_passes_through(self):
        assert validate_args({}, {"anything": 1}) == {"anything": 1}


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #


class TestToolRegistry:
    def test_empty_registry_is_truthy(self):
        """空注册表必须为真值。

        不定义 ``__bool__`` 时 Python 会退回 ``__len__``，
        "空注册表"被判为假，于是 ``registry or GLOBAL`` 这类写法
        会**静默丢弃传入的注册表**（本项目踩过这个坑）。
        """
        assert bool(ToolRegistry()) is True
        assert len(ToolRegistry()) == 0

    def test_duplicate_registration_is_idempotent(self):
        registry = ToolRegistry()
        registry.register("t", lambda: 1, description="a")
        registry.register("t", lambda: 2, description="b")
        assert len(registry) == 1

    def test_schemas_are_openai_shaped(self):
        registry = ToolRegistry()
        registry.register(
            "search_docs",
            lambda query, top_k=4: {"ok": True},
            description="检索",
            schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        )
        schema = registry.schemas()[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search_docs"
        assert schema["function"]["parameters"]["required"] == ["query"]

    def test_execute_validates_and_returns_result(self):
        registry = ToolRegistry()
        registry.register("echo", lambda text: {"echo": text}, schema={
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"],
        })
        result = registry.execute("echo", {"text": "hi"})
        assert result.ok and result.payload == {"echo": "hi"}

    def test_execute_wraps_failure_instead_of_raising(self):
        """工具挂掉不能把整轮对话搞崩 —— 必须返回结果对象而不是抛异常。"""
        registry = ToolRegistry()

        def boom():
            raise RuntimeError("内部错误")

        registry.register("boom", boom, schema={"type": "object", "properties": {}})
        result = registry.execute("boom", {})
        assert result.ok is False
        assert "内部错误" in result.error

    def test_unknown_tool_returns_error_result(self):
        result = ToolRegistry().execute("nope", {})
        assert result.ok is False
        assert "未注册的工具" in result.error

    def test_invalid_args_return_error_result(self):
        registry = ToolRegistry()
        registry.register("t", lambda need: 1, schema={
            "type": "object", "properties": {"need": {"type": "string"}}, "required": ["need"],
        })
        result = registry.execute("t", {})
        assert result.ok is False

    def test_call_counts_are_tracked(self):
        registry = ToolRegistry()
        registry.register("t", lambda: 1, schema={"type": "object", "properties": {}})
        registry.execute("t", {})
        registry.execute("t", {})
        assert registry.call_counts["t"] == 2


class TestToolResultSummary:
    def test_error_is_serialized(self):
        result = ToolResult(name="t", ok=False, error="炸了")
        assert "炸了" in result.summary()

    def test_long_text_fields_are_shrunk_not_cut(self):
        """截断必须"缩字段"，不能"切字符串"。

        直接切序列化后的 JSON 会把结构切坏，下游解析失败后会误判成
        "检索没有结果" —— 本项目真实踩过：**引用里有正确片段，回答却说没检索到**。
        """
        import json

        payload = {"tool": "search_docs", "results": [{"text": "长" * 5000, "source": "a.md"}]}
        summary = ToolResult(name="search_docs", ok=True, payload=payload).summary(limit=200)
        # 截断后仍然必须是合法 JSON
        parsed = json.loads(summary)
        assert parsed["results"][0]["source"] == "a.md"
        assert len(parsed["results"][0]["text"]) <= 201

    def test_short_payload_is_untouched(self):
        payload = {"tool": "x", "results": [{"text": "短"}]}
        assert "短" in ToolResult(name="x", ok=True, payload=payload).summary()


# --------------------------------------------------------------------------- #
# 内置工具
# --------------------------------------------------------------------------- #


class TestBuiltinTools:
    def test_registry_exposes_three_tools(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        assert registry.names == ["list_documents", "rewrite_question", "search_docs"]

    def test_search_docs_returns_citation_and_signals(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        result = registry.execute("search_docs", {"query": "年假结转到明年", "top_k": 3}, role="public")
        assert result.ok
        payload = result.payload
        assert payload["count"] >= 1
        assert "relevance" in payload["results"][0]
        assert "citation" in payload["results"][0]
        assert "has_evidence" in payload

    def test_search_docs_respects_role(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        public = registry.execute("search_docs", {"query": "绩效工资占比", "top_k": 5}, role="public")
        hr = registry.execute("search_docs", {"query": "绩效工资占比", "top_k": 5}, role="hr")
        public_sources = {item["source"] for item in public.payload["results"]}
        hr_sources = {item["source"] for item in hr.payload["results"]}
        assert "hr/compensation.md" not in public_sources
        assert "hr/compensation.md" in hr_sources

    def test_search_docs_flags_restricted_target(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        result = registry.execute("search_docs", {"query": "P4 职级的绩效工资占比是多少", "top_k": 4}, role="public")
        assert result.payload["restricted_sources"] == ["hr/compensation.md"]
        assert result.payload["has_evidence"] is False

    def test_has_evidence_true_for_answerable_question(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        result = registry.execute("search_docs", {"query": "年假可以结转到明年吗", "top_k": 4}, role="public")
        assert result.payload["has_evidence"] is True

    def test_has_evidence_false_for_out_of_scope(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        result = registry.execute("search_docs", {"query": "今天天气怎么样", "top_k": 4}, role="public")
        assert result.payload["has_evidence"] is False

    def test_role_is_injected_and_cannot_be_forged(self, kb: KnowledgeBase):
        """角色由框架注入，模型无法通过参数伪造。

        这是安全上的硬要求：权限参数绝不能交给模型决定。
        """
        registry = build_registry(kb)
        # 模型试图传 role=hr 提升权限
        result = registry.execute("search_docs", {"query": "绩效工资占比", "top_k": 5, "role": "hr"}, role="public")
        assert result.payload["role"] == "public"
        sources = {item["source"] for item in result.payload["results"]}
        assert "hr/compensation.md" not in sources

    def test_list_documents_filters_by_role(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        public = registry.execute("list_documents", {}, role="public").payload
        hr = registry.execute("list_documents", {}, role="hr").payload
        assert public["count"] < hr["count"]

    def test_rewrite_question_needs_history(self, kb: KnowledgeBase):
        registry = build_registry(kb)
        result = registry.execute(
            "rewrite_question",
            {"question": "那最多多少天", "history": ["年假能结转到明年吗"]},
        )
        assert result.ok
        payload = result.payload
        assert payload["changed"] is True
        assert "年假" in payload["rewritten"]


class TestRewriteRule:
    def test_no_history_returns_original(self):
        assert _rewrite("那最多多少天", []) == "那最多多少天"

    def test_no_coreference_returns_original(self):
        question = "年假可以结转到明年吗"
        assert _rewrite(question, ["年假有多少天"]) == question

    def test_coreference_is_resolved(self):
        rewritten = _rewrite("那最多能结转多少天", ["年假可以结转到明年吗"])
        assert "年假" in rewritten
        assert "那" not in rewritten

    def test_pronoun_is_resolved(self):
        rewritten = _rewrite("它的有效期是多久", ["密码有什么复杂度要求"])
        assert "密码" in rewritten
