"""测试：Web 层（HTTP 处理函数、JSON 契约）。

**注意这里用的是 `invoke` 而不是真的起 HTTP 服务**：
测试目标是我们自己的处理逻辑与返回契约，不是标准库的 HTTP 实现。
直接调函数更快、更稳，也不会因为端口占用而随机失败。

（另外记一个真实踩过的坑，见下方 `test_utf8_request_body` 的注释：
用 PowerShell 的 `Invoke-RestMethod` 手工验证接口时，
**中文请求体必须显式转成 UTF-8 字节**，否则会被按本地代码页编码，
服务端收到的是一堆乱码，表现成"所有问题都被拒答"——
当时一度以为是服务端 bug，实际是客户端编码问题。）
"""

from __future__ import annotations

import json

import pytest

from docpilot.core import Config
from docpilot.rag import KnowledgeBase
from docpilot.web import STATIC_DIR, handle_ask, handle_documents


class TestHandleAsk:
    def test_answers_answerable_question(self, config: Config):
        result = handle_ask(config, {"question": "年假没休完可以结转到明年吗？", "role": "public"})
        assert "error" not in result
        assert result["refused"] is False
        assert result["citations"]

    def test_empty_question_returns_error(self, config: Config):
        assert "error" in handle_ask(config, {"question": "   "})
        assert "error" in handle_ask(config, {})

    def test_permission_isolation_over_http(self, config: Config):
        public = handle_ask(config, {"question": "P4 职级的绩效工资占比是多少？", "role": "public"})
        assert public["refused"] is True
        assert "权限" in public["text"]

        hr = handle_ask(config, {"question": "P4 职级的绩效工资占比是多少？", "role": "hr"})
        assert hr["refused"] is False
        assert any("compensation" in cite["label"] for cite in hr["citations"])

    def test_out_of_scope_is_refused(self, config: Config):
        assert handle_ask(config, {"question": "今天天气怎么样？", "role": "public"})["refused"] is True

    def test_result_is_json_serializable(self, config: Config):
        result = handle_ask(config, {"question": "年假可以结转到明年吗", "role": "public"})
        payload = json.dumps(result, ensure_ascii=False)
        assert "年假" in payload
        restored = json.loads(payload)
        assert restored["question"] == "年假可以结转到明年吗"

    def test_history_supports_multi_turn(self, config: Config):
        result = handle_ask(
            config,
            {"question": "那最多能结转多少天？", "role": "public", "history": ["年假可以结转到明年吗？"]},
        )
        assert result["refused"] is False
        assert "年假" in (result["rewritten_question"] or "")

    def test_utf8_request_body(self, config: Config):
        """中文请求体经 UTF-8 编解码后必须原样还原。

        这条测试锁的是上面注释里那个坑：客户端若按本地代码页编码，
        服务端拿到的会是乱码，表现为"什么问题都拒答"。
        """
        question = "年假没休完可以结转到明年吗？"
        raw = json.dumps({"question": question, "role": "public"}, ensure_ascii=False).encode("utf-8")
        decoded = json.loads(raw.decode("utf-8"))
        assert decoded["question"] == question
        result = handle_ask(config, decoded)
        assert result["refused"] is False


class TestHandleDocuments:
    def test_public_sees_fewer_documents(self, config: Config):
        public = handle_documents(config, "public")
        hr = handle_documents(config, "hr")
        assert public["role"] == "public"
        assert len(public["documents"]) < len(hr["documents"])
        assert "stats" in public

    def test_restricted_document_is_hidden_from_public(self, config: Config):
        sources = {doc["source"] for doc in handle_documents(config, "public")["documents"]}
        assert "hr/compensation.md" not in sources
        assert "legal/conduct-policy.md" not in sources


class TestStaticAssets:
    def test_index_html_exists_and_is_real_page(self):
        index = STATIC_DIR / "index.html"
        assert index.exists(), "缺少 Web 页面"
        content = index.read_text(encoding="utf-8")
        assert "DocPilot" in content
        assert "/api/ask" in content
        assert "/api/documents" in content

    def test_page_has_no_obvious_broken_quote_pairs(self):
        """锁一个真实踩过的坑：JS 字符串里用中文引号 ``"..."`` 会提前闭合字符串。

        HTML 里有大量中文，写 `"这本身就是"没有依据"的信号"` 这类字符串
        会让浏览器报语法错误、整页失去交互。
        """
        content = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        # 全角引号不应出现在 JS 单引号字符串内部造成提前闭合
        assert '"没有依据"' not in content
