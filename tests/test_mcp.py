"""测试：MCP 服务端（协议握手 / 工具列表 / 工具调用 / 提权防护）。

MCP 这一层最容易出两类事故，所以测试也围绕它们：

1. **协议不合规** —— 通知被回复、响应里出现裸换行、未知方法让会话崩掉。
   这类问题的症状是"宿主连不上"，从现象很难反推到某一行代码，
   所以这里逐条锁住：通知不回复、响应单行、坏消息不中断循环。
2. **权限被提权** —— 客户端在 ``arguments`` 里塞 ``role: "hr"``。
   这是本项目"安全参数不交给模型"（见 tools.py 文件头）在 MCP 场景下的延伸：
   调用方从"我们的 Agent"变成了"宿主的模型"，攻击面更大。

刻意**不起子进程、不联网络**：``serve()`` 的流是可注入的，
所以整条握手可以在同一个进程里用 ``StringIO`` 跑完。
"""

from __future__ import annotations

import json
from io import StringIO

import pytest

from docpilot.mcp import (
    EXPOSED_TOOLS,
    PROTOCOL_VERSION,
    McpServer,
    encode,
    resolve_role,
    serve,
)
from docpilot.rag import KnowledgeBase


@pytest.fixture()
def server(kb: KnowledgeBase) -> McpServer:
    """public 角色的服务端（默认就是最有价值的那一档：它看不到薪酬文档）。"""
    return McpServer(kb, role="public")


def _initialize(server: McpServer, version: str = PROTOCOL_VERSION) -> dict:
    response = server.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": version, "capabilities": {}, "clientInfo": {"name": "test"}},
        }
    )
    assert response is not None
    return response["result"]


def _call(server: McpServer, name: str, arguments: dict, msg_id: int = 2) -> dict:
    response = server.handle(
        {"jsonrpc": "2.0", "id": msg_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    )
    assert response is not None
    return response


# --------------------------------------------------------------------------- #
# 握手
# --------------------------------------------------------------------------- #


class TestHandshake:
    def test_initialize_declares_tools_capability(self, server: McpServer):
        result = _initialize(server)
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert result["capabilities"]["tools"] == {"listChanged": False}
        assert result["serverInfo"]["name"] == "docpilot"
        # instructions 是"跨进程的 SYSTEM_PROMPT"：宿主靠它知道该拒答
        assert "has_evidence" in result["instructions"]

    def test_initialize_accepts_older_supported_version(self, server: McpServer):
        result = _initialize(server, version="2024-11-05")
        assert result["protocolVersion"] == "2024-11-05"

    def test_initialize_falls_back_for_unknown_version(self, server: McpServer):
        """规范：不支持客户端请求的版本时，回自己支持的版本。"""
        result = _initialize(server, version="1999-01-01")
        assert result["protocolVersion"] == PROTOCOL_VERSION

    def test_notifications_get_no_response(self, server: McpServer):
        """通知**不得**回复 —— 回了会让宿主把响应错配到下一条请求上。"""
        server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert server.initialized is True
        # 未实现的通知也一样：不回复，也不报错
        assert server.handle({"jsonrpc": "2.0", "method": "notifications/cancelled"}) is None

    def test_ping(self, server: McpServer):
        assert server.handle({"jsonrpc": "2.0", "id": 9, "method": "ping"})["result"] == {}


# --------------------------------------------------------------------------- #
# 工具列表
# --------------------------------------------------------------------------- #


class TestToolsList:
    def test_exposes_exactly_the_whitelist(self, server: McpServer):
        tools = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})["result"]["tools"]
        assert [tool["name"] for tool in tools] == list(EXPOSED_TOOLS)
        # 刻意不暴露：改写是宿主侧职责，ask 会与宿主的 Agent 抢决策
        assert "rewrite_question" not in [tool["name"] for tool in tools]
        assert "ask" not in [tool["name"] for tool in tools]

    def test_search_schema_is_reusable_and_has_no_role_parameter(self, server: McpServer):
        tools = {tool["name"]: tool for tool in server.handle(
            {"jsonrpc": "2.0", "id": 3, "method": "tools/list"}
        )["result"]["tools"]}
        schema = tools["search_docs"]["inputSchema"]
        assert schema["required"] == ["query"]
        # 安全设计的**结构性保证**：schema 里根本没有 role，客户端无从传起
        assert "role" not in schema["properties"]
        assert tools["search_docs"]["annotations"]["readOnlyHint"] is True


# --------------------------------------------------------------------------- #
# 工具调用
# --------------------------------------------------------------------------- #


class TestToolsCall:
    def test_search_returns_text_and_structured_content(self, server: McpServer):
        result = _call(server, "search_docs", {"query": "年假没休完可以结转到明年吗？"})["result"]
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        # 文本块必须是合法 JSON（宿主可能直接解析它）
        payload = json.loads(result["content"][0]["text"])
        assert payload["has_evidence"] is True
        assert result["structuredContent"]["count"] >= 1
        assert result["structuredContent"]["results"][0]["citation"]

    def test_list_documents_respects_role(self, server: McpServer):
        result = _call(server, "list_documents", {})["result"]["structuredContent"]
        sources = {doc["source"] for doc in result["documents"]}
        assert "hr/leave-policy.md" in sources
        assert "hr/compensation.md" not in sources, "public 角色不应看到薪酬文档"

    def test_client_cannot_escalate_role(self, server: McpServer):
        """核心安全测试：在 arguments 里塞 role="hr" 不能提权。"""
        result = _call(
            server,
            "search_docs",
            {"query": "P4 职级的绩效工资占比是多少？", "role": "hr"},
        )["result"]["structuredContent"]
        # 权限由服务端注入并**覆盖**客户端传值
        assert result["role"] == "public"
        assert result["has_evidence"] is False
        assert "hr/compensation.md" in result["restricted_sources"]
        assert all("compensation" not in item["source"] for item in result["results"])

    def test_hr_server_can_read_the_same_document(self, kb: KnowledgeBase):
        """对照组：同一个问题、同一个客户端，只换服务端角色就能看到。"""
        hr_server = McpServer(kb, role="hr")
        result = _call(hr_server, "search_docs", {"query": "P4 职级的绩效工资占比是多少？"})["result"]
        assert result["structuredContent"]["has_evidence"] is True
        assert any("compensation" in item["source"] for item in result["structuredContent"]["results"])

    def test_schema_violation_is_a_tool_error_not_a_crash(self, server: McpServer):
        """参数越界：返回 isError，而不是抛异常把会话打断。"""
        result = _call(server, "search_docs", {"query": "年假", "top_k": 999})["result"]
        assert result["isError"] is True
        assert "上限" in result["content"][0]["text"]

    def test_unknown_tool_is_invalid_params(self, server: McpServer):
        response = _call(server, "rm_rf", {})
        assert response["error"]["code"] == -32602
        assert "未知工具" in response["error"]["message"]

    def test_unknown_method_is_method_not_found(self, server: McpServer):
        response = server.handle({"jsonrpc": "2.0", "id": 7, "method": "resources/list"})
        assert response["error"]["code"] == -32601


# --------------------------------------------------------------------------- #
# 传输层
# --------------------------------------------------------------------------- #


class TestStdioTransport:
    def test_response_is_single_line(self, server: McpServer):
        """裸换行会破坏换行分帧，宿主直接断连。"""
        response = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert "\n" not in encode(response)

    def test_encoded_json_keeps_chinese_readable(self):
        """中文原样输出、不转义成 \\uXXXX：省字节，排查 stderr 时也能直接看懂。"""
        assert encode({"text": "知识库"}) == '{"text":"知识库"}'

    def test_utf8_bom_is_tolerated(self, kb: KnowledgeBase):
        """手工造输入文件常带 BOM；不能因此让首条 initialize 失败。

        这是**实测撞到过**的问题：带 BOM 的首行会以 -32700 收场，
        表现为"宿主连不上"，而根因只在第一个字节里。
        """
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping"}, ensure_ascii=False)
        out = StringIO()
        assert serve(kb, role="public", instream=StringIO("\ufeff" + payload + "\n"), outstream=out) == 0
        assert json.loads(out.getvalue())["result"] == {}

    def test_serve_handles_full_session_and_bad_json(self, kb: KnowledgeBase):
        lines = [
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": PROTOCOL_VERSION}},
                ensure_ascii=False,
            ),
            "{ 这不是合法 JSON",
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json.dumps(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                 "params": {"name": "search_docs", "arguments": {"query": "密码多少位"}}},
                ensure_ascii=False,
            ),
            "",
        ]
        out = StringIO()
        assert serve(kb, role="public", instream=StringIO("\n".join(lines)), outstream=out) == 0

        responses = [json.loads(line) for line in out.getvalue().splitlines()]
        # 3 条有效请求 → 3 条响应；通知不回复；坏 JSON 不中断后续请求
        assert [item.get("id") for item in responses] == [1, None, 2]
        assert responses[1]["error"]["code"] == -32700
        assert responses[2]["result"]["isError"] is False


# --------------------------------------------------------------------------- #
# 角色解析
# --------------------------------------------------------------------------- #


class TestResolveRole:
    def test_priority_cli_over_env_over_config(self, config, monkeypatch):
        assert resolve_role(config, "finance") == "finance"
        monkeypatch.setenv("DOCPILOT_MCP_ROLE", "legal")
        assert resolve_role(config) == "legal"
        assert resolve_role(config, "finance") == "finance", "--role 必须压过环境变量"
        monkeypatch.delenv("DOCPILOT_MCP_ROLE")
        assert resolve_role(config) == "public", "都没给时回配置里的默认角色"

    def test_empty_value_does_not_produce_empty_role(self, config, monkeypatch):
        monkeypatch.setenv("DOCPILOT_MCP_ROLE", "   ")
        assert resolve_role(config) == "public"
