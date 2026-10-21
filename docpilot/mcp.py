"""MCP 服务端：把 DocPilot 的检索能力以 stdio 协议对外暴露（零依赖手写）。

为什么值得做
------------
CLI / Web / Agent 这三个"前端"只能在本项目内部用。MCP 把 ``search_docs`` 变成
**任何宿主都能挂载的能力**：Claude Desktop / Cursor / Cline 配一行 JSON 就能用
同一份知识库，而且引用、拒答、权限隔离的行为**完全一致** ——
因为这里复用的是同一个 :class:`~docpilot.tools.ToolRegistry`，不是另写一套。

三个刻意的设计决定
------------------
1. **手写协议，不引官方 SDK。** MCP 的 stdio 传输就是"一行一条 JSON-RPC 2.0"，
   本服务只需要 ``initialize`` / ``notifications/initialized`` / ``tools/list`` /
   ``tools/call`` / ``ping`` 五个方法，也就两百来行。好处是与项目"零核心依赖"
   一致（clone 完就能挂）；代价是规范演进要自己跟 —— 所以每个字段旁边都标了
   它属于哪一步，升级时逐条核对即可。参照的规范版本：**2025-06-18**。
2. **角色在服务端绑定，绝不作为工具参数。** 这是项目里"安全参数不交给模型"
   （见 tools.py 文件头）的延伸：MCP 的调用方是**宿主里的模型**，把 role 暴露成
   参数等于允许客户端自称 HR。角色来自 ``--role`` / ``DOCPILOT_MCP_ROLE`` /
   配置 ``mcp.role``；而 ``needs_role`` 会在参数校验**之后**覆盖客户端传来的
   同名字段 —— ``tests/test_mcp.py`` 里有一条专门的提权测试锁住这一点。
3. **stdout 是协议通道，日志只能进 stderr。** 一句 ``print()`` 就会污染
   JSON-RPC 流，让宿主直接断连 —— 这是这类服务端最常见的事故。

安全边界（如实写出来，而不是假装没有）
--------------------------------------
* 检索结果会**原样进入宿主模型的上下文**，所以文档正文里的恶意指令不再被
  ``agent.SYSTEM_PROMPT`` 约束。MCP 侧能做的只有把结果标成数据（返回值里
  只有东西、没有指令），以及不暴露任何写操作（``annotations.readOnlyHint=true``）。
* 一个进程绑定一个角色，多角色就多进程；这是为了让"权限"是部署属性而不是
  请求属性。接真实权限中心时，把 :func:`_resolve_role` 换掉即可，协议层不用改。
"""

from __future__ import annotations

import io
import json
import os
import sys
from typing import Any, Dict, IO, List, Optional, Sequence

from . import __version__
from .core import Config, ToolNotFound
from .rag import KnowledgeBase
from .tools import ToolRegistry, build_registry

__all__ = [
    "McpServer",
    "serve",
    "main",
    "resolve_role",
    "encode",
    "EXPOSED_TOOLS",
    "PROTOCOL_VERSION",
    "SUPPORTED_VERSIONS",
]

# --------------------------------------------------------------------------- #
# 协议常量
# --------------------------------------------------------------------------- #

#: 本服务实现的规范版本。客户端要求其它版本时，回退到这里（规范要求：
#   server 若不支持客户端请求的版本，应回复自己支持的版本，由客户端决定是否继续）。
PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_VERSIONS: Sequence[str] = ("2025-06-18", "2025-03-26", "2024-11-05")

SERVER_NAME = "docpilot"
SERVER_VERSION = __version__

#: 暴露给宿主的工具白名单。
#:
#: 刻意**只有两个**：
#: * ``rewrite_question`` 不暴露 —— 指代消解是宿主那一侧的事（宿主本来就拿着
#:   完整对话），拿它当工具只会让宿主多绕一步；
#: * 不暴露 ``ask``（整条 ReAct 链路）—— 宿主自己就在跑 Agent，再塞一个自带
#:   循环的 Agent 会互相抢决策，属于"双 Agent 反模式"。
EXPOSED_TOOLS: Sequence[str] = ("search_docs", "list_documents")

#: initialize 返回的 instructions。这相当于"跨进程的 SYSTEM_PROMPT"：
#: 把项目的三条纪律（可溯源 / 没依据就拒答 / 无权限不要说）交给宿主。
INSTRUCTIONS = (
    "DocPilot 提供企业知识库检索（只检索，不代答）。"
    "调用 search_docs 取回片段后：把每条的 `citation` 写进答案作为引用；"
    "`has_evidence=false` 表示知识库里没有可用依据，应当明确拒答而**不要**用常识补全；"
    "`restricted_sources` 非空表示问题指向了当前角色无权访问的文档，"
    "应当告知无权限，而**不要**拿别的文档代替回答。"
)


class _InvalidParams(Exception):
    """参数不合法 → 映射到 JSON-RPC ``-32602``（客户端错误，不是工具执行失败）。"""


# --------------------------------------------------------------------------- #
# 服务端
# --------------------------------------------------------------------------- #


class McpServer:
    """一个角色、一份知识库的 MCP 服务端。

    刻意不持有"已初始化"之外的状态：MCP 是有状态的会话协议，但我们的工具
    全是只读且无会话状态，这样即使宿主并发调用也不会互相干扰。
    """

    def __init__(
        self,
        kb: KnowledgeBase,
        role: str = "public",
        registry: Optional[ToolRegistry] = None,
        exposed: Sequence[str] = EXPOSED_TOOLS,
    ) -> None:
        self.kb = kb
        self.role = (role or "public").strip().lower() or "public"
        # 复用 Agent 用的同一份注册表：schema 校验、权限注入、证据判定
        # 全部走同一条路径 —— 这是"同一个内核、不同前端"的关键，而不是再实现一遍。
        self.registry = registry or build_registry(kb)
        self.exposed = tuple(exposed)
        self.initialized = False
        self.client_info: Dict[str, Any] = {}
        self.protocol_version = PROTOCOL_VERSION

    # -- 消息分发 ------------------------------------------------------- #
    def handle(self, message: Any) -> Optional[Dict[str, Any]]:
        """处理一条 JSON-RPC 消息。

        Returns:
            响应字典；通知（没有 ``id``）返回 ``None`` —— 规范规定通知**不得**回复。
        """
        if not isinstance(message, dict):
            return _error_response(None, -32600, "请求必须是 JSON 对象")

        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message

        if not isinstance(method, str):
            return None if is_notification else _error_response(msg_id, -32600, "缺少 method 字段")

        params = message.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return None if is_notification else _error_response(msg_id, -32602, "params 必须是 JSON 对象")

        try:
            if method == "initialize":
                result: Dict[str, Any] = self._initialize(params)
            elif method == "notifications/initialized":
                # 客户端确认握手完成。这是一个**通知**，不能有响应。
                self.initialized = True
                return None
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": self._tools()}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                if is_notification:
                    return None
                return _error_response(msg_id, -32601, f"未实现的方法：{method}")
        except _InvalidParams as exc:
            if is_notification:
                return None
            return _error_response(msg_id, -32602, str(exc))
        except Exception as exc:  # noqa: BLE001 - 协议层必须兜住一切，否则宿主直接断连
            if is_notification:
                return None
            return _error_response(msg_id, -32603, f"{type(exc).__name__}: {exc}")

        if is_notification:
            return None
        return {"jsonrpc": "2.0", "id": msg_id, "result": result}

    # -- initialize ----------------------------------------------------- #
    def _initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """握手：协商协议版本、声明能力、返回 serverInfo 与 instructions。"""
        requested = str(params.get("protocolVersion") or "")
        # 规范：不支持请求版本时，回自己支持的版本，由客户端决定是否继续。
        version = requested if requested in SUPPORTED_VERSIONS else PROTOCOL_VERSION
        self.protocol_version = version
        self.client_info = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        return {
            "protocolVersion": version,
            # 只声明 tools。不声明 resources / prompts，是因为没实现 ——
            # 声明了却不响应，会让客户端卡在列表请求上。
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS,
        }

    # -- tools/list ----------------------------------------------------- #
    def _tools(self) -> List[Dict[str, Any]]:
        """把注册表里的工具描述翻译成 MCP 的 ``Tool`` 结构。"""
        tools: List[Dict[str, Any]] = []
        for name in self.exposed:
            try:
                tool = self.registry.get(name)
            except ToolNotFound:  # pragma: no cover - 白名单里写了不存在的工具
                continue
            tools.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    # 复用同一份 JSON Schema：Agent 用它做 function calling，
                    # 这里直接作为 MCP 的 inputSchema —— 两个前端不可能漂移。
                    "inputSchema": dict(tool.schema or {"type": "object", "properties": {}}),
                    # readOnlyHint 是给宿主的提示：不写数据、无副作用，可以免确认。
                    # 我们这个服务全是只读工具，所以三条都成立。
                    "annotations": {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False},
                }
            )
        return tools

    # -- tools/call ----------------------------------------------------- #
    def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具，返回 MCP 的 ``CallToolResult``。"""
        name = params.get("name")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(name, str) or name not in self.exposed:
            raise _InvalidParams(f"未知工具：{name!r}（可用：{', '.join(self.exposed)}）")
        if not isinstance(arguments, dict):
            raise _InvalidParams("arguments 必须是 JSON 对象")

        # role 由**这里**注入（needs_role=True），客户端即使传了 role 也会被覆盖。
        result = self.registry.execute(name, arguments, role=self.role)

        if not result.ok:
            # 工具执行失败属于**结果**而不是协议错误：规范用 isError 标记，
            # 这样宿主模型能看到失败原因，自己决定换问法还是放弃。
            return {
                "content": [{"type": "text", "text": result.error or "工具执行失败"}],
                "isError": True,
            }

        payload = result.payload
        return {
            # 文本块给模型读（沿用 Agent 的摘要逻辑：缩字段而不是切字符串，见 tools.py），
            "content": [{"type": "text", "text": result.summary()}],
            # 结构化块给宿主程序用；规范建议同时给文本块以兼容老客户端。
            "structuredContent": payload if isinstance(payload, dict) else {"value": payload},
            "isError": False,
        }


# --------------------------------------------------------------------------- #
# 传输层：stdio（一行一条 JSON-RPC）
# --------------------------------------------------------------------------- #


def encode(response: Dict[str, Any]) -> str:
    """把响应编码成**单行** JSON。

    MCP 的 stdio 传输用换行分帧，消息里不能出现裸换行 ——
    ``json.dumps`` 会把字符串里的 ``\\n`` 转义成两个字符，所以这里是安全的。
    ``ensure_ascii=False`` 让中文以 UTF-8 原样输出（可读性好，也省字节）。
    """
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def serve(
    kb: KnowledgeBase,
    role: str = "public",
    instream: Optional[IO[str]] = None,
    outstream: Optional[IO[str]] = None,
) -> int:
    """进入 stdio 消息循环，直到输入流结束（宿主关闭管道）。

    ``instream`` / ``outstream`` 可注入，是为了让测试能在**同一个进程里**
    用 ``StringIO`` 跑完整握手 —— 不需要起子进程、不需要网络，
    这也是本项目"离线确定性可测"风格的延续。
    """
    server = McpServer(kb, role=role)

    if instream is None or outstream is None:
        # Windows 上 sys.stdin/stdout 默认按 GBK（cp936）编解码，而 MCP 规定
        # UTF-8。这里显式包一层：否则中文参数会解码失败，表现成"协议不通"，
        # 排查起来非常费劲（本项目在 Web 接口上踩过一模一样的坑）。
        instream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")
        outstream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)

    for raw in instream:
        # 容忍 UTF-8 BOM。规范里没有 BOM，但**手工造的联调文件几乎一定会带**
        # （Windows 上用 PowerShell 的 `-Encoding utf8` 写出来就带）——
        # 一条 BOM 会让首条 initialize 变成 -32700，现象是"宿主连不上"，
        # 而真正的原因藏在第一个字节里。本项目第一次手工联调就撞上了。
        line = raw.lstrip("\ufeff").strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            # 解析失败也要回错误并**继续读**：否则一条坏消息就让整个会话挂断。
            response: Optional[Dict[str, Any]] = _error_response(None, -32700, f"JSON 解析失败：{exc}")
        else:
            response = server.handle(message)
        if response is not None:
            outstream.write(encode(response) + "\n")
            outstream.flush()

    return 0


def resolve_role(config: Config, role: Optional[str] = None) -> str:
    """确定服务端角色：``--role`` > 环境变量 > 配置 ``mcp.role`` > public。

    优先级刻意让**命令行最高**：部署脚本可以显式钉死角色，而环境变量便于
    在容器里按实例注入（一个实例一个租户）。
    """
    candidate = role or os.environ.get("DOCPILOT_MCP_ROLE") or config.get("mcp.role") or "public"
    return str(candidate).strip().lower() or "public"


def main(config: Config, role: Optional[str] = None) -> int:
    """CLI 入口：构建索引 → 打印一行"人看的"提示（stderr）→ 进入 stdio 循环。"""
    resolved = resolve_role(config, role)
    kb = KnowledgeBase(config)
    stats = kb.build()

    # ⚠ 只能写 stderr。stdout 是协议通道，写一个字都会让宿主解析失败。
    visible = [doc["source"] for doc in kb.documents(role=resolved)]
    print(
        f"[docpilot-mcp] role={resolved} · {stats['documents']} 篇文档 / "
        f"{stats['chunks']} 个片段 · 可见 {len(visible)} 篇 · protocol={PROTOCOL_VERSION}",
        file=sys.stderr,
    )
    print(f"[docpilot-mcp] 工具：{', '.join(EXPOSED_TOOLS)}（只读）", file=sys.stderr)
    return serve(kb, role=resolved)


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #


def _error_response(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    """构造 JSON-RPC 错误响应（结构照规范：jsonrpc / id / error{code,message}）。"""
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}
