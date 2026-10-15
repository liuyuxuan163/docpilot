"""LLM 层：Mock 引擎与 OpenAI 兼容网关，接口完全一致。

为什么一定要有 Mock 模式？
-------------------------
1. **面试现场不能赌网络和 API Key**：``python -m docpilot demo`` 必须永远能跑。
2. **测试必须确定性**：单元测试里不能出现"今天模型心情不好所以挂了"。
3. **成本可控**：改检索策略、改切块、跑评测时不该每次都花钱。

Mock 的定位是**"确定性参考策略"**而不是"假装回答"：
它不编造内容，而是按固定剧本调用真实工具，最后**用工具真实返回的片段**组织答案。
所以换成真实模型后，链路、工具、评测口径完全不变。

::

    user 提问
      └─ MockLLM 决策：改写问题？检索？还是作答？
            ├── 调 rewrite_question（多轮指代消解）
            ├── 调 search_docs（混合检索）
            └── 直接输出答案（带 [1][2] 引用标号）

切换真实模型只需环境变量，上层代码零改动::

    export DOCPILOT_PROVIDER=deepseek
    export DEEPSEEK_API_KEY=sk-xxxx
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .core import Config, LLMAuthError, LLMError, Message, ToolCall

__all__ = ["LLMResponse", "BaseLLM", "MockLLM", "OpenAICompatLLM", "build_llm", "estimate_tokens"]


def estimate_tokens(text: str) -> int:
    """粗略 token 估算（中英混排约 2.2 字符/token），仅用于本地成本展示。"""
    return max(1, int(len(text or "") / 2.2))


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_message(self) -> Message:
        return Message(role="assistant", content=self.content, tool_calls=list(self.tool_calls))


# --------------------------------------------------------------------------- #
# 抽象基类
# --------------------------------------------------------------------------- #


class BaseLLM:
    provider = "base"
    model = "unknown"

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        raise NotImplementedError

    # -- 共享工具 ------------------------------------------------------- #
    @staticmethod
    def count_tokens(messages: Sequence[Message], tools: Optional[List[Dict[str, Any]]] = None) -> int:
        blob = json.dumps(
            {"m": [m.to_dict() for m in messages], "t": tools or []}, ensure_ascii=False, default=str
        )
        return estimate_tokens(blob)

    @staticmethod
    def referenced_docs(messages: Sequence[Message]) -> List[str]:
        """统计历史里已经检索过的文档，用于判断"是否还缺证据"。"""
        docs: List[str] = []
        for message in messages:
            if message.role != "tool":
                continue
            data = _extract_json(message.content)
            if data and data.get("tool") == "search_docs":
                for item in data.get("results", []) or []:
                    source = str(item.get("source", ""))
                    if source and source not in docs:
                        docs.append(source)
        return docs


# --------------------------------------------------------------------------- #
# OpenAI 兼容实现（DeepSeek / OpenAI / vLLM / One-API 都能接）
# --------------------------------------------------------------------------- #


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: float = 60.0,
        provider: str = "openai_compat",
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMAuthError(
                "未找到 API Key。请设置 DEEPSEEK_API_KEY / OPENAI_API_KEY / DOCPILOT_API_KEY，"
                "或把 DOCPILOT_PROVIDER 设为 mock 以离线运行。"
            )
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise LLMError("需要安装 openai 包：pip install openai") from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    def complete(self, messages: Sequence[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_api() for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_output_tokens", self.max_output_tokens),
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        started = time.perf_counter()
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - 依赖网络
            name = type(exc).__name__.lower()
            if "auth" in name or "401" in str(exc):
                raise LLMAuthError(f"鉴权失败：{exc}") from exc
            raise LLMError(f"LLM 调用失败：{exc}") from exc
        latency = (time.perf_counter() - started) * 1000

        choice = response.choices[0]
        message = choice.message
        tool_calls: List[ToolCall] = []
        for call in getattr(message, "tool_calls", None) or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=call.id, name=call.function.name, arguments=args))

        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or self.count_tokens(messages, tools),
            completion_tokens=getattr(usage, "completion_tokens", 0) or estimate_tokens(message.content or ""),
            model=self.model,
            provider=self.provider,
            latency_ms=latency,
        )


# --------------------------------------------------------------------------- #
# Mock 实现：确定性"参考策略"
# --------------------------------------------------------------------------- #


class MockLLM(BaseLLM):
    """离线确定性引擎：按剧本调用真实工具，并用工具真实返回的片段组织答案。"""

    provider = "mock"

    # 需要改写的指代词：出现这些词说明问题依赖上文，必须先消解。
    # 必须与 tools.COREFERENCE_HINTS 保持一致 —— 这里决定"是否触发改写"，
    # 那里决定"怎么改写"，两处不一致会导致触发却不生效。
    _COREFERENCE_HINTS = ("它", "这个", "那个", "那", "上面", "刚才", "前面", "该", "此", "这些", "那些")
    # 明显的"不相关/超纲"提问特征 → 用于演示拒答
    _NEVER_MENTIONED = ("报销", "发票", "股票", "天气", "午饭", "比特币")

    def __init__(self, model: str = "mock-docpilot", min_score: float = 0.32) -> None:
        self.model = model
        self.min_score = float(min_score)
        self.call_count = 0

    # ------------------------------------------------------------------ #
    def complete(self, messages: Sequence[Message], tools: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> LLMResponse:
        started = time.perf_counter()
        self.call_count += 1
        available = _tool_names(tools)
        question = _first_user_question(messages)
        called = [c.name for m in messages for c in (m.tool_calls or [])]

        # ① 多轮指代消解：问题依赖上文且还没改写时，先改写
        if (
            "rewrite_question" in available
            and "rewrite_question" not in called
            and _needs_rewrite(question, self._COREFERENCE_HINTS)
        ):
            return self._respond(
                messages,
                content="问题里有指代词，先结合上文改写成一个可独立检索的问题。",
                tool_calls=[ToolCall(name="rewrite_question", arguments={"question": question})],
                started=started,
            )

        # ② 检索：还没检索过就检索
        if "search_docs" in available and "search_docs" not in called:
            query = _last_rewritten(messages) or question
            return self._respond(
                messages,
                content="先在知识库中检索相关文档片段。",
                tool_calls=[ToolCall(name="search_docs", arguments={"query": query, "top_k": 4})],
                started=started,
            )

        # ③ 作答（用检索回来的真实片段组织）
        return self._respond(messages, content=self._answer(messages, question), tool_calls=[], started=started)

    # ------------------------------------------------------------------ #
    def _answer(self, messages: Sequence[Message], question: str) -> str:
        payload = _last_search_payload(messages)
        hits = list(payload.get("results", []) or [])
        if not hits:
            return "知识库中没有检索到与问题相关的内容，因此无法回答。"

        # 判定"有没有依据"直接用工具算好的结论，而不是在这里再猜一遍阈值。
        # 工具层的 has_evidence = (覆盖率 ≥ 0.35) 或 (top-1 领先间隔 ≥ 0.08)：
        #   · 覆盖率管"确实对上了"；
        #   · 领先间隔管"对上的那一段是突出的、不是所有段落都沾一点"。
        # 早期版本直接拿混合分 score 做 `< 0.3` 判定，出现了最尴尬的失败：
        # **引用里明明有正确片段，回答却说"没有依据"**（量纲不可比）。
        best_relevance = float(payload.get("best_relevance", 0.0) or 0.0)
        margin = float(payload.get("margin", 0.0) or 0.0)
        restricted = list(payload.get("restricted_sources", []) or [])

        # 权限优先：问题指向了当前角色无权访问的文档时，明确告知而不是拿别的文档凑答案。
        if restricted:
            names = "、".join(f"`{s}`" for s in restricted[:3])
            return (
                f"这个问题涉及受限文档（{names}），你当前角色（{payload.get('role', 'public')}）"
                "没有访问权限。请联系对应文档的负责人申请授权，我不能用其它文档的内容代替回答。"
            )

        if not payload.get("has_evidence", False):
            if best_relevance <= 0.0:
                return f"知识库中没有任何文档涉及「{question}」，我无法回答这个问题。"
            return (
                f"检索结果与问题的实义字符覆盖率仅 {best_relevance:.2f}，且没有明显突出的片段"
                f"（领先间隔 {margin:.2f}），说明问题在知识库中没有对应依据，因此我不做推测。"
                "建议补充相关文档，或把问题问得更具体一些。"
            )

        lines: List[str] = ["根据知识库中的以下内容：", ""]
        for index, hit in enumerate(hits, 1):
            text = _condense(str(hit.get("text", "")))
            lines.append(f"[{index}] {text}（来源：{hit.get('citation', hit.get('source', ''))}）")
            lines.append("")
        lines.append("综合以上片段可以回答：" + _condense(str(hits[0].get("text", "")), 160))
        lines.append("")
        lines.append("说明：以上回答仅基于检索到的文档片段，未做任何推测；如需更完整的口径请查阅原文。")
        return "\n".join(lines)

    @staticmethod
    def _respond(
        messages: Sequence[Message],
        content: str,
        tool_calls: List[ToolCall],
        started: float,
    ) -> LLMResponse:
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=BaseLLM.count_tokens(messages),
            completion_tokens=estimate_tokens(content) + 20 * len(tool_calls),
            model="mock-docpilot",
            provider="mock",
            latency_ms=(time.perf_counter() - started) * 1000 + 8.0,
        )


# --------------------------------------------------------------------------- #
# 装配
# --------------------------------------------------------------------------- #


def build_llm(config: Config) -> BaseLLM:
    """按配置装配 LLM。**未配置 Key 时自动降级为 Mock**，而不是抛错。

    这是"新同学 clone 完三分钟能跑起来"的关键体验设计：
    环境不全的时候，系统应该可用并明确告诉你它降级了，而不是直接起不来。
    """
    provider = config.provider
    if provider == "mock":
        return MockLLM(min_score=float(config.get("agent.min_retrieval_score", 0.32)))

    api_key = config.api_key()
    if not api_key:
        return MockLLM(min_score=float(config.get("agent.min_retrieval_score", 0.32)))

    return OpenAICompatLLM(
        model=config.model,
        api_key=api_key,
        base_url=config.base_url(),
        timeout_s=float(config.get("llm.timeout_s", 60)),
        provider=provider,
        temperature=float(config.get("llm.temperature", 0.1)),
        max_output_tokens=int(config.get("llm.max_output_tokens", 1024)),
    )


# --------------------------------------------------------------------------- #
# 内部工具
# --------------------------------------------------------------------------- #

_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """从工具结果里抽出 JSON，**并容忍截断**。

    为什么必须容忍截断：工具结果会被框架按字符数截断后才喂给模型
    （见 :class:`~docpilot.tools.ToolResult.summary`），截断点很可能落在
    JSON 中间。如果不做修复，解析就失败，下游会误判成"检索没有结果" ——
    本项目真实踩过这个坑：**引用里明明有正确片段，回答却说"没有检索到"**。

    修复策略：先尝试直接解析；失败则做括号配平补全（补 ``]``/``}``、
    去掉结尾不完整的键值对）后重试。
    """
    if not text:
        return None
    candidates: List[str] = []
    fence = _JSON_RE.search(text)
    if fence:
        candidates.append(fence.group(1))
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
        else:
            # 没有走到 depth==0：说明是截断的 JSON，交给修复器
            candidates.append(_repair_json(text[start:]))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            repaired = _repair_json(candidate)
            try:
                data = json.loads(repaired)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
    return None


def _repair_json(text: str) -> str:
    """把截断的 JSON 补成合法 JSON。

    两步：
    1. **括号配平**：补上缺失的 ``]`` / ``}``；
    2. **去掉结尾不完整的成员**：截断点常落在 ``"text": "内容`` 这种
       键值对中间，光补括号会得到 ``{"text": "内容": null}`` 这种非法结构。
       做法是回退到最后一个完整的 ``,`` 或 ``{`` / ``[`` 之后重新配平。

    这个函数存在的唯一理由：工具结果会被按字符数截断后才喂给模型，
    截断点很可能落在 JSON 中间。不修复的话下游会误判成"检索没有结果"。
    """
    balanced = _balance_json(text)
    try:
        json.loads(balanced)
        return balanced
    except json.JSONDecodeError:
        pass

    # 从后往前找最后一个"完整的成员边界"，把它之后的内容整体丢弃
    for index in range(len(text) - 1, -1, -1):
        if text[index] in ",{[":
            candidate = _balance_json(text[:index])
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue
    return "{}"


def _balance_json(text: str) -> str:
    """只做括号配平：补全未闭合的字符串与括号。"""
    out: List[str] = []
    stack: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string:
            out.append(ch)
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
            out.append(ch)
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            out.append(ch)
        else:
            out.append(ch)

    repaired = "".join(out).rstrip()
    if in_string:
        repaired += '"'
    repaired = re.sub(r"[,:\s]+$", "", repaired)
    while stack:
        repaired += stack.pop()
    return repaired


def _tool_names(tools: Optional[Sequence[Dict[str, Any]]]) -> set[str]:
    names: set[str] = set()
    for item in tools or []:
        fn = item.get("function") if isinstance(item, dict) else None
        if isinstance(fn, dict) and fn.get("name"):
            names.add(str(fn["name"]))
    return names


def _first_user_question(messages: Sequence[Message]) -> str:
    """取出本次真正要回答的问题。

    **必须优先找 ``问题：`` 标记，而不是"第一条 user 消息"。**

    因为 ReAct 会把对话历史作为一个 user 消息先塞进去，历史里那句
    "年假可以结转到明年吗？" 就成了第一条 user 消息。若直接取第一条，
    本轮真正的问题"那最多能结转多少天？"会被彻底忽略 ——
    表现是：多轮追问不会触发改写、检索用错问句。本项目实测踩过这个坑。

    所以顺序是：先扫所有消息里的 ``问题：`` 标记（那是框架自己写的，最可信），
    找不到再退化为"第一条非历史的 user 消息"。
    """
    markers = ("问题：", "问题:")
    for message in messages:
        if message.role != "user" or not message.content:
            continue
        for marker in markers:
            index = message.content.find(marker)
            if index >= 0:
                tail = message.content[index + len(marker) :].strip().splitlines()
                if tail and tail[0].strip():
                    return tail[0].strip()
    for message in messages:
        if message.role == "user" and message.content and not message.content.startswith("##"):
            return message.content.strip().splitlines()[0] if message.content.strip() else ""
    return ""


def _needs_rewrite(question: str, hints: Sequence[str]) -> bool:
    """问题是否依赖上文（含指代词）。

    注意：**不要按长度一刀切**。
    早期实现里加了 "长度 > 40 就判定不需要改写"，结果
    "那最多能结转多少天？有上限吗？" 这种长一点的追问直接被跳过改写，
    检索用一个没有主题词的问句去打库，必然检索偏 —— 实测发现的缺口。
    是否依赖上文只取决于"有没有指代词"，与长度无关。
    """
    if not question:
        return False
    if question.startswith(("它", "这个", "那个", "那", "他", "她")):
        return True
    return any(hint in question for hint in hints)


def _last_rewritten(messages: Sequence[Message]) -> Optional[str]:
    for message in reversed(list(messages)):
        if message.role != "tool":
            continue
        data = _extract_json(message.content)
        if data and data.get("tool") == "rewrite_question" and data.get("rewritten"):
            return str(data["rewritten"])
    return None


def _last_search_results(messages: Sequence[Message]) -> List[Dict[str, Any]]:
    return list(_last_search_payload(messages).get("results", []) or [])


def _last_search_payload(messages: Sequence[Message]) -> Dict[str, Any]:
    """取最近一次 search_docs 的完整 payload（含 has_evidence 等判定结论）。"""
    for message in reversed(list(messages)):
        if message.role != "tool":
            continue
        data = _extract_json(message.content)
        if data and data.get("tool") == "search_docs":
            return data
    return {}


def _condense(text: str, limit: int = 220) -> str:
    """把片段压成一句话长度的摘要（去掉表格符号与多余空白）。"""
    cleaned = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    return cleaned if len(cleaned) <= limit else cleaned[:limit] + "…"
