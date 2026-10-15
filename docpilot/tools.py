"""工具层：注册表 + JSON Schema 入参校验 + 内置工具。

一个"通用 Agent"看起来业务无关，但**工具协议必须严格**，否则换任何业务都会出问题：

* 每个工具声明 schema → 直接喂给 LLM 做 function calling，同时用于**入参校验**；
* 模型给出越界参数时**在本地拒绝**，而不是让错误参数流到下游（省钱、也更容易定位）；
* 统一返回 :class:`ToolResult`，失败也返回结果对象而不抛异常 ——
  一个工具挂掉不该让整轮对话崩掉；
* 权限（``role``）由**工具自己从上下文里取**，而不是让模型传参 ——
  安全参数绝不能交给模型决定。
"""

from __future__ import annotations

import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .core import ToolDenied, ToolError, ToolNotFound
from .rag import KnowledgeBase

__all__ = ["ToolResult", "Tool", "ToolRegistry", "build_registry", "validate_args"]


# --------------------------------------------------------------------------- #
# 入参校验（极简 JSON Schema 子集：type / required / default / min / max / enum）
# --------------------------------------------------------------------------- #

_TYPE_MAP: Dict[str, tuple] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
}


def validate_args(schema: Dict[str, Any], args: Dict[str, Any], where: str = "args") -> Dict[str, Any]:
    """按 schema 校验并补全默认值。

    **宁可在这里报错，也不要把错误参数放进去** —— 这是最省钱的拦截点，
    比让错误请求打到检索层或模型层再失败便宜得多。
    """
    if not schema:
        return dict(args or {})
    props: Dict[str, Any] = schema.get("properties", {}) or {}
    required: Sequence[str] = schema.get("required", []) or []
    out: Dict[str, Any] = {}

    for key in required:
        if key not in (args or {}) and "default" not in props.get(key, {}):
            raise ToolError(f"{where}: 缺少必填参数 `{key}`")

    for key, spec in props.items():
        if key in (args or {}):
            value = args[key]
        elif "default" in spec:
            value = spec["default"]
        else:
            continue
        expected = spec.get("type")
        if expected:
            types = _TYPE_MAP.get(expected, ())
            # bool 是 int 的子类，必须显式排除，否则 True 会被当成合法 integer
            if types and (expected == "integer" and isinstance(value, bool) or not isinstance(value, types)):
                raise ToolError(f"{where}: 参数 `{key}` 期望 {expected}，实际 {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"{where}: 参数 `{key}` 只能是 {spec['enum']}，实际 {value!r}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                raise ToolError(f"{where}: 参数 `{key}` 小于下限 {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                raise ToolError(f"{where}: 参数 `{key}` 大于上限 {spec['maximum']}")
        out[key] = value

    for key, value in (args or {}).items():
        out.setdefault(key, value)
    return out


# --------------------------------------------------------------------------- #
# 工具定义与结果
# --------------------------------------------------------------------------- #


@dataclass
class ToolResult:
    name: str
    ok: bool
    payload: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0

    def summary(self, limit: int = 600) -> str:
        """渲染成给模型看的文本。

        **关键点：截断要"缩字段"，不要"切字符串"。**

        直接对序列化后的 JSON 做字符截断，会把结构切坏（少括号、字符串没闭合），
        下游解析失败后会误判成"检索没有结果" —— 本项目真实踩过这个坑：
        **引用里明明有正确片段，回答却说"没有检索到"**。

        正确做法是先把每个片段的长文本字段压到 ``limit`` 字符，
        让整体结构保持合法；只有在极个别情况下才退化为纯文本截断。
        """
        if not self.ok:
            return json.dumps({"tool": self.name, "error": self.error}, ensure_ascii=False)
        payload = _shrink_payload(self.payload, limit)
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
        if len(text) > limit * 6:
            # 兜底：即使缩过字段仍然过长，做结构感知截断并附上说明
            text = text[: limit * 6] + "…（结果过长已截断）"
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "error": self.error, "latency_ms": round(self.latency_ms, 2)}


def _shrink_payload(payload: Any, text_limit: int) -> Any:
    """递归压缩 payload 里的长文本字段，保持结构完整。"""
    if isinstance(payload, str):
        return payload if len(payload) <= text_limit else payload[:text_limit] + "…"
    if isinstance(payload, dict):
        return {key: _shrink_payload(value, text_limit) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_shrink_payload(item, text_limit) for item in payload]
    return payload


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    schema: Dict[str, Any] = field(default_factory=dict)
    needs_role: bool = False   # True 表示由注册表注入当前用户角色（安全参数不交给模型）

    def openai_schema(self) -> Dict[str, Any]:
        params = dict(self.schema or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": params}}


class ToolRegistry:
    """工具注册表：注册、校验、执行。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self.call_counts: Dict[str, int] = {}

    def __bool__(self) -> bool:
        """注册表恒为真。

        必须显式定义：不定义时 Python 会退回 ``__len__``，"空注册表"会被判为假，
        ``registry or default`` 这类写法就会静默丢弃传入的注册表。
        """
        return True

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    # -- 注册 ----------------------------------------------------------- #
    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str = "",
        schema: Optional[Dict[str, Any]] = None,
        needs_role: bool = False,
    ) -> Tool:
        tool = Tool(
            name=name,
            description=description or (inspect.getdoc(handler) or name).strip().split("\n")[0],
            handler=handler,
            schema=schema or _infer_schema(handler),
            needs_role=needs_role,
        )
        self._tools[name] = tool  # 重复注册时覆盖，保证幂等
        return tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolNotFound(f"未注册的工具：{name}（可用：{', '.join(self.names)}）")
        return self._tools[name]

    def schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].openai_schema() for name in self.names]

    # -- 执行 ----------------------------------------------------------- #
    def execute(self, name: str, arguments: Dict[str, Any], role: str = "public") -> ToolResult:
        started = time.perf_counter()
        try:
            tool = self.get(name)
            normalized = validate_args(tool.schema, arguments or {}, where=f"tool:{name}")
            if tool.needs_role:
                # 安全参数由框架注入，模型无法伪造（提高权限也不能覆盖 role）
                normalized["role"] = role
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
            payload = tool.handler(**normalized)
            return ToolResult(name=name, ok=True, payload=payload, latency_ms=(time.perf_counter() - started) * 1000)
        except (ToolError, ToolDenied) as exc:
            return ToolResult(name=name, ok=False, error=str(exc), latency_ms=(time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001 - 工具层必须兜住所有异常
            return ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}", latency_ms=(time.perf_counter() - started) * 1000)


def _infer_schema(handler: Callable[..., Any]) -> Dict[str, Any]:
    """从类型注解推断 schema（未显式声明时的兜底）。"""
    sig = inspect.signature(handler)
    props: Dict[str, Any] = {}
    required: List[str] = []
    for pname, param in sig.parameters.items():
        ann = param.annotation
        jtype = "string"
        if ann in (int,):
            jtype = "integer"
        elif ann in (float,):
            jtype = "number"
        elif ann in (bool,):
            jtype = "boolean"
        elif ann in (list, List):
            jtype = "array"
        elif ann in (dict, Dict):
            jtype = "object"
        entry: Dict[str, Any] = {"type": jtype}
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        else:
            required.append(pname)
        props[pname] = entry
    return {"type": "object", "properties": props, "required": required}


# --------------------------------------------------------------------------- #
# 内置工具
# --------------------------------------------------------------------------- #

# 指代词表：判断问题是否依赖上文。
#
# 「那」「那么」这类**指示代词**必须包含在内：
# "那最多能结转多少天" 靠自身的词完全无法定位主题，是最典型的多轮追问。
# 早期只放了 "那个"，漏掉了单独的 "那"，导致这类追问不会触发改写（实测发现的缺口）。
COREFERENCE_HINTS = ("它", "这个", "那个", "那", "上面", "刚才", "前面", "这些", "那些", "该", "此")


def build_registry(kb: KnowledgeBase) -> ToolRegistry:
    """创建带内置工具的注册表。

    三个工具刚好覆盖一个 RAG Agent 的最小闭环：
    **改写问题（消解指代）→ 检索文档（带权限）→ 列出文档（元信息问答）**。
    """
    registry = ToolRegistry()

    # 证据判定阈值集中在这里（唯一真相源），由 config 注入
    min_evidence = float(kb.config.get("agent.min_evidence_relevance", 0.28))
    min_span = int(kb.config.get("agent.min_evidence_span", 2))

    # -- 工具 1：混合检索 ---------------------------------------------- #
    def search_docs(query: str, top_k: int = 4, role: str = "public") -> Dict[str, Any]:
        """在知识库中做混合检索（BM25 + 稠密 + RRF + 重排），返回带引用标签的片段。

        Args:
            query: 检索查询，建议使用独立完整的问题而不是指代词。
            top_k: 返回片段数。
            role: 当前用户角色（由框架注入，模型无法伪造）。
        """
        hits = kb.search(query, top_k=int(top_k), role=role)
        results = [
            {
                "chunk_id": hit.chunk_id,
                "source": hit.source,
                "section": hit.section,
                "citation": hit.cite(),
                "score": round(float(hit.score), 5),
                # relevance：0~1 的实义字符覆盖率（语义明确，用于判断"有没有依据"）
                "relevance": round(float(hit.metadata.get("relevance", 0.0)), 5),
                # contrast：top-1 相对其余片段的领先倍数（区分"真命中"与"零散字面重合"）
                "contrast": round(float(hit.metadata.get("contrast", 0.0)), 3),
                # margin：top-1 与 top-2 的覆盖率差（拒答判定的辅助依据）
                "margin": round(float(hit.metadata.get("margin", 0.0)), 5),
                # longest_span_chars：问题与该段落的**最长连续命中字符数**
                # （区分"真命中"与"只共用两三个字"的关键指标）
                "longest_span_chars": int(hit.metadata.get("longest_span_chars", 0)),
                "coverage": round(float(hit.metadata.get("coverage", 0.0)), 5),
                "dense_rank": hit.dense_rank,
                "sparse_rank": hit.sparse_rank,
                "text": hit.text,
            }
            for hit in hits
        ]
        best_relevance = max((item["relevance"] for item in results), default=0.0)
        best_span = max((int(item.get("longest_span_chars", 0)) for item in results), default=0)
        margin = float(results[0].get("margin", 0.0)) if results else 0.0
        contrast = float(results[0].get("contrast", 0.0)) if results else 0.0

        # 权限提示：先看"问题是否明确指向了当前角色无权访问的文档"。
        # 这一步必须在证据判定**之前** —— 否则会拿别的文档凑答案，
        # 既答错问题又泄露"存在这份受限文档"的信息。
        # 判定从严（见 KnowledgeBase.restricted_matches 的注释），避免误报。
        restricted = kb.restricted_matches(query, role=role)

        # has_evidence：判定"这段检索结果是否真的构成回答依据"。
        #
        # 判定逻辑放在工具层，模型与 Mock 拿到的是**结论**，
        # 而不是让每个调用方自己猜阈值（阈值只应有一处定义）。
        #
        # 规则 = 实义覆盖率达标 AND 存在足够长的连续命中
        #   · relevance（覆盖率）管"问题的多个词确实对上了"，
        #     能挡掉"今天天气怎么样"这类完全无重合的提问；
        #   · longest_span（最长连续命中字符数）管"不是零散撞字"，
        #     能挡掉"公司股票代码"这类只共用"公司/代码"两个字的提问 ——
        #     它覆盖率有 0.33，但最长连续命中只有 2 个字符（"代码"）。
        #
        # 阈值由 scripts/_calibrate_span.py 在 16 条应回答 + 9 条应拒答的标注集上
        # 网格搜索得到：**误拒 0 条**，残余 4 条幻觉全部是"用词与某段高度重合、
        # 但实际不构成答案"（如"竞争对手的核心算法"撞上"禁止上传代码给外部 AI"）。
        #
        # ⚠ 这是**词面启发式**，不是语义判断。要做到语义级拒答需要真实 embedding
        #   或加一次模型判定 —— 本项目刻意保留并**如实写进评测**，而不是调参掩盖。
        # 阈值从闭包读取（不要在函数内重新赋值 —— 那会让 Python 把它当成局部变量，
        # 进而在读取时报 UnboundLocalError：本项目自己踩过这个低级坑）。
        has_evidence = (
            bool(results)
            and not restricted
            and best_relevance >= min_evidence
            and best_span >= min_span
        )
        return {
            "tool": "search_docs",
            "query": query,
            "role": role,
            "count": len(results),
            "best_relevance": round(best_relevance, 5),
            "best_span_chars": best_span,
            "margin": round(margin, 5),
            "contrast": round(contrast, 3),
            "has_evidence": has_evidence,
            "restricted_sources": restricted,
            "results": results,
            "note": "结果来自用户有权访问的文档；引用时使用 citation 字段。",
        }

    registry.register(
        "search_docs",
        search_docs,
        description="在知识库中做混合检索，返回与问题最相关的文档片段及其引用标签。",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "独立完整的检索查询"},
                "top_k": {"type": "integer", "default": 4, "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        },
        needs_role=True,
    )

    # -- 工具 2：多轮问题改写 ------------------------------------------ #
    def rewrite_question(question: str, history: Optional[List[str]] = None) -> Dict[str, Any]:
        """把含指代词的问题结合上文改写成一个可独立检索的问题。

        多轮问答的**第一大失败原因**就是直接把"那它呢"丢给检索器。
        这一步用轻量规则实现（可替换为一次小模型调用），但它必须在链路里存在。
        """
        history = list(history or [])
        rewritten = _rewrite(question, history)
        return {
            "tool": "rewrite_question",
            "original": question,
            "rewritten": rewritten,
            "changed": rewritten != question,
            "history_turns": len(history),
        }

    registry.register(
        "rewrite_question",
        rewrite_question,
        description="当问题含有'它/这个/上面说'等指代词时，结合上文改写为可独立检索的问题。",
        schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "history": {"type": "array", "items": {"type": "string"}, "description": "最近几轮问题"},
            },
            "required": ["question"],
        },
    )

    # -- 工具 3：列出可见文档 ------------------------------------------ #
    def list_documents(role: str = "public") -> Dict[str, Any]:
        """列出当前用户有权访问的文档清单（用于"知识库里有什么"这类元问题）。"""
        docs = kb.documents(role=role)
        return {
            "tool": "list_documents",
            "role": role,
            "count": len(docs),
            "documents": docs,
        }

    registry.register(
        "list_documents",
        list_documents,
        description="列出当前用户有权访问的知识库文档清单（回答'有哪些文档/知识库里有什么'）。",
        schema={"type": "object", "properties": {}},
        needs_role=True,
    )
    # 该功能无需添加新的工具，可在list_documents（）中实现。
    # def kb_stats() -> Dict[str, Any]:
    #     """返回知识库的统计信息"""
    #     return {"tool": "kb_stats", "stats": kb.stats()}
    # registry.register(
    #     "kb_stats", kb_stats,
    #     description = "返回知识库的文档数、片段数等统计信息。",
    #     schema = {"type": "object", "propertries": {}},
    # )

    return registry



def _rewrite(question: str, history: Sequence[str]) -> str:
    """规则式指代消解：把"它/这个"替换为上一轮问题里的核心名词。

    真实项目里这一步会换成一次小模型调用；这里用规则实现，好处是
    **完全确定、可单测**，也能让你在面试里讲清"改写的职责边界在哪"。
    """
    question = (question or "").strip()
    if not question or not history:
        return question
    if not any(hint in question for hint in COREFERENCE_HINTS):
        return question

    previous = history[-1]
    topic = _extract_topic(previous)
    if not topic:
        return question

    rewritten = question
    for hint in COREFERENCE_HINTS:
        if hint in rewritten:
            rewritten = rewritten.replace(hint, topic, 1)
            break
    # 补一个完整问句形态，便于检索
    if rewritten == question:
        rewritten = f"{topic} {question}"
    return _clean(rewritten)


def _extract_topic(text: str) -> str:
    """从上一轮问题里抽出主题词（取最长的中文名词片段或英文关键词）。"""
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{1,20}|[\u4e00-\u9fff]{2,8}", text or "")
    candidates = [c for c in candidates if c not in {"什么", "怎么", "如何", "可以", "需要", "哪些"}]
    if not candidates:
        return ""
    return max(candidates, key=len)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
