"""核心契约层：类型、配置、错误、mini-YAML —— 零第三方依赖。

这一层刻意保持"薄"：只放**跨模块共享的数据结构**与配置读取。
业务逻辑一律不放这里，避免它长成一个大杂烩。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# 错误体系（区分"可重试"与"不可重试"，工具层据此决定是否重试）
# --------------------------------------------------------------------------- #


class DocPilotError(Exception):
    """项目异常基类。"""

    retryable = False

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context


class ConfigError(DocPilotError):
    pass


class ToolError(DocPilotError):
    retryable = True


class ToolNotFound(ToolError):
    retryable = False


class ToolDenied(ToolError):
    """被策略拒绝（未授权 / 参数越界）。不可重试。"""

    retryable = False


class LLMError(DocPilotError):
    retryable = True


class LLMAuthError(LLMError):
    retryable = False


# --------------------------------------------------------------------------- #
# 对话与工具协议
# --------------------------------------------------------------------------- #


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class ToolCall:
    """一次工具调用请求（OpenAI function-calling 协议的超集）。"""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("call"))

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass
class Message:
    """一条对话消息。``tool_calls`` 只出现在 assistant 上，``tool_call_id`` 只在 tool 上。"""

    role: str  # system | user | assistant | tool
    content: str = ""
    name: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    def to_api(self) -> Dict[str, Any]:
        """转成 OpenAI 兼容接口需要的格式。"""
        out: Dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in self.tool_calls
            ]
        if self.role == "tool" and self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            out["name"] = self.name
        if self.tool_calls:
            out["tool_calls"] = [c.to_dict() for c in self.tool_calls]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        return out


# --------------------------------------------------------------------------- #
# 检索与答案
# --------------------------------------------------------------------------- #


@dataclass
class RetrievedChunk:
    """一条检索结果。``cite()`` 产出可直接写进答案的引用标签。"""

    chunk_id: str
    text: str
    source: str
    title: str = ""
    section: str = ""
    score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def cite(self) -> str:
        """引用标签：``文件名#小节``。

        只取**叶子小节名**而不是完整标题路径 —— 完整路径在嵌套文档里会长到
        ``员工手册 > 第三章 > 3.2 休假``，写进答案后完全没法读。
        """
        leaf = str(self.section).split(">")[-1].strip()
        return f"{self.source}#{leaf}" if leaf else self.source

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["citation"] = self.cite()
        return out


@dataclass
class Citation:
    """答案里引用到的一个来源片段。"""

    marker: str          # 答案正文里的标号，如 "[1]"
    chunk_id: str
    source: str
    section: str = ""
    snippet: str = ""

    @property
    def label(self) -> str:
        return f"{self.source}#{self.section}" if self.section else self.source

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["label"] = self.label
        return out


@dataclass
class Answer:
    """一次问答的完整产出：可直接渲染，也可直接评测。"""

    question: str = ""
    rewritten_question: str = ""
    text: str = ""
    citations: List[Citation] = field(default_factory=list)
    refused: bool = False               # 是否拒答
    refusal_reason: str = ""
    tool_calls: List[str] = field(default_factory=list)
    steps: int = 0
    tokens: int = 0
    latency_ms: float = 0.0
    degraded: bool = False              # 是否因预算/异常降级
    trace: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: List[RetrievedChunk] = field(default_factory=list)

    @property
    def citation_labels(self) -> List[str]:
        return [c.label for c in self.citations]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "rewritten_question": self.rewritten_question,
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "tokens": self.tokens,
            "latency_ms": round(self.latency_ms, 1),
            "degraded": self.degraded,
            "retrieved": [c.to_dict() for c in self.retrieved],
        }

    def to_markdown(self) -> str:
        lines: List[str] = [f"### 问题\n\n{self.question}"]
        if self.rewritten_question and self.rewritten_question != self.question:
            lines.append(f"\n> 检索用改写问题：{self.rewritten_question}")
        lines.append("\n### 回答\n")
        if self.refused:
            lines.append(f"**无法回答**（{self.refusal_reason}）\n")
        lines.append(self.text or "（空）")
        if self.citations:
            lines.append("\n### 引用来源\n")
            for cite in self.citations:
                lines.append(f"- **{cite.marker}** `{cite.label}`")
                if cite.snippet:
                    lines.append(f"  > {cite.snippet[:110]}…")
        lines.append(
            f"\n---\n*工具调用 {len(self.tool_calls)} 次 · 推理 {self.steps} 步 · "
            f"约 {self.tokens} token · {self.latency_ms:.0f} ms"
            + ("　（已降级）" if self.degraded else "")
            + "*"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

DEFAULT_CONFIG: Dict[str, Any] = {
    "agent": {
        "max_steps": 6,          # ReAct 最大轮次（精简版只留步数上限）
        "max_tool_calls": 8,
        "min_retrieval_score": 0.32,   # 低于此分数视为"没有依据"→ 拒答
        "history_turns": 4,            # 多轮改写时带入的历史轮数
    },
    "llm": {
        "provider": "mock",      # mock | deepseek | openai | openai_compat
        "model": "deepseek-flash",
        "temperature": 0.1,
        "max_output_tokens": 1024,
        "timeout_s": 60,
    },
    "rag": {
        "chunk_size": 360,
        "chunk_overlap": 50,
        "top_k_dense": 8,
        "top_k_sparse": 8,
        "top_k_final": 4,
        "rrf_k": 60,
        "rrf_weights": [1.0, 1.2],     # [稠密, 稀疏]，中文短查询稀疏侧略占优
        "embed_dim": 256,
        "min_token_len": 2,
        "enable_dense": True,          # 通道开关：仅供消融实验，正常使用保持 True
        "enable_sparse": True,
    },
    "web": {"host": "127.0.0.1", "port": 8020},
}


class Config:
    """点号访问的配置对象，支持 YAML 覆盖与环境变量覆盖。"""

    def __init__(self, data: Dict[str, Any], root: Optional[Path] = None) -> None:
        self._data = data
        self.root = root or Path.cwd()

    # -- 读取 ----------------------------------------------------------- #
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path_of(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (self.root / path)

    @property
    def provider(self) -> str:
        return str(os.environ.get("DOCPILOT_PROVIDER") or self.get("llm.provider", "mock")).lower()

    @property
    def model(self) -> str:
        return str(os.environ.get("DOCPILOT_MODEL") or self.get("llm.model", "deepseek-chat"))

    def api_key(self) -> Optional[str]:
        for env in ("DOCPILOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            if os.environ.get(env):
                return os.environ[env]
        return None

    def base_url(self) -> Optional[str]:
        explicit = os.environ.get("DOCPILOT_BASE_URL")
        if explicit:
            return explicit
        if self.provider == "deepseek":
            return "https://api.deepseek.com/v1"
        if self.provider == "openai":
            return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return os.environ.get("OPENAI_BASE_URL")

    def describe(self) -> str:
        return f"provider={self.provider} model={self.model} max_steps={self.get('agent.max_steps')}"


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str | Path] = None, root: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """加载配置：内置默认值 ← configs/docpilot.yaml ← configs/docpilot.local.yaml ← overrides。"""
    base_root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    data = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝

    candidate = Path(path) if path else (base_root / "configs" / "docpilot.yaml")
    if candidate.exists():
        data = _deep_merge(data, loads_yaml(candidate.read_text(encoding="utf-8")))

    local = base_root / "configs" / "docpilot.local.yaml"
    if local.exists():
        data = _deep_merge(data, loads_yaml(local.read_text(encoding="utf-8")))

    if overrides:
        data = _deep_merge(data, overrides)
    return Config(data, base_root)


# --------------------------------------------------------------------------- #
# mini-YAML：只为读配置文件，支持嵌套映射/列表/注释/引号/块标量
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - 取决于环境
    import yaml as _pyyaml

    _HAS_PYYAML = True
except Exception:  # pragma: no cover
    _pyyaml = None
    _HAS_PYYAML = False


def loads_yaml(text: str, force_mini: bool = False) -> Dict[str, Any]:
    """解析 YAML 文本为字典。

    Args:
        force_mini: 强制使用内置 mini-YAML。

    **为什么需要这个开关**：装了 PyYAML 时它会接管解析，于是内置实现
    （以及它给出的人性化报错）在装了 PyYAML 的机器上永远不会被走到 ——
    也就永远不会被测到。有了这个开关，两条路径都能测。
    """
    if _HAS_PYYAML and not force_mini:  # pragma: no cover
        try:
            data = _pyyaml.safe_load(text)
        except Exception as exc:  # noqa: BLE001 - 统一异常类型
            raise ConfigError(f"YAML 解析失败：{exc}") from exc
        return data if isinstance(data, dict) else (data or {})
    return _MiniYaml(text).parse()


class _MiniYaml:
    """mini-YAML：覆盖配置文件用到的语法子集，行为可预测、可单测。"""

    def __init__(self, text: str) -> None:
        self.lines: List[tuple[int, str, int]] = []
        for lineno, raw in enumerate(text.splitlines(), 1):
            # Tab 检查必须在"去注释/去空白"**之前**做：
            # `\t` 会被后面的 strip 吃掉，等再去查就永远查不到了（本项目踩过这个坑）。
            if raw[: len(raw) - len(raw.lstrip(" \t"))].find("\t") >= 0:
                raise ConfigError(f"第 {lineno} 行：缩进不允许使用 Tab，请改用空格")
            line = self._strip_comment(raw)
            if not line.strip():
                continue
            stripped = line.rstrip()
            indent = len(stripped) - len(stripped.lstrip(" "))
            self.lines.append((indent, stripped.strip(), lineno))
        self.pos = 0

    @staticmethod
    def _strip_comment(line: str) -> str:
        out: List[str] = []
        quote: Optional[str] = None
        for i, ch in enumerate(line):
            if quote:
                out.append(ch)
                if ch == quote and line[i - 1 : i] != "\\":
                    quote = None
                continue
            if ch in ("'", '"'):
                quote = ch
                out.append(ch)
                continue
            if ch == "#" and (i == 0 or line[i - 1] in " \t"):
                break
            out.append(ch)
        return "".join(out)

    def parse(self) -> Dict[str, Any]:
        if not self.lines:
            return {}
        value, _ = self._block(self.lines[0][0])
        return value if isinstance(value, dict) else {}

    def _block(self, indent: int) -> tuple[Any, int]:
        if self.pos >= len(self.lines):
            return None, indent
        cur_indent, content, _ = self.lines[self.pos]
        if cur_indent < indent:
            return None, cur_indent
        if content == "-" or content.startswith("- "):
            return self._list(cur_indent)
        return self._mapping(cur_indent)

    def _list(self, indent: int) -> tuple[List[Any], int]:
        items: List[Any] = []
        while self.pos < len(self.lines):
            cur_indent, content, lineno = self.lines[self.pos]
            if cur_indent < indent or not (content == "-" or content.startswith("- ")):
                break
            if cur_indent > indent:
                raise ConfigError(f"第 {lineno} 行：列表项缩进不一致")
            payload = content[1:].strip()
            self.pos += 1
            if not payload:
                if self.pos < len(self.lines) and self.lines[self.pos][0] > indent:
                    value, _ = self._block(self.lines[self.pos][0])
                    items.append(value)
                else:
                    items.append(None)
            elif _looks_like_kv(payload):
                items.append(self._inline_map_item(payload, indent))
            else:
                items.append(_scalar(payload))
        return items, indent

    def _inline_map_item(self, payload: str, indent: int) -> Dict[str, Any]:
        key, rest = _split_key(payload)
        node: Dict[str, Any] = {}
        child_indent = self.lines[self.pos][0] if (self.pos < len(self.lines) and self.lines[self.pos][0] > indent) else indent + 2
        node[key] = self._value_for(rest, child_indent)
        while self.pos < len(self.lines):
            cur_indent, content, _ = self.lines[self.pos]
            if cur_indent < child_indent or content.startswith("- "):
                break
            k, r = _split_key(content)
            self.pos += 1
            node[k] = self._value_for(r, child_indent)
        return node

    def _mapping(self, indent: int) -> tuple[Dict[str, Any], int]:
        node: Dict[str, Any] = {}
        while self.pos < len(self.lines):
            cur_indent, content, lineno = self.lines[self.pos]
            if cur_indent < indent:
                break
            if cur_indent > indent or content.startswith("- "):
                raise ConfigError(f"第 {lineno} 行：意外的缩进 {content!r}")
            key, rest = _split_key(content)
            self.pos += 1
            node[key] = self._value_for(rest, indent)
        return node, indent

    def _value_for(self, rest: str, indent: int) -> Any:
        if rest in ("|", ">", "|-", ">-"):
            return self._block_scalar(indent, folded=rest.startswith(">"))
        if rest:
            return _scalar(rest)
        if self.pos < len(self.lines) and self.lines[self.pos][0] > indent:
            value, _ = self._block(self.lines[self.pos][0])
            return value
        return None

    def _block_scalar(self, indent: int, folded: bool) -> str:
        collected: List[str] = []
        base: Optional[int] = None
        while self.pos < len(self.lines):
            cur_indent, content, _ = self.lines[self.pos]
            if cur_indent <= indent:
                break
            base = cur_indent if base is None else base
            collected.append(" " * (cur_indent - base) + content)
            self.pos += 1
        return (" " if folded else "\n").join(collected) + ("" if folded else "\n")


def _looks_like_kv(text: str) -> bool:
    quote: Optional[str] = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == ":" and (i + 1 >= len(text) or text[i + 1] in " \t"):
            return True
    return False


def _split_key(content: str) -> tuple[str, str]:
    quote: Optional[str] = None
    for i, ch in enumerate(content):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == ":":
            key = content[:i].strip().strip("'\"")
            if not key:
                raise ConfigError(f"键为空：{content!r}")
            return key, content[i + 1 :].strip()
    raise ConfigError(f"缺少 ':' —— {content!r}")


def _split_inline(text: str, sep: str = ",") -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    quote: Optional[str] = None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [_scalar(x) for x in _split_inline(inner)] if inner else []
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if not inner:
            return {}
        out: Dict[str, Any] = {}
        for item in _split_inline(inner):
            k, v = _split_key(item)
            out[k] = _scalar(v)
        return out
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def short_hash(*parts: Any) -> str:
    """稳定的短哈希，用作 chunk_id 与缓存键。"""
    blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def ensure_utf8_stdio() -> None:
    """把 stdout / stderr 切到 UTF-8（幂等，失败静默）。

    为什么需要这个函数
    ------------------
    Windows 的默认代码页是 GBK（cp936）。当 stdout **接管道或被重定向**时
    （CI、``| Select-Object``、写日志文件），Python 不走控制台的宽字符接口，
    而是按 locale 编码输出 —— 于是 ``eval`` 里的 ``✓``/``✗``、``info`` 里的
    ``✅``/``❌`` 会直接抛 ``UnicodeEncodeError``，让一条**已经跑完**的命令
    因为最后一行符号打不出来而失败（本项目在 Windows 上实测踩到：
    ``python -m docpilot eval | ...`` 直接以退出码 1 结束）。

    三个刻意的选择：

    * 用 ``errors="replace"`` —— 宁可显示成 ``?``，也不让评测因为打印失败而失败；
    * 只在编码**不是** UTF-8 时才改写 —— 接在真正的控制台上时 Python 用宽字符
      接口输出，本来就没这个问题，不要多此一举；
    * 用鸭子类型取 ``reconfigure`` —— 测试里 stdout 常被替换成 ``StringIO``，
      它没有这个方法，此处必须优雅跳过。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", "") or "").replace("-", "").lower()
        if encoding == "utf8":
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # 流已关闭或不支持改写：保持原样
            continue
