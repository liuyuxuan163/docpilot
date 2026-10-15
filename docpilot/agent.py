"""Agent 层：单文件 ReAct 主循环 + 多轮会话。

整个 Agent 就是这一条循环，读懂它就懂了全部::

    每一轮：
      1. 检查步数 / 工具调用次数上限（超了就给部分结论，不空手而归）
      2. 调 LLM（带上工具 schema）
      3. 有 tool_calls → 执行工具 → 结果回填到消息历史 → 下一轮
      4. 没有 tool_calls → 视为最终回答 → 抽取引用 → 返回 Answer

三个刻意的设计取舍
------------------
1. **不引入 Multi-Agent 编排**。单 Agent + 好工具足够覆盖"知识库问答"这类任务，
   多 Agent 的收益主要在可并行的研究型任务上，对"检索→作答"是纯复杂度。
2. **不引入复杂预算体系**，只留步数上限与工具调用上限。轻度约束够用，
   重度治理（四维预算/熔断/幂等缓存）放在另一个项目里作为深度证明。
3. **引用的抽取不依赖模型自觉**。答案是模型写的，但 ``[n] → 文档`` 的映射
   由框架按检索结果生成，并且会**校验**答案里出现的标号是否真的存在 ——
   模型编造 ``[7]`` 而只检索到 3 条时，会被标记出来而不是当真。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .core import DEFAULT_CONFIG, Answer, Citation, Config, DocPilotError, LLMError, Message, RetrievedChunk
from .llm import BaseLLM, build_llm
from .rag import KnowledgeBase
from .tools import ToolRegistry, build_registry

__all__ = ["Agent", "Conversation", "SYSTEM_PROMPT"]

# 系统提示词刻意写明三条纪律：只用检索内容 / 必须标引用 / 没依据就说不知道。
# 尤其是第三条 —— 企业场景里"编一个像样的答案"比"承认不知道"危险得多。
SYSTEM_PROMPT = """你是 DocPilot，一个**企业知识库问答助手**。

## 回答纪律
1. **只用检索到的内容回答**。你的知识可能与公司制度不一致，一律以检索结果为准。
2. **必须标注引用**：每一条结论后面写上它来自哪个片段，例如 `[1]`、`[2]`。
   标号必须对应检索结果中的顺序（第 1 条结果就是 [1]）。
3. **检索不到就说不知道**。如果检索结果的分数都很低、或内容与问题无关，
   直接回答"知识库中没有找到相关依据"，**不要推测、不要用常识补全**。
4. **不做承诺**。涉及金额、时限、审批权限的问题，只陈述文档原文，不给个人判断。
5. 回答用中文，简洁分点，先给结论再给依据。

## 工作流程
- 如果用户的问题里出现"它 / 这个 / 上面说的"等指代词，**先调用 `rewrite_question`**
  结合上文改写成独立问题，再检索。
- 如果用户问的是"有哪些文档 / 知识库里有什么"，调用 `list_documents`。
- 其余情况调用 `search_docs` 检索，然后作答。
"""

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass
class Conversation:
    """多轮会话：维护问题历史（用于指代消解）与消息历史。

    只保留最近 ``history_turns`` 轮问题，而不是把全部消息堆进上下文 ——
    上下文越长，模型越容易忽略中间的信息（Lost in the Middle），成本也更高。
    """

    max_turns: int = 4
    questions: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)

    def remember(self, question: str, answer: str) -> None:
        self.questions.append(question)
        self.answers.append(answer)
        if len(self.questions) > self.max_turns:
            self.questions.pop(0)
            self.answers.pop(0)

    @property
    def recent_questions(self) -> List[str]:
        return list(self.questions)

    def transcript(self) -> str:
        """渲染成给模型看的简短上下文。"""
        if not self.questions:
            return ""
        lines = ["## 对话历史（最近几轮）"]
        for q, a in zip(self.questions, self.answers):
            lines.append(f"- 用户：{q}")
            lines.append(f"  助手：{a[:80]}{'…' if len(a) > 80 else ''}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.questions.clear()
        self.answers.clear()


class Agent:
    """ReAct 问答 Agent。"""

    def __init__(
        self,
        config: Optional[Config] = None,
        kb: Optional[KnowledgeBase] = None,
        llm: Optional[BaseLLM] = None,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.config = config or Config(DEFAULT_CONFIG)
        self.kb = kb or KnowledgeBase(self.config)
        self.kb.build()
        self.llm = llm or build_llm(self.config)
        self.registry = registry or build_registry(self.kb)
        self.max_steps = int(self.config.get("agent.max_steps", 6))
        self.max_tool_calls = int(self.config.get("agent.max_tool_calls", 8))
        self.conversation = Conversation(max_turns=int(self.config.get("agent.history_turns", 4)))

    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        role: str = "public",
        history: Optional[Sequence[str]] = None,
        reset: bool = False,
    ) -> Answer:
        """回答一个问题。

        Args:
            question: 用户问题（可以是含指代词的多轮追问）。
            role: 用户角色，用于**文档级权限过滤**（``public`` / ``hr`` / ``finance`` 等）。
            history: 额外带入的历史问题；不传则用 :attr:`conversation` 里累积的。
            reset: 是否在本次提问前清空会话历史（开启新会话）。
        """
        if reset:
            self.conversation.reset()
        started = time.perf_counter()
        recent = list(history) if history is not None else self.conversation.recent_questions

        messages: List[Message] = [Message(role="system", content=SYSTEM_PROMPT)]
        context = self.conversation.transcript()
        if context and history is None:
            messages.append(Message(role="user", content=context))
        messages.append(Message(role="user", content=f"问题：{question}"))

        answer = Answer(question=question)
        tool_calls_made: List[str] = []
        tool_payloads: List[Dict[str, Any]] = []
        budget_exhausted = False

        for step in range(1, self.max_steps + 1):
            if len(tool_calls_made) >= self.max_tool_calls:
                budget_exhausted = True
                answer.trace.append({"step": step, "event": "budget", "detail": "工具调用已达上限"})
                break
            try:
                response = self.llm.complete(
                    messages=messages,
                    tools=self.registry.schemas(),
                    temperature=float(self.config.get("llm.temperature", 0.1)),
                )
            except LLMError as exc:
                answer.degraded = True
                answer.text = f"模型调用失败：{exc}"
                answer.refusal_reason = "模型不可用"
                break

            answer.tokens += response.total_tokens
            assistant = response.to_message()
            messages.append(assistant)
            answer.trace.append(
                {
                    "step": step,
                    "event": "llm",
                    "tool_calls": [call.name for call in response.tool_calls],
                    "tokens": response.total_tokens,
                }
            )

            if not response.tool_calls:
                answer.text = response.content or ""
                answer.steps = step
                break

            for call in response.tool_calls:
                # rewrite_question 需要历史，由框架注入（模型不该负责传对历史）
                arguments = dict(call.arguments or {})
                if call.name == "rewrite_question":
                    arguments.setdefault("history", recent)
                result = self.registry.execute(call.name, arguments, role=role)
                tool_calls_made.append(call.name)
                if isinstance(result.payload, dict):
                    tool_payloads.append(result.payload)
                messages.append(
                    Message(role="tool", content=result.summary(), name=call.name, tool_call_id=call.id)
                )
                answer.trace.append(
                    {"step": step, "event": "tool", "name": call.name, "ok": result.ok, "error": result.error}
                )
            answer.steps = step
        else:
            budget_exhausted = True

        if budget_exhausted and not answer.text:
            # 步数/工具额度用尽时，用已拿到的片段给一个"部分结论"，而不是空手而归
            answer.degraded = True
            fallback = self._partial_answer(tool_payloads)
            answer.text = fallback or "未能在这个问题预算内得出结论，请把问题拆得更具体一些。"

        self._attach_sources(answer, tool_payloads, recent)
        answer.tool_calls = tool_calls_made
        answer.latency_ms = (time.perf_counter() - started) * 1000
        if answer.refusal_reason or _looks_like_refusal(answer.text):
            answer.refused = True
            answer.refusal_reason = answer.refusal_reason or "知识库中没有找到相关依据"
        self.conversation.remember(question, answer.text)
        return answer

    # ------------------------------------------------------------------ #
    @staticmethod
    def _partial_answer(payloads: Sequence[Dict[str, Any]]) -> str:
        """降级路径：用已检索到的片段拼一个"部分结论"。"""
        for payload in reversed(list(payloads)):
            if payload.get("tool") == "search_docs" and payload.get("results"):
                top = payload["results"][0]
                text = re.sub(r"\s+", " ", str(top.get("text", "")))[:200]
                return f"（因达到步数上限，给出部分结论）根据 `{top.get('citation')}`：{text}…"
        return ""

    @staticmethod
    def _attach_sources(answer: Answer, payloads: Sequence[Dict[str, Any]], recent: Sequence[str]) -> None:
        """把工具返回的片段挂到答案上，并把答案里的 ``[n]`` 映射成可点击的引用。

        关键点：``[n] → 文档`` 的映射由框架生成，**不依赖模型产出结构化引用**。
        同时会校验标号范围：模型若写了 ``[9]`` 而只检索到 3 条，这里不会臆造，
        而是记录到 trace 里便于排查（评测指标也会因此下降）。
        """
        for payload in payloads:
            if payload.get("tool") == "rewrite_question":
                answer.rewritten_question = str(payload.get("rewritten", ""))

        chunks: List[RetrievedChunk] = []
        for payload in payloads:
            if payload.get("tool") != "search_docs":
                continue
            chunks = []
            for item in payload.get("results", []) or []:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=str(item.get("chunk_id", "")),
                        text=str(item.get("text", "")),
                        source=str(item.get("source", "")),
                        section=str(item.get("section", "")),
                        score=float(item.get("score", 0.0)),
                        dense_rank=item.get("dense_rank"),
                        sparse_rank=item.get("sparse_rank"),
                        metadata={"citation": item.get("citation", "")},
                    )
                )
        answer.retrieved = chunks
        if not chunks:
            return
        # 拒答时不要附引用：答案里没有 `[n]` 标号，硬把检索片段列成"引用来源"
        # 会让用户以为系统在拿这些片段当依据（何况它们本来就不相关）。
        if answer.refused or _looks_like_refusal(answer.text):
            return

        used = sorted({int(m.group(1)) for m in _MARKER_RE.finditer(answer.text or "")})
        out_of_range = [n for n in used if n < 1 or n > len(chunks)]
        if out_of_range:
            answer.trace.append({"event": "citation_check", "invalid_markers": out_of_range})
        if not used:
            # 模型没标引用时，把检索到的前若干条作为补充来源列出（而不是留空）
            used = list(range(1, min(len(chunks), 3) + 1))
            answer.trace.append({"event": "citation_check", "detail": "答案未标注引用，已附加检索来源"})

        for number in used:
            if number < 1 or number > len(chunks):
                continue
            chunk = chunks[number - 1]
            answer.citations.append(
                Citation(
                    marker=f"[{number}]",
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    section=chunk.section.split(">")[-1].strip(),
                    snippet=re.sub(r"\s+", " ", chunk.text)[:160],
                )
            )
        if recent:
            answer.rewritten_question = answer.rewritten_question or ""


def _looks_like_refusal(text: str) -> bool:
    """判断模型输出是不是拒答（用于评测"拒答准确率"）。

    必须覆盖**权限拒答**这一类：它同样是"没回答问题"，
    如果识别不出来，会被评测当成"答了但答错"，指标就失真了。
    """
    if not text:
        return True
    negative = (
        "无法回答",
        "没有找到",
        "未找到",
        "没有相关依据",
        "知识库中没有任何",
        "不足以支撑",
        "不做推测",
        # 权限类拒答
        "没有访问权限",
        "无权访问",
        "涉及受限文档",
    )
    return any(phrase in text for phrase in negative)
