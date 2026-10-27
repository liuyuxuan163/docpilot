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
4. **无权限要明说，不能说成"没找到"。** 如果工具返回里 `restricted_sources` 非空，
   说明**问题指向了你无权访问的文档**（不是知识库没有这份内容）。
   此时必须明确告知"你当前角色没有访问权限、请联系文档负责人"，
   而**不能**说"知识库中没有相关依据"，也**不能**拿其它文档凑一个答案。
   > 这一条是真实模型暴露出来的：Mock 模式下权限话术是硬编码的，
   > 所以 Mock 基线永远看不出"模型把无权限说成没有依据"这个安全问题。
5. **不做承诺**。涉及金额、时限、审批权限的问题，只陈述文档原文，不给个人判断。
6. 回答用中文，简洁分点，先给结论再给依据。

## 工作流程
- 如果用户的问题里出现"它 / 这个 / 上面说的"等指代词，**先结合上面的对话历史自行判断它指什么**；
  判断不了再调用 `rewrite_question`。**调用改写工具后必须继续检索**，不要凭空作答。
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
        # 对话历史**必须无条件渲染进 prompt**。
        #
        # 这里修的是一个只有真实模型才会暴露的缺陷：原来写的是
        #     if context and history is None:  # ← 传了 history 就不放上下文
        # 而评测/多轮场景恰恰是"传 history + reset=True"：
        # conversation 被清空 → transcript() 为空 → 模型只看到一句孤立的问题，
        # 于是它合理地回答"你用了'它'，但没有上文"。
        # Mock 基线下永远发现不了：Mock 不需要理解上下文，它只按剧本调
        # rewrite_question 再检索，所以分数一直是 1.0。
        #
        # 现在：优先用调用方给的 history（只有问题、没有答案），
        # 其次用会话里累积的完整 transcript。模型能自己消解指代之后，
        # 规则式改写就降级成兜底手段。
        if recent:
            messages.append(Message(role="user", content=_render_history(recent, self.conversation)))
        elif self.conversation.questions:
            messages.append(Message(role="user", content=self.conversation.transcript()))
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


def _render_history(recent: Sequence[str], conversation: "Conversation") -> str:
    """把历史问题渲染成给模型看的上下文（只有问题也要能读）。

    为什么以问题为主：多轮追问真正需要的是"上一轮在问什么主题"，
    而答案往往很长、还会把模型带偏。只有当 history 与会话记录对得上时，
    才附上上一轮答案的首句作为补充线索。
    """
    lines = ["## 对话历史（最近几轮，按时间顺序）"]
    for index, question in enumerate(recent):
        lines.append(f"- 用户：{question}")
        if index < len(conversation.questions) and conversation.questions[index] == question:
            answer = conversation.answers[index] if index < len(conversation.answers) else ""
            if answer:
                lines.append(f"  助手：{answer[:80]}{'…' if len(answer) > 80 else ''}")
    lines += [
        "",
        "如果当前问题里有指代词（它/这个/那个/上面说的），请结合上面的历史**自行判断指向**；",
        "判断不了再调用 rewrite_question。**调用改写工具后必须继续检索**，不要凭空作答。",
    ]
    return "\n".join(lines)


_REFUSAL_PHRASES = (
    "无法回答", "无法提供", "无法给出", "无法确认", "无法判断", "无法据此",
    "没有找到", "未找到", "没有检索到", "未检索到", "检索不到",
    "没有相关依据", "没有相关", "没有关于", "没有提及", "未提及", "没有提到",
    "未包含", "不包含", "未涉及", "没有涉及", "没有直接规定", "没有给出",
    "没有写明", "没有规定", "知识库中没有任何",
    "不足以支撑", "不足以", "缺乏依据", "不做推测", "暂无",
    # 权限类拒答：必须覆盖，否则会被评测当成"答了但答错"
    "没有访问权限", "无权访问", "涉及受限文档", "受限来源",
    # 追问类（真实模型常用："请补充具体对象"）
    "请补充", "建议咨询", "建议向",
)

# "否定词 + 缺什么" 的组合，比死词表覆盖更广：
# 真实模型会说"知识库中没有『13 薪』的相关规定"，这条句式不在任何固定列表里。
_REFUSAL_RE = re.compile(
    r"(没有|未|无法|不能|暂无|缺乏)[^。；\n]{0,14}"
    r"(规定|条款|依据|说明|标准|时限|比例|数据|信息|说法|内容|记录|办法|流程"
    # 权限类必须单独覆盖：真实模型会说"没有访问《薪酬制度》文档的权限"，
    # 中间插了文档名，固定短语 "没有访问权限" 匹配不上。
    r"|权限|授权)"
)

# 判定只在前 90 字里做：正常作答里也常出现"文档未包含 X"这种补充说明，
# 看全文会把正常作答误判成拒答（实测正例误判从 3.6% 升到 20%）。
_REFUSAL_WINDOW = 90


def _first_lines(text: str) -> str:
    """取"结论区"：跳过 '结论：' 这类前缀，截到第一个句号/换行。"""
    head = re.sub(r"\s+", " ", (text or "").strip()).lstrip("#* ")
    head = re.sub(r"^(结论|答案|回答|简答)\s*[：:]?\s*", "", head)
    for sep in ("。", "；", "\n"):
        index = head.find(sep)
        if 0 < index <= 80:
            head = head[:index]
            break
    return head[:_REFUSAL_WINDOW]


def _looks_like_refusal(text: str) -> bool:
    """判断模型输出是不是拒答（用于评测"拒答准确率"）。

    必须覆盖**权限拒答**这一类：它同样是"没回答问题"，
    如果识别不出来，会被评测当成"答了但答错"，指标就失真了。

    **为什么只看结论区**：这个函数原来是对全文做关键词匹配，而词表是照着
    Mock 的话术写的。接真实模型后两个方向都错 —— 9 条正确拒答被漏判
    （模型说的是"没有 X 这个说法/未检索到/没有直接规定"），3 条正常作答被误判
    （正文里出现"文档未包含…"这类补充说明）。

    改法：窗口收窄到结论区 + 短语表换成"否定词 + 缺什么"的正则。
    在 129 条真实模型答案上，负例召回 87.8% → **98.6%**，
    正例误判 7.3% → **3.6%**（`scripts/eval_refusal_detector.py` 可复现）。
    """
    if not text:
        return True
    head = _first_lines(text)
    if any(phrase in head for phrase in _REFUSAL_PHRASES):
        return True
    return bool(_REFUSAL_RE.search(head))
