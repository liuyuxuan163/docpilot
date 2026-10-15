"""检索层：切块 → 稀疏(BM25) + 稠密(本地嵌入) → RRF 融合 → 启发式重排。

为什么"通用问答助手"也必须有这一层
------------------------------------
企业文档里的关键信息往往是**低频精确词**：制度编号（``HR-2024-07``）、
金额、时限（``15 个工作日``）、系统名（``SSO``）。纯向量检索对这类词很弱；
而用户又常用口语提问（"年假能攒到明年吗"），纯 BM25 也招架不住。
所以两路都要，融合用 RRF。

RRF 只用**排名**不用**分数**：

    fused(d) = Σ_r  w_r / (k + rank_r(d))

好处是不用去纠结 BM25 分数（无界）和余弦相似度（[-1,1]）怎么对齐量纲 ——
这也是 Elasticsearch 等系统默认采用它的原因（其默认 ``rank_constant=60``）。

本模块纯 Python + 标准库实现，几十到几万 chunk 的规模完全够用，
且**完全确定性**（同一输入永远同一结果），这是能对检索质量写回归测试的前提。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .core import Config, RetrievedChunk, short_hash
from .matching import _FILLER, _best_match, significant_spans, span_coverage

__all__ = [
    "tokenize",
    "chunk_markdown",
    "BM25Index",
    "LocalEmbedder",
    "HybridRetriever",
    "KnowledgeBase",
    "rrf_fuse",
    "salient_terms",
]

# --------------------------------------------------------------------------- #
# 分词：中文 bigram + trigram，英文/数字整词
# --------------------------------------------------------------------------- #

_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_TOKEN_RE = re.compile(f"[{_CJK}]+|[A-Za-z][A-Za-z0-9_\\-\\.]*|\\d+(?:\\.\\d+)?")
_CJK_RE = re.compile(f"[{_CJK}]+")

STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "或", "对", "把", "被", "为", "以", "及", "等", "就", "都", "也", "很",
    "会", "能", "要", "有", "无", "中", "上", "下", "后", "前", "我", "你", "他", "她", "它", "们", "这", "那",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "be", "as", "by", "at",
    "it", "this", "that", "from", "we", "you", "can", "will", "not", "how", "what", "when", "where", "which",
}

# 同义词双向扩展：把口语问法与文档书面语打通
SYNONYMS: Dict[str, List[str]] = {
    "年假": ["带薪年休假", "annual leave", "年休假"],
    "调休": ["调休假", "加班换休"],
    "报销": ["费用报销", "报销流程"],
    "差旅": ["出差", "差旅费"],
    "转正": ["试用期转正", "转正评估"],
    "请假": ["休假申请", "请假流程"],
    "加班": ["延时工作", "overtime"],
    "权限": ["访问权限", "授权"],
    "密码": ["口令", "登录密码", "password"],
    "远程": ["居家办公", "远程办公", "work from home"],
    "安全": ["信息安全", "security"],
    "绩效": ["绩效考核", "performance"],
    "工资": ["薪资", "薪酬", "salary"],
}


def tokenize(text: str, min_len: int = 2, expand: bool = True) -> List[str]:
    """切成检索 token。

    中文用 **bigram + trigram** 双粒度：bigram 保召回，trigram 提区分度。
    只用 bigram 时"年假能攒"会把讲"年假申请"的段落也排到前面 —— 这是实测过的退化。
    同义词扩展只对**整段中文/英文词**生效，不对 n-gram 生效（否则噪声爆炸）。

    **n-gram 必须过滤"纯虚词组合"**：中文 n-gram 是在字符层面生成的，
    "的了是在" 会产出 ``的了`` / ``了是`` / ``了是在`` 这类完全无意义的组合。
    它们不仅无效，还会污染倒排索引与 IDF（把无关文档的 BM25 分数抬起来）。
    过滤规则：n-gram 中至少要有一个实义字。
    """
    if not text:
        return []
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text):
        if _CJK_RE.fullmatch(raw):
            if len(raw) >= 2:
                tokens.extend(
                    raw[i : i + 2] for i in range(len(raw) - 1) if _has_content_char(raw[i : i + 2])
                )
            if len(raw) >= 3:
                tokens.extend(
                    raw[i : i + 3] for i in range(len(raw) - 2) if _has_content_char(raw[i : i + 3])
                )
            if len(raw) <= 4 and _has_content_char(raw):
                tokens.append(raw)
            if expand:
                tokens.extend(_expand(raw))
        else:
            token = raw.lower()
            tokens.append(token)
            if expand:
                tokens.extend(_expand(token))
    out: List[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        if len(token) < min_len and not _CJK_RE.fullmatch(token):
            continue
        out.append(token)
    return out


def _has_content_char(segment: str) -> bool:
    """段落里至少有一个实义字（不是纯虚词/纯疑问词）。"""
    return any(ch not in _CJK_FILLER for ch in segment)


# 与 matching._FILLER 保持一致的虚词集合。
# 放在两个模块会漂移，所以这里从 matching 导入同一份定义（见文件顶部 import）。
_CJK_FILLER = _FILLER


def _expand(token: str) -> List[str]:
    """同义词双向扩展：查询写口语要能召回书面语，反之亦然。"""
    out: List[str] = []
    for canonical, variants in SYNONYMS.items():
        if token == canonical:
            out.extend(variants)
        elif token in variants:
            out.append(canonical)
            out.extend(v for v in variants if v != token)
    return out


# --------------------------------------------------------------------------- #
# 切块：结构优先（按标题）+ 长度收敛（贪心拼接 + 句子边界二次切分）
# --------------------------------------------------------------------------- #

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SENT_RE = re.compile(r"(?<=[。！？；\.\!\?;])\s*")


def chunk_markdown(
    text: str,
    source: str,
    chunk_size: int = 360,
    chunk_overlap: int = 50,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[RetrievedChunk]:
    """把一篇 Markdown 文档切成带章节路径的 chunk。

    保留章节路径的两个用处：① 引用可以精确到小节；② 检索过滤时可以按章节标签筛。
    """
    if not text.strip():
        return []
    sections: List[Tuple[str, List[str]]] = []
    stack: Dict[int, str] = {}
    current_title = Path(source).stem
    buffer: List[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            if buffer:
                sections.append((current_title, buffer))
                buffer = []
            level = len(match.group(1))
            stack[level] = match.group(2).strip()
            for deeper in [k for k in stack if k > level]:
                stack.pop(deeper, None)
            current_title = " > ".join(stack[k] for k in sorted(stack))
            continue
        buffer.append(line)
    if buffer:
        sections.append((current_title, buffer))

    chunks: List[RetrievedChunk] = []
    seen: Set[str] = set()
    for section, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        paragraphs: List[str] = []
        for block in re.split(r"\n\s*\n", body):
            block = block.strip()
            if not block:
                continue
            paragraphs.extend(_split_long(block, chunk_size))

        buf = ""
        for para in paragraphs:
            candidate = f"{buf}\n{para}".strip() if buf else para
            if len(candidate) <= chunk_size:
                buf = candidate
                continue
            if buf:
                _emit(chunks, seen, buf, source, section, metadata)
            tail = buf[-chunk_overlap:] if (buf and chunk_overlap > 0) else ""
            buf = f"{tail}\n{para}".strip() if tail else para
        if buf:
            _emit(chunks, seen, buf, source, section, metadata)
    return chunks


def _emit(
    chunks: List[RetrievedChunk],
    seen: Set[str],
    text: str,
    source: str,
    section: str,
    metadata: Optional[Dict[str, Any]],
) -> None:
    """生成一条 chunk（内容去重，ID 稳定）。"""
    cleaned = re.sub(r"[ \t]{2,}", " ", text).strip()
    if not cleaned:
        return
    digest = hashlib.blake2b(cleaned.encode("utf-8"), digest_size=8).hexdigest()
    if digest in seen:
        return
    seen.add(digest)
    leaf = section.split(">")[-1].strip()
    chunks.append(
        RetrievedChunk(
            chunk_id=f"ck_{short_hash(source, section, cleaned[:64])}",
            text=cleaned,
            source=source,
            title=Path(source).stem,
            section=section,
            metadata={**(metadata or {}), "section_leaf": leaf},
        )
    )


def _split_long(text: str, size: int) -> List[str]:
    """按句子边界把超长段落切开（不硬切，避免把一句话劈成两半）。"""
    if len(text) <= size:
        return [text]
    out: List[str] = []
    buf = ""
    for sentence in [s for s in _SENT_RE.split(text) if s.strip()]:
        if len(buf) + len(sentence) <= size:
            buf += sentence
            continue
        if buf:
            out.append(buf.strip())
        while len(sentence) > size:
            out.append(sentence[:size].strip())
            sentence = sentence[size:]
        buf = sentence
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if c]


# --------------------------------------------------------------------------- #
# 稀疏检索：BM25（含 BM25+ 下界修正，避免长文档被系统性低估）
# --------------------------------------------------------------------------- #


class BM25Index:
    """内存 BM25 倒排索引。自己实现而不是装 rank_bm25，是为了零依赖 + 可增量可序列化。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 0.5, min_len: int = 2) -> None:
        self.k1, self.b, self.delta, self.min_len = k1, b, delta, min_len
        self.doc_ids: List[str] = []
        self.term_freq: Dict[str, Counter] = {}
        self.doc_len: Dict[str, int] = {}
        self.postings: Dict[str, Set[str]] = {}
        self._avgdl = 0.0

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self.term_freq:
            self.remove(doc_id)
        counts = Counter(tokenize(text, min_len=self.min_len))
        self.doc_ids.append(doc_id)
        self.term_freq[doc_id] = counts
        self.doc_len[doc_id] = max(1, sum(counts.values()))
        for term in counts:
            self.postings.setdefault(term, set()).add(doc_id)
        self._avgdl = 0.0

    def remove(self, doc_id: str) -> None:
        for term in self.term_freq.pop(doc_id, {}):
            bucket = self.postings.get(term)
            if bucket:
                bucket.discard(doc_id)
                if not bucket:
                    self.postings.pop(term, None)
        self.doc_len.pop(doc_id, None)
        if doc_id in self.doc_ids:
            self.doc_ids.remove(doc_id)
        self._avgdl = 0.0

    @property
    def avgdl(self) -> float:
        if self._avgdl <= 0 and self.doc_len:
            self._avgdl = sum(self.doc_len.values()) / len(self.doc_len)
        return self._avgdl or 1.0

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        if not df or not self.n_docs:
            return 0.0
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_id: str) -> float:
        counts = self.term_freq.get(doc_id)
        if not counts:
            return 0.0
        dl = self.doc_len.get(doc_id, 1)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        total = 0.0
        for term in set(tokenize(query, min_len=self.min_len)):
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = self.idf(term)
            total += idf * (tf * (self.k1 + 1)) / (tf + norm + 1e-9) + self.delta * idf
        return total

    def search(self, query: str, top_k: int = 8, allowed: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        candidates: Set[str] = set()
        for term in set(tokenize(query, min_len=self.min_len)):
            candidates |= self.postings.get(term, set())
        if allowed is not None:
            candidates &= allowed
        scored = [(doc_id, self.score(query, doc_id)) for doc_id in candidates]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]

    # -- 持久化 --------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "term_freq": {k: dict(v) for k, v in self.term_freq.items()},
            "params": [self.k1, self.b, self.delta, self.min_len],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BM25Index":
        k1, b, delta, min_len = data.get("params", [1.5, 0.75, 0.5, 2])
        index = cls(k1=k1, b=b, delta=delta, min_len=int(min_len))
        index.doc_ids = list(data.get("doc_ids", []))
        index.doc_len = {str(k): int(v) for k, v in dict(data.get("doc_len", {})).items()}
        for doc_id, counts in dict(data.get("term_freq", {})).items():
            index.term_freq[str(doc_id)] = Counter({str(k): int(v) for k, v in counts.items()})
            for term in counts:
                index.postings.setdefault(str(term), set()).add(str(doc_id))
        return index


# --------------------------------------------------------------------------- #
# 稠密检索：本地确定性嵌入（hashing trick + 符号投影 + L2 归一化）
# --------------------------------------------------------------------------- #


class LocalEmbedder:
    """本地嵌入：token 哈希投影到固定维度，无需模型、无需网络、结果确定。

    它没有神经网络嵌入的语义泛化能力，但在"领域术语密集的企业文档"上
    配合 BM25 混合检索已经够用，而且换来三个关键收益：
    **零依赖、零成本、完全可复现**（检索评测才有意义）。

    工程上要升级到真实 embedding 只需替换 ``embed()`` 的实现，调用方不用改。
    """

    name = "local_hashing"

    def __init__(self, dim: int = 256, min_len: int = 2) -> None:
        self.dim = int(dim)
        self.min_len = int(min_len)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = tokenize(text, min_len=self.min_len)
        if not tokens:
            return vec
        counts = Counter(tokens)
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            sign = 1.0 if (value >> 63) & 1 else -1.0
            weight = (1.0 + math.log(count)) * (1.4 if len(token) == 3 else 1.0)
            vec[value % self.dim] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def signature(self) -> str:
        return f"{self.name}:{self.dim}:{self.min_len}"


# --------------------------------------------------------------------------- #
# RRF 融合 + 启发式重排
# --------------------------------------------------------------------------- #


def rrf_fuse(ranked_lists: Sequence[Sequence[str]], weights: Optional[Sequence[float]] = None, k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion：只用排名，免去分数量纲对齐问题。"""
    weights = list(weights) if weights else [1.0] * len(ranked_lists)
    fused: Dict[str, float] = {}
    for weight, ranked in zip(weights, ranked_lists):
        for rank, doc_id in enumerate(ranked, 1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
    return fused


def salient_terms(text: str, limit: int = 6) -> List[str]:
    """抽取查询里的**实义片段**（用于计算覆盖度）。

    直接复用 :mod:`docpilot.matching` 的片段切分：去掉标点与虚词后，
    "P4 职级的绩效工资占比是多少？" → ``["P4", "职级", "绩效工资占比"]``。
    这比用 n-gram 当"主题词"靠谱得多 —— 后者会产出
    ``占比是 / 工资占 / 效工资`` 这类跨词边界的组合，完全无法用于判断相关性。
    """
    return significant_spans(text)[:limit]


def _rerank(
    query: str,
    chunks: List[RetrievedChunk],
    base: Dict[str, float],
    corpus_size: Optional[int] = None,
) -> List[RetrievedChunk]:
    """无模型重排：IDF 加权覆盖 + 字符覆盖 + 标题/文件名先验。

    Args:
        corpus_size: 知识库总片段数，用于计算片段级 IDF。
            传 ``None`` 时退化为"按候选池估算"，只在单测里这样用。

    "文件名先验"是实测有效的：查询"年假怎么算"时，如果 ``leave-policy.md``
    命中"年假"，说明整篇文档都在讲这个主题，应当加权；
    否则同样提到"年假"的零散段落会把它挤下去。

    同时写入用于**拒答判定**的信号：

    * ``metadata["relevance"]``：IDF 加权覆盖率为主的综合相关度（0~1）；
    * ``metadata["coverage"]`` / ``idf_coverage``：原始覆盖率与 IDF 加权覆盖率；
    * ``metadata["longest_span_chars"]``：最长连续命中字符数（拒答主判据）。

    为什么不用 ``score`` 判断"有没有依据"？因为 ``score`` 是 RRF 融合分与覆盖率
    的加权混合值，**量纲不可比、无法跨查询设固定阈值**。早期版本直接用它做
    ``< 0.3`` 判定，出现了最尴尬的失败：**引用里明明有正确片段，却说"没有依据"**。
    用不可比的混合分数做绝对阈值，是很容易犯又很难查的错。
    """
    query_tokens = set(tokenize(query, min_len=1))
    spans = significant_spans(query)
    if not query_tokens and not spans:
        return chunks

    # 片段级 IDF：**通用短语不该和高区分度短语等权**。
    # "新员工入职" 出现在整个 onboarding 文档里，几乎每段都沾；
    # 而 "年假折算" 只出现在休假办法里。若不加权，
    # 问"新员工年假怎么折算"会被 onboarding 的通用短语抬到第一，
    # 而真正写着折算公式的 #年假额度与计算方式 掉到第六（实测过）。
    total_chunks = max(1, int(corpus_size or 0) or len(chunks))
    span_idf: Dict[str, float] = {}
    for span in spans:
        df = sum(1 for c in chunks if span in f"{c.section} {c.text}")
        span_idf[span] = math.log(1.0 + total_chunks / (1.0 + df))

    for chunk in chunks:
        text = f"{chunk.section} {chunk.text}"
        coverage, longest_ratio = span_coverage(query, text)
        body = set(tokenize(chunk.text, min_len=1))
        title = set(tokenize(f"{chunk.section} {chunk.title}", min_len=1))
        doc = set(tokenize(chunk.source.rsplit('/', 1)[-1].rsplit('.', 1)[0].replace('-', ' '), min_len=1))
        token_coverage = len(query_tokens & body) / max(1, len(query_tokens))
        title_hit = len(query_tokens & title) / max(1, len(query_tokens))
        doc_hit = len(query_tokens & doc) / max(1, len(query_tokens))

        # IDF 加权覆盖：命中"高区分度片段"比命中"通用短语"更有价值
        weighted_total = 0.0
        weighted_hit = 0.0
        for span in spans:
            weight = span_idf.get(span, 1.0) * len(span)
            weighted_total += weight
            weighted_hit += weight * (_best_match(span, text) / max(1, len(span)))
        idf_coverage = weighted_hit / weighted_total if weighted_total > 0 else 0.0

        # 相关性以 IDF 加权覆盖为主，字符覆盖与最长连续命中为辅
        relevance = 0.55 * idf_coverage + 0.25 * coverage + 0.20 * longest_ratio
        rank_score = 0.7 * token_coverage + 0.2 * title_hit + 0.1 * doc_hit

        chunk.metadata["relevance"] = round(relevance, 5)
        chunk.metadata["coverage"] = round(coverage, 5)
        chunk.metadata["idf_coverage"] = round(idf_coverage, 5)
        chunk.metadata["longest_span"] = round(longest_ratio, 5)
        # 最长连续命中**字符数**（不是比例）：用于拒答判定。
        # 比例会随问题长度变化，字符数是绝对量，阈值更好标定（见 tools.build_registry）。
        chunk.metadata["longest_span_chars"] = max(
            (_best_match(span, text) for span in spans), default=0
        )
        # 排序分：融合分保留一部分（含 BM25/向量信息），再叠加字符级覆盖
        chunk.score = round(0.4 * base.get(chunk.chunk_id, 0.0) + 0.6 * (0.6 * relevance + 0.4 * rank_score), 6)

    # 排序：**先按 relevance，再按融合分**。
    #
    # 这里修的是一个真实 bug：早期只按混合 score 排序，而 score 里含
    # token 覆盖率与标题命中，结果"标题碰巧含'常见问题'"的无关文档
    # 会把真正相关的段落挤到后面（实测：问"出差住酒店超预算"时
    # hr/onboarding.md 排第一，而真正写着"超标部分由个人承担"的
    # finance/expense-policy.md#住宿标准 掉到第六）。
    # 而 relevance 是"问题与段落的实义覆盖度"，语义明确，必须做主序；
    # 融合分只在 relevance 相同时用来打破平局。
    chunks.sort(key=lambda c: (-float(c.metadata.get("relevance", 0.0)), -c.score, c.chunk_id))

    # 对比度与间隔（margin）：用于拒答判定。
    #
    # 实测数据（见 docs/架构说明.md 的"拒答判定"一节）：
    #   真命中问题的 margin 通常 ≥ 0.08，超纲问题大多为 0.00 或很小；
    #   而"零散字面重合"（如问"公司股票代码"命中讲代码安全的段落）
    #   表现为 margin 为 0 —— 所有片段一样相关，等于没有一段真的相关。
    #
    # **诚实说明**：这套判定本质是**词面启发式**，不是语义模型。
    # 它对"完全超纲"（天气/午饭）与"零散字重合"效果很好，
    # 但对"问题用词与某段文档高度重合、实际却不构成答案"的情形会误判
    # （评测集里的 `refuse-03 竞争对手核心算法` 就是这类已知失败）。
    # 要做到真正的语义级拒答，需要换成真实 embedding 或加一次模型判定，
    # 本项目刻意保留这一限制并写进评测，而不是假装它不存在。
    relevance_values = [float(c.metadata.get("relevance", 0.0)) for c in chunks]
    top_relevance = relevance_values[0] if relevance_values else 0.0
    second = relevance_values[1] if len(relevance_values) > 1 else 0.0
    rest = relevance_values[1:]
    mean_rest = sum(rest) / len(rest) if rest else 0.0
    margin = round(top_relevance - second, 5)
    contrast = round(top_relevance / mean_rest, 3) if mean_rest > 1e-6 else 0.0
    for chunk in chunks:
        chunk.metadata["margin"] = margin
        chunk.metadata["contrast"] = contrast
    return chunks


# --------------------------------------------------------------------------- #
# 混合检索器 + 知识库
# --------------------------------------------------------------------------- #


@dataclass
class HybridRetriever:
    """BM25 + 稠密 → RRF → 重排。对外只暴露 ``search()``。"""

    chunks: Dict[str, RetrievedChunk]
    bm25: BM25Index
    embedder: LocalEmbedder
    vectors: Dict[str, List[float]] = field(default_factory=dict)
    top_k_dense: int = 8
    top_k_sparse: int = 8
    rrf_k: int = 60
    weights: Tuple[float, float] = (1.0, 1.2)
    last_debug: Dict[str, Any] = field(default_factory=dict)

    def search(
        self,
        query: str,
        top_k: int = 4,
        allowed_ids: Optional[Set[str]] = None,
    ) -> List[RetrievedChunk]:
        """混合检索。``allowed_ids`` 用于**权限隔离**（在检索阶段就滤掉无权访问的 chunk）。"""
        if not query.strip() or not self.chunks:
            return []
        allowed = allowed_ids if allowed_ids is not None else set(self.chunks)
        # 全库片段数（不是候选数）：IDF 的分母必须反映整库的"常见程度"
        corpus_size = len(self.chunks)

        sparse_hits = self.bm25.search(query, top_k=self.top_k_sparse, allowed=allowed)
        dense_hits = self._dense_search(query, self.top_k_dense, allowed)

        sparse_ids = [doc_id for doc_id, _ in sparse_hits]
        dense_ids = [doc_id for doc_id, _ in dense_hits]
        fused = rrf_fuse([dense_ids, sparse_ids], weights=list(self.weights), k=self.rrf_k)

        sparse_rank = {doc_id: i for i, (doc_id, _) in enumerate(sparse_hits, 1)}
        dense_rank = {doc_id: i for i, (doc_id, _) in enumerate(dense_hits, 1)}

        pool: List[RetrievedChunk] = []
        for chunk_id, score in fused.items():
            base = self.chunks.get(chunk_id)
            if base is None:
                continue
            pool.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    text=base.text,
                    source=base.source,
                    title=base.title,
                    section=base.section,
                    score=score,
                    dense_rank=dense_rank.get(chunk_id),
                    sparse_rank=sparse_rank.get(chunk_id),
                    metadata=dict(base.metadata),
                )
            )
        pool = _rerank(query, pool, fused, corpus_size=corpus_size)[: max(top_k * 2, top_k)]

        self.last_debug = {
            "query": query,
            "sparse_top": sparse_ids[:5],
            "dense_top": dense_ids[:5],
            "allowed": len(allowed),
            "returned": [c.chunk_id for c in pool[:top_k]],
        }
        return pool[:top_k]

    def _dense_search(self, query: str, top_k: int, allowed: Set[str]) -> List[Tuple[str, float]]:
        qvec = self.embedder.embed(query)
        scored: List[Tuple[str, float]] = []
        for chunk_id, vec in self.vectors.items():
            if chunk_id not in allowed:
                continue
            scored.append((chunk_id, sum(a * b for a, b in zip(qvec, vec))))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]


class KnowledgeBase:
    """知识库：文档加载 → 切块 → 建索引 → 检索（含权限过滤）。

    权限模型刻意做得很简单但真实：**文档级 + 角色级**。
    ``visibility: public`` 全员可见；``visibility: [hr, finance]`` 仅这些角色可见。
    过滤发生在**检索阶段**而不是生成阶段 —— 没权限的片段根本不会进入上下文，
    这比"生成后再打码"安全得多。
    """

    def __init__(self, config: Config, docs_dir: Optional[str | Path] = None) -> None:
        rag = config.get("rag", {}) or {}
        self.config = config
        self.docs_dir = Path(docs_dir) if docs_dir else config.path_of("data/docs")
        self.chunk_size = int(rag.get("chunk_size", 360))
        self.chunk_overlap = int(rag.get("chunk_overlap", 50))
        self.rrf_k = int(rag.get("rrf_k", 60))
        self.weights = tuple(rag.get("rrf_weights", [1.0, 1.2]))[:2]
        self.embedder = LocalEmbedder(int(rag.get("embed_dim", 256)), int(rag.get("min_token_len", 2)))
        self.bm25 = BM25Index(min_len=int(rag.get("min_token_len", 2)))
        self.chunks: Dict[str, RetrievedChunk] = {}
        self.vectors: Dict[str, List[float]] = {}
        self.retriever = HybridRetriever(chunks=self.chunks, bm25=self.bm25, embedder=self.embedder, vectors=self.vectors)
        self.build_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # 构建索引
    # ------------------------------------------------------------------ #
    def build(self, force: bool = False) -> Dict[str, Any]:
        """从 ``data/docs/*.md`` 构建索引。

        每篇文档的 front matter 里可以声明 ``visibility`` 与 ``owner``，
        这些元数据会随 chunk 一起存下来，供权限过滤使用。
        """
        if self.chunks and not force:
            return self.build_stats or {"chunks": len(self.chunks), "cached": True}

        self.chunks.clear()
        self.vectors.clear()
        self.bm25 = BM25Index(min_len=self.embedder.min_len)

        files = sorted(self.docs_dir.rglob("*.md")) if self.docs_dir.exists() else []
        for path in files:
            raw = path.read_text(encoding="utf-8")
            meta, body = _split_front_matter(raw)
            source = path.relative_to(self.docs_dir).as_posix()
            meta.setdefault("visibility", ["public"])
            for chunk in chunk_markdown(
                body,
                source=source,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                metadata=meta,
            ):
                self.chunks[chunk.chunk_id] = chunk
                self.bm25.add(chunk.chunk_id, f"{chunk.section} {chunk.text}")
                self.vectors[chunk.chunk_id] = self.embedder.embed(f"{chunk.section} {chunk.text}")

        self.retriever = HybridRetriever(
            chunks=self.chunks,
            bm25=self.bm25,
            embedder=self.embedder,
            vectors=self.vectors,
            top_k_dense=int(self.config.get("rag.top_k_dense", 8)),
            top_k_sparse=int(self.config.get("rag.top_k_sparse", 8)),
            rrf_k=self.rrf_k,
            weights=self.weights,  # type: ignore[arg-type]
        )
        self.build_stats = {
            "documents": len(files),
            "chunks": len(self.chunks),
            "docs_dir": str(self.docs_dir),
        }
        return self.build_stats

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def allowed_ids(self, role: str = "public") -> Set[str]:
        """按角色算出可访问的 chunk 集合。

        规则：``visibility`` 含 ``public`` 或含该角色 → 可见。
        生产环境可以把这里换成接权限中心，接口不变。
        """
        role = (role or "public").strip().lower()
        allowed: Set[str] = set()
        for chunk_id, chunk in self.chunks.items():
            visibility = chunk.metadata.get("visibility", ["public"])
            if isinstance(visibility, str):
                visibility = [visibility]
            visible = {str(v).strip().lower() for v in visibility}
            if "public" in visible or role in visible or role == "public" and "public" in visible:
                allowed.add(chunk_id)
        return allowed

    def search(self, query: str, top_k: int = 4, role: str = "public") -> List[RetrievedChunk]:
        if not self.chunks:
            self.build()
        return self.retriever.search(query, top_k=top_k, allowed_ids=self.allowed_ids(role))

    def documents(self, role: str = "public") -> List[Dict[str, Any]]:
        """列出当前角色可见的文档（供"这个知识库里有什么"这类提问使用）。"""
        allowed = self.allowed_ids(role)
        seen: Dict[str, Dict[str, Any]] = {}
        for chunk_id in sorted(allowed):
            chunk = self.chunks[chunk_id]
            entry = seen.setdefault(
                chunk.source,
                {"source": chunk.source, "title": chunk.title, "sections": 0, "visibility": chunk.metadata.get("visibility")},
            )
            entry["sections"] += 1
        return list(seen.values())

    def restricted_matches(self, query: str, role: str = "public") -> List[str]:
        """找出"问题明确指向、但当前角色看不到"的文档。

        这个方法解决一个**很容易被忽略的安全问题**：
        如果只做检索过滤，那么无权限的用户提问"P4 绩效工资占比"时，
        系统会拿一份**别的**（比如差旅住宿标准）文档来回答 —— 既答错了问题，
        又暴露了"公司里确实有这份薪酬文档"的事实。

        判定标准刻意**从严**，必须同时满足：

        * 该受限文档自身的字符覆盖率 ≥ ``restricted_min_relevance``（默认 0.55）；
        * 且明显高于当前**可见**文档的最好覆盖水平（至少高出 0.15）。

        从严的理由：权限提示是一句"你不能看"，**误报会直接让用户以为系统坏了**。
        宁可漏报（退化为普通的"知识库中没有依据"），
        也不要因为沾了一两个词就喊"无权限"。
        """
        threshold = float(self.config.get("agent.restricted_min_relevance", 0.55))
        allowed = self.allowed_ids(role)

        visible_best = 0.0
        for chunk_id in allowed:
            chunk = self.chunks[chunk_id]
            coverage, _ = span_coverage(query, f"{chunk.section} {chunk.text}")
            visible_best = max(visible_best, coverage)

        hidden: List[str] = []
        for chunk_id, chunk in self.chunks.items():
            if chunk_id in allowed or chunk.source in hidden:
                continue
            coverage, _ = span_coverage(query, f"{chunk.section} {chunk.text}")
            if coverage >= threshold and coverage >= visible_best + 0.15:
                hidden.append(chunk.source)
        return hidden

    def stats(self) -> Dict[str, Any]:
        return {
            "documents": len({c.source for c in self.chunks.values()}),
            "chunks": len(self.chunks),
            "embedder": self.embedder.signature(),
            "bm25_terms": len(self.bm25.postings),
        }


def _split_front_matter(text: str) -> Tuple[Dict[str, Any], str]:
    """拆出 Markdown 的 YAML front matter（用于声明权限等元数据）。"""
    from .core import loads_yaml

    match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.S)
    if not match:
        return {}, text
    try:
        meta = loads_yaml(match.group(1)) or {}
    except Exception:  # noqa: BLE001 - 元数据坏了不该导致整篇文档不可用
        meta = {}
    return meta, text[match.end() :]
