"""文本匹配工具：不用分词器，也能算出"问题被文档覆盖了多少"。

问题背景
--------
中文没有空格，我们为了检索召回用了 bigram + trigram。但这带来一个副作用：
**"覆盖率"这个指标会被稀释得毫无意义**。

举例：问题是"P4 职级的绩效工资占比是多少？"，n-gram 切出来是
``占比是 / 工资占 / 效工资 / 是多少 / 比是多 / 的绩效`` ——
即使文档里明明白白写着"绩效工资占比 30%"，这些**跨词边界的组合**一个都匹配不上，
覆盖率算出来接近 0，系统于是错误地回答"没有依据"。

正确做法是**按字符测覆盖**：先取出问题里的"有意义字符"（去掉标点与虚词），
再看这些字符在文档里的**最大连续命中长度**。文档里写着"绩效工资占比"，
就能连续命中 6 个字符，覆盖率自然很高；而完全无关的问题最多命中 1~2 个零散字。

这个方案不需要分词器（零依赖）、不依赖词典，且对中英文都成立。
"""

from __future__ import annotations

import re
from typing import List, Tuple

__all__ = ["span_coverage", "significant_spans"]

_CJK = r"\u4e00-\u9fff"
# 无检索价值的虚词/疑问词（作为**分隔符**使用，不是"删除后拼接"）
#
# ⚠ 不要把"看起来像虚词、实际是词的一部分"的字放进来：
#   「办」（怎么办）、「久」（多久）、「否/能」（能否）都是实义字。
#   本项目踩过这个坑 —— 把「办」当虚词会把"出差住酒店超预算"切碎，命中率下降。
#
# 但「可」「以」这类能愿动词要放进来：「年假没休完**可以**结转到明年」
# 如果不断开，会得到 "年假没休完可" 这种带着半截虚词的长片段，
# 反而降低了匹配的精确度。
_FILLER = set("的了是在和与或对把被为以及等都也很会要有无我你他她它们这那"
              "什么怎么如何哪些多少吗呢吧啊请问可以")
_PUNCT_RE = re.compile(r"[\s，。、；：！？（）【】《》,.;:!?()\[\]{}<>\"'`~\-_/\\|+=*&^%$#@]+")


def significant_spans(text: str, min_len: int = 2) -> List[str]:
    """把问题切成"有意义片段"。

    **关键：遇到虚词要"断开"，而不是"删掉后把两边粘起来"。**
    早期实现是"去掉虚词字符后拼接"，于是
    ``"年假没休完可以结转到明年吗"`` → ``"年假没休完结转到明年"`` ——
    一个现实中不存在的字符串，最长连续命中反而变短，导致明明命中的问题被判"没依据"。

    正确做法：虚词与标点都作为**分隔符**，
    得到 ``["年假", "休完", "结转", "明年"]`` 这样的真实词组。
    """
    if not text:
        return []
    spans: List[str] = []
    buffer: List[str] = []
    for ch in text:
        is_sep = ch in _FILLER or bool(_PUNCT_RE.match(ch))
        if is_sep:
            if buffer:
                spans.append("".join(buffer))
                buffer = []
            continue
        buffer.append(ch)
    if buffer:
        spans.append("".join(buffer))

    out: List[str] = []
    for span in spans:
        if len(span) >= min_len or re.search(f"[{_CJK}]", span):
            out.append(span)
    return out


def span_coverage(query: str, document: str, min_span: int = 2) -> Tuple[float, float]:
    """计算问题在文档里的覆盖程度。

    Returns:
        ``(coverage, longest_span_ratio)``

        * ``coverage``：问题实义字符中，能在文档里找到的比例（0~1）
        * ``longest_span_ratio``：最长连续命中长度 / 问题实义字符总数（0~1）
          这个值对"只有一两个零散字命中"与"整句都对上"区分度很大。

    实现说明：因为问题片段通常很短（2~10 字），直接对每个片段在文档里找
    **最长公共子串长度**的代价可以接受（文档几百字，片段几个），
    而且比"子串完全包含"宽容得多 —— "绩效工资占比"能匹配"绩效工资基数占"这样的改写。
    """
    doc = document or ""
    spans = significant_spans(query, min_len=min_span)
    total_chars = sum(len(s) for s in spans)
    if total_chars == 0 or not doc:
        return 0.0, 0.0

    matched_chars = 0
    longest = 0
    for span in spans:
        best = _best_match(span, doc)
        matched_chars += best
        longest = max(longest, best)

    coverage = matched_chars / total_chars
    longest_ratio = longest / total_chars
    return round(min(1.0, coverage), 5), round(min(1.0, longest_ratio), 5)


def _best_match(span: str, document: str, depth: int = 0) -> int:
    """片段与文档的匹配长度：先找最长连续命中，再尝试递归切分取更优。

    为什么需要递归切分：整段问题常常被切成一个长片段
    （"出差住酒店超预算"），它在文档里几乎不可能原样出现，最长连续命中只有 2 个字符；
    但如果切成 "酒店" / "超预算" 分别去匹配，就能各命中 2 个与 3 个字符，
    显著提升区分度。

    这本质上是一个轻量分词：**在文档里找不到整段时，就试着把它切开**。
    深度限制为 2，避免组合爆炸。
    """
    if not span or not document:
        return 0
    if span in document:
        return len(span)
    best = _longest_common_substring(span, document)
    if depth >= 2 or len(span) <= 4:
        return best
    # 在若干候选切点里试切，取最好的两半之和（不超过原长）
    for cut in range(2, len(span) - 1):
        left, right = span[:cut], span[cut:]
        if len(left) < 2 or len(right) < 2:
            continue
        if left not in document and right not in document:
            continue
        candidate = _best_match(left, document, depth + 1) + _best_match(right, document, depth + 1)
        best = max(best, min(candidate, len(span)))
    return best


def _longest_common_substring(needle: str, haystack: str) -> int:
    """needle 在 haystack 中能匹配到的最长连续长度（不要求整段出现）。

    用滚动数组的动态规划，时间复杂度 O(len(needle) × len(haystack))。
    片段短（≤10 字）、文档几百字，单价可忽略；换来的是对改写的鲁棒性。
    """
    if not needle or not haystack:
        return 0
    # 快速路径：整段包含
    if needle in haystack:
        return len(needle)
    best = 0
    previous = [0] * (len(haystack) + 1)
    for i in range(1, len(needle) + 1):
        current = [0] * (len(haystack) + 1)
        for j in range(1, len(haystack) + 1):
            if needle[i - 1] == haystack[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
        # 已经等于片段长度就不必继续
        if best == len(needle):
            break
    return best
