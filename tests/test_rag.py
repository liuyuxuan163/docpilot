"""测试：检索层（分词 / BM25 / 本地嵌入 / RRF / 切块 / 权限隔离）。

检索是 RAG 系统的地基：**检索错了，生成层再强也救不回来**。
所以这一组测试覆盖的是"地基"的每个零件，而不是只看最终答案。
"""

from __future__ import annotations

import pytest

from docpilot.core import RetrievedChunk
from docpilot.matching import significant_spans
from docpilot.rag import (
    BM25Index,
    HybridRetriever,
    KnowledgeBase,
    LocalEmbedder,
    chunk_markdown,
    rrf_fuse,
    salient_terms,
    tokenize,
)


# --------------------------------------------------------------------------- #
# 分词
# --------------------------------------------------------------------------- #


class TestTokenizer:
    def test_chinese_produces_bigrams_and_trigrams(self):
        tokens = tokenize("年假结转", expand=False)
        assert "年假" in tokens
        assert "结转" in tokens
        # trigram 也要有，用来提升区分度
        assert "年假结" in tokens

    def test_stopwords_removed(self):
        tokens = tokenize("的了是 and the", expand=False)
        assert tokens == []

    def test_english_and_numbers_kept_as_words(self):
        tokens = tokenize("VPN 90 天", expand=False)
        assert "vpn" in tokens
        assert "90" in tokens

    def test_synonym_expansion_is_bidirectional(self):
        """同义词必须双向扩展。

        查询写口语（"年假"）要能召回书面语（"年休假"）；
        查询写书面语也要能召回口语 —— 只做单向会让一半的问法失效。
        """
        forward = tokenize("年假", expand=True)
        assert any("年休假" in token or "带薪年休假" in token for token in forward)
        backward = tokenize("带薪年休假", expand=True)
        assert "年假" in backward

    def test_empty_input(self):
        assert tokenize("") == []


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #


class TestBM25:
    def test_ranks_relevant_document_first(self):
        index = BM25Index()
        index.add("d1", "员工年假可以结转到下一年度，最多结转五天")
        index.add("d2", "差旅报销需在十五个工作日内提交")
        index.add("d3", "密码长度不少于十二位")
        hits = index.search("年假结转", top_k=2)
        assert hits and hits[0][0] == "d1"

    def test_idf_penalizes_ubiquitous_terms(self):
        index = BM25Index()
        for i in range(5):
            index.add(f"d{i}", "公司员工管理办法")
        index.add("rare", "公司年假结转规定")
        # 只出现在一篇里的词应当有更高 IDF
        assert index.idf("年假") > index.idf("公司")

    def test_remove_updates_index(self):
        index = BM25Index()
        index.add("d1", "年假结转")
        assert index.search("年假")
        index.remove("d1")
        assert index.search("年假") == []
        assert index.n_docs == 0

    def test_readd_same_id_does_not_duplicate(self):
        index = BM25Index()
        index.add("d1", "年假")
        index.add("d1", "年假结转")
        assert index.n_docs == 1

    def test_serialization_round_trip(self):
        index = BM25Index()
        index.add("d1", "年假结转下一年度")
        index.add("d2", "差旅报销十五工作日")
        restored = BM25Index.from_dict(index.to_dict())
        assert restored.n_docs == index.n_docs
        assert restored.search("年假结转") == index.search("年假结转")

    def test_allowed_filter_restricts_search(self):
        index = BM25Index()
        index.add("d1", "年假结转")
        index.add("d2", "年假申请流程")
        hits = index.search("年假", top_k=5, allowed={"d2"})
        assert [doc_id for doc_id, _ in hits] == ["d2"]


# --------------------------------------------------------------------------- #
# 本地嵌入
# --------------------------------------------------------------------------- #


class TestLocalEmbedder:
    def test_deterministic(self):
        embedder = LocalEmbedder(dim=64)
        assert embedder.embed("年假结转") == embedder.embed("年假结转")

    def test_normalized(self):
        import math

        embedder = LocalEmbedder(dim=64)
        vector = embedder.embed("年假结转下一年度")
        norm = math.sqrt(sum(v * v for v in vector))
        assert abs(norm - 1.0) < 1e-6

    def test_similar_text_scores_higher_than_unrelated(self):
        embedder = LocalEmbedder(dim=256)
        query = embedder.embed("年假可以结转到明年吗")
        near = embedder.embed("年假结转上限五天")
        far = embedder.embed("差旅住宿标准一线城市")
        dot = lambda a, b: sum(x * y for x, y in zip(a, b))  # noqa: E731
        assert dot(query, near) > dot(query, far)

    def test_empty_text_gives_zero_vector(self):
        vector = LocalEmbedder(dim=16).embed("")
        assert all(v == 0.0 for v in vector)

    def test_signature_changes_with_dim(self):
        assert LocalEmbedder(64).signature() != LocalEmbedder(128).signature()


# --------------------------------------------------------------------------- #
# RRF 融合
# --------------------------------------------------------------------------- #


class TestRRF:
    def test_document_in_both_lists_wins(self):
        fused = rrf_fuse([["a", "b"], ["a", "c"]])
        assert fused["a"] > fused["b"]
        assert fused["a"] > fused["c"]

    def test_weights_shift_ranking(self):
        fused = rrf_fuse([["a"], ["b"]], weights=[1.0, 5.0])
        assert fused["b"] > fused["a"]

    def test_rank_constant_dampens_top_rank(self):
        # k 越大，名次差异带来的分差越小
        sharp = rrf_fuse([["a", "b"]], k=1)
        smooth = rrf_fuse([["a", "b"]], k=1000)
        assert (sharp["a"] - sharp["b"]) > (smooth["a"] - smooth["b"])

    def test_empty_input(self):
        assert rrf_fuse([]) == {}
        assert rrf_fuse([[], []]) == {}


# --------------------------------------------------------------------------- #
# 切块
# --------------------------------------------------------------------------- #


class TestChunking:
    def test_keeps_section_path(self):
        text = "# 手册\n\n## 第三章\n\n### 年假额度\n\n年假按入职年限确定。\n"
        chunks = chunk_markdown(text, source="hr/leave.md", chunk_size=200)
        section = chunks[-1].section
        assert "年假额度" in section
        assert "手册" in section

    def test_citation_uses_leaf_section(self):
        text = "# 手册\n\n## 第三章\n\n### 年假额度\n\n年假按入职年限确定。\n"
        chunks = chunk_markdown(text, source="hr/leave.md", chunk_size=200)
        assert chunks[-1].cite() == "hr/leave.md#年假额度"

    def test_respects_chunk_size(self):
        body = "。".join(f"这是第{i}句用于测试切块大小的内容" for i in range(40))
        chunks = chunk_markdown(f"# 标题\n\n{body}\n", source="a.md", chunk_size=120)
        assert chunks
        assert all(len(chunk.text) <= 200 for chunk in chunks)

    def test_deduplicates_identical_content(self):
        text = "# 标题\n\n重复内容在这里出现。\n\n# 另一标题\n\n重复内容在这里出现。\n"
        chunks = chunk_markdown(text, source="a.md", chunk_size=500)
        texts = [chunk.text for chunk in chunks]
        assert len(texts) == len(set(texts))

    def test_metadata_is_propagated(self):
        chunks = chunk_markdown("# 标题\n\n正文内容。\n", source="a.md", metadata={"visibility": ["hr"]})
        assert chunks[0].metadata["visibility"] == ["hr"]

    def test_empty_document(self):
        assert chunk_markdown("", source="a.md") == []
        assert chunk_markdown("   \n\n  ", source="a.md") == []

    def test_chunk_ids_are_stable(self):
        text = "# 标题\n\n正文内容。\n"
        first = chunk_markdown(text, source="a.md")
        second = chunk_markdown(text, source="a.md")
        assert first[0].chunk_id == second[0].chunk_id


# --------------------------------------------------------------------------- #
# 知识库与权限
# --------------------------------------------------------------------------- #


class TestKnowledgeBase:
    def test_build_indexes_all_documents(self, kb: KnowledgeBase):
        stats = kb.stats()
        assert stats["documents"] == 8
        assert stats["chunks"] > 50

    def test_search_returns_citations(self, kb: KnowledgeBase):
        hits = kb.search("年假可以结转到明年吗", top_k=3, role="public")
        assert hits
        assert hits[0].source == "hr/leave-policy.md"
        assert hits[0].cite().startswith("hr/leave-policy.md#")

    def test_permission_hides_restricted_document(self, kb: KnowledgeBase):
        """权限隔离的核心断言：public 角色检索不到受限文档。"""
        hits = kb.search("绩效工资占比 年终奖系数", top_k=10, role="public")
        assert all(hit.source != "hr/compensation.md" for hit in hits)

    def test_permission_grants_for_authorized_role(self, kb: KnowledgeBase):
        hits = kb.search("绩效工资占比 年终奖系数", top_k=10, role="hr")
        assert any(hit.source == "hr/compensation.md" for hit in hits)

    def test_documents_are_filtered_by_role(self, kb: KnowledgeBase):
        public_docs = {doc["source"] for doc in kb.documents(role="public")}
        hr_docs = {doc["source"] for doc in kb.documents(role="hr")}
        assert "hr/compensation.md" not in public_docs
        assert "hr/compensation.md" in hr_docs
        assert public_docs < hr_docs  # public 可见集合应是 hr 的真子集

    def test_restricted_matches_detects_hidden_target(self, kb: KnowledgeBase):
        """问受限文档才能回答的问题时，要能识别出"目标在受限文档里"。"""
        restricted = kb.restricted_matches("P4 职级的绩效工资占比是多少？", role="public")
        assert "hr/compensation.md" in restricted

    def test_restricted_matches_is_not_over_eager(self, kb: KnowledgeBase):
        """权限提示必须从严 —— 误报"你无权访问"会让用户以为系统坏了。"""
        assert kb.restricted_matches("年假可以结转到明年吗？", role="public") == []
        assert kb.restricted_matches("差旅住宿标准是多少？", role="public") == []

    def test_restricted_matches_empty_for_authorized_role(self, kb: KnowledgeBase):
        assert kb.restricted_matches("绩效工资占比", role="hr") == []

    def test_allowed_ids_subset_relationship(self, kb: KnowledgeBase):
        assert kb.allowed_ids("public") < kb.allowed_ids("hr")

    def test_build_is_idempotent(self, kb: KnowledgeBase):
        before = kb.stats()
        kb.build()
        assert kb.stats() == before

    def test_search_on_empty_query(self, kb: KnowledgeBase):
        assert kb.search("", top_k=3) == []
        assert kb.search("   ", top_k=3) == []


# --------------------------------------------------------------------------- #
# 检索质量：用真实知识库跑"该命中哪一篇"的回归测试
# --------------------------------------------------------------------------- #


class TestRetrievalQuality:
    CASES = [
        ("年假没休完可以结转到明年吗", "hr/leave-policy.md", "public"),
        ("出差回来后多久必须提交报销单", "finance/expense-policy.md", "public"),
        ("密码多少位多久更换一次", "it/security-policy.md", "public"),
        ("VPN 连不上怎么排查", "it/it-faq.md", "public"),
        ("试用期多久转正评估什么时候发起", "hr/onboarding.md", "public"),
        ("代码评审多久要响应", "engineering/code-review.md", "public"),
        ("P4 职级绩效工资占比", "hr/compensation.md", "hr"),
        ("收受供应商礼品怎么处理", "legal/conduct-policy.md", "legal"),
    ]

    @pytest.mark.parametrize("query,expected_source,role", CASES)
    def test_top1_hits_expected_document(self, kb: KnowledgeBase, query, expected_source, role):
        hits = kb.search(query, top_k=4, role=role)
        assert hits, f"检索无结果：{query}"
        assert hits[0].source == expected_source, f"{query} → {hits[0].source}，期望 {expected_source}"


class TestRelevanceSignals:
    """``has_evidence`` 所依赖的两个信号必须有区分度，否则拒答判定会失灵。"""

    def test_in_scope_question_has_evidence(self, kb: KnowledgeBase):
        hits = kb.search("年假可以结转到明年吗", top_k=4, role="public")
        assert float(hits[0].metadata["relevance"]) >= 0.28
        assert int(hits[0].metadata["longest_span_chars"]) >= 2

    def test_out_of_scope_question_lacks_evidence(self, kb: KnowledgeBase):
        hits = kb.search("今天天气怎么样", top_k=4, role="public")
        assert not hits or (
            float(hits[0].metadata["relevance"]) < 0.28
            or int(hits[0].metadata["longest_span_chars"]) < 2
        )

    def test_longest_span_chars_present(self, kb: KnowledgeBase):
        hits = kb.search("差旅报销时效", top_k=2, role="public")
        assert all("longest_span_chars" in hit.metadata for hit in hits)
