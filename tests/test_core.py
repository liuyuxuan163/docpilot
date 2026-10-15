"""测试：核心类型、配置解析、mini-YAML、文本匹配。

这一组测试的重点是**边界与坑**，而不是覆盖率数字：
每个用例基本都对应开发过程中真实踩到过的一个问题。
"""

from __future__ import annotations

import pytest

from docpilot.core import (
    DEFAULT_CONFIG,
    Answer,
    Config,
    ConfigError,
    Message,
    RetrievedChunk,
    ToolCall,
    load_config,
    loads_yaml,
    short_hash,
)
from docpilot.matching import significant_spans, span_coverage


# --------------------------------------------------------------------------- #
# mini-YAML：装了 PyYAML 也要测内置实现（两条路径都得能用）
# --------------------------------------------------------------------------- #


class TestMiniYaml:
    """全部用 force_mini=True 走内置实现。

    否则装了 PyYAML 的环境会走 PyYAML，内置解析器与它的人性化报错就永远测不到。
    """

    @staticmethod
    def parse(text: str):
        return loads_yaml(text, force_mini=True)

    def test_nested_mapping_and_scalars(self):
        data = self.parse(
            """
            agent:
              max_steps: 6
              name: pilot
              debug: false
              ratio: 0.25
            """
        )
        assert data["agent"]["max_steps"] == 6
        assert data["agent"]["name"] == "pilot"
        assert data["agent"]["debug"] is False
        assert data["agent"]["ratio"] == 0.25

    def test_inline_list_and_mapping(self):
        data = self.parse(
            """
            gates: [a, b, c]
            weights: [1.0, 1.2]
            empty: []
            obj: {x: 1, y: two}
            """
        )
        assert data["gates"] == ["a", "b", "c"]
        assert data["weights"] == [1.0, 1.2]
        assert data["empty"] == []
        assert data["obj"] == {"x": 1, "y": "two"}

    def test_block_list_of_mappings(self):
        data = self.parse(
            """
            cases:
              - id: one
                role: public
              - id: two
                role: hr
            """
        )
        assert data["cases"] == [{"id": "one", "role": "public"}, {"id": "two", "role": "hr"}]

    def test_comments_and_quotes(self):
        data = self.parse(
            """
            # 整行注释
            a: "含 # 的值"      # 行尾注释
            b: '单引号'
            c: value # 注释
            """
        )
        assert data["a"] == "含 # 的值"
        assert data["b"] == "单引号"
        assert data["c"] == "value"

    def test_tab_indent_is_rejected(self):
        # Tab 缩进在 YAML 里是错误，必须给出可读报错而不是静默错解析
        with pytest.raises(ConfigError, match="Tab"):
            self.parse("a:\n\tb: 1\n")

    def test_missing_colon_reports_error(self):
        with pytest.raises(ConfigError, match="缺少"):
            self.parse("a:\n  just_a_value_without_colon\n")

    def test_block_scalar(self):
        data = self.parse(
            """
            text: |
              line1
              line2
            """
        )
        assert "line1" in data["text"] and "line2" in data["text"]

    def test_pyyaml_path_also_works(self):
        # 默认路径（装了 PyYAML 时走 PyYAML）同样要能解析
        assert loads_yaml("a: 1\n")["a"] == 1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


class TestConfig:
    def test_dot_access_and_defaults(self, config: Config):
        assert config.get("agent.max_steps") == 6
        assert config.get("does.not.exist") is None
        assert config.get("does.not.exist", "fallback") == "fallback"

    def test_local_override_merges_deeply(self, workdir):
        root = workdir / "proj"
        (root / "configs").mkdir(parents=True)
        (root / "configs" / "docpilot.yaml").write_text(
            "agent:\n  max_steps: 3\nrag:\n  chunk_size: 100\n", encoding="utf-8"
        )
        (root / "configs" / "docpilot.local.yaml").write_text("agent:\n  max_steps: 9\n", encoding="utf-8")
        loaded = load_config(root=root)
        # 覆盖生效，且未覆盖的字段保留
        assert loaded.get("agent.max_steps") == 9
        assert loaded.get("rag.chunk_size") == 100
        # 默认值仍在
        assert loaded.get("rag.rrf_k") == DEFAULT_CONFIG["rag"]["rrf_k"]

    def test_env_overrides_provider_and_model(self, config: Config, monkeypatch):
        monkeypatch.setenv("DOCPILOT_PROVIDER", "deepseek")
        monkeypatch.setenv("DOCPILOT_MODEL", "my-model")
        assert config.provider == "deepseek"
        assert config.model == "my-model"

    def test_api_key_resolution_order(self, config: Config, monkeypatch):
        for name in ("DOCPILOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert config.api_key() is None
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        assert config.api_key() == "sk-openai"
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
        assert config.api_key() == "sk-deepseek"

    def test_base_url_by_provider(self, config: Config, monkeypatch):
        monkeypatch.delenv("DOCPILOT_BASE_URL", raising=False)
        monkeypatch.setenv("DOCPILOT_PROVIDER", "deepseek")
        assert config.base_url() == "https://api.deepseek.com/v1"

    def test_path_of_resolves_against_root(self, config: Config):
        assert config.path_of("data/docs").is_absolute()


# --------------------------------------------------------------------------- #
# 数据契约
# --------------------------------------------------------------------------- #


class TestDataContracts:
    def test_message_to_api_with_tool_calls(self):
        call = ToolCall(name="search_docs", arguments={"query": "年假"})
        message = Message(role="assistant", content="", tool_calls=[call])
        payload = message.to_api()
        assert payload["role"] == "assistant"
        assert payload["tool_calls"][0]["function"]["name"] == "search_docs"
        # 参数必须被序列化成 JSON 字符串（OpenAI 协议要求）
        assert '"年假"' in payload["tool_calls"][0]["function"]["arguments"]

    def test_tool_message_carries_call_id(self):
        message = Message(role="tool", content="{}", name="search_docs", tool_call_id="call_1")
        assert message.to_api()["tool_call_id"] == "call_1"

    def test_citation_uses_leaf_section(self):
        # 引用标签必须只取叶子小节：完整标题路径会长到没法读
        chunk = RetrievedChunk(
            chunk_id="c1", text="x", source="hr/leave.md", section="员工手册 > 第三章 > 年假额度"
        )
        assert chunk.cite() == "hr/leave.md#年假额度"

    def test_citation_without_section_falls_back_to_source(self):
        chunk = RetrievedChunk(chunk_id="c1", text="x", source="hr/leave.md", section="")
        assert chunk.cite() == "hr/leave.md"

    def test_answer_markdown_marks_refusal(self):
        answer = Answer(question="q", text="没有依据", refused=True, refusal_reason="知识库无依据")
        rendered = answer.to_markdown()
        assert "无法回答" in rendered
        assert "知识库无依据" in rendered

    def test_short_hash_is_deterministic_and_sensitive(self):
        assert short_hash("a", "b") == short_hash("a", "b")
        assert short_hash("a", "b") != short_hash("a", "c")

    def test_tool_call_ids_are_unique(self):
        assert ToolCall(name="x").id != ToolCall(name="x").id


# --------------------------------------------------------------------------- #
# 文本匹配：本项目最核心也最容易写错的一小块
# --------------------------------------------------------------------------- #


class TestMatching:
    def test_spans_are_split_at_filler_words(self):
        """虚词必须作为**分隔符**，不能"删掉后把两边粘起来"。

        早期实现会产出 "年假没休完结转到明年" 这种现实中不存在的字符串，
        导致最长连续命中变短、明明命中的问题被判"没有依据"。
        """
        spans = significant_spans("年假没休完可以结转到明年吗")
        # 必须按「可以」这个虚词断开，得到两个真实词组
        assert spans == ["年假没休完", "结转到明年"], spans
        # 绝不能出现跨虚词粘连出来的长串
        assert all("可以" not in span for span in spans)
        assert not any("没休完可以" in span for span in spans)

    def test_empty_input(self):
        assert significant_spans("") == []
        assert span_coverage("", "anything") == (0.0, 0.0)
        assert span_coverage("问题", "") == (0.0, 0.0)

    def test_identical_text_scores_high(self):
        coverage, longest = span_coverage("年假怎么折算", "年假怎么折算按自然月")
        assert coverage > 0.8
        # longest_ratio 的分母是"问题实义字符总数"，而问题被切成 年假 / 折算 两段，
        # 单段最长只有 2 字 → 2/4 = 0.5。这是该指标的定义，不是缺陷。
        assert longest >= 0.5

    def test_unrelated_text_scores_low(self):
        coverage, longest = span_coverage("今天天气怎么样", "员工薪酬由基本工资与绩效工资构成")
        assert coverage < 0.5
        assert longest < 0.5

    def test_partial_match_is_credited(self):
        # 改写过的表述也应拿到部分分（"绩效工资占比" vs "绩效工资基数占"）
        coverage, _ = span_coverage("绩效工资占比是多少", "绩效工资基数占月度总薪酬的比例")
        assert coverage > 0.4

    def test_recursive_splitting_beats_plain_lcs(self):
        """整段找不到时要能切开找子短语 —— 且必须**优于**朴素的"最长连续命中"。

        选例说明：``"报销时效流程"`` 整段在文档里不存在，
        朴素最长连续命中只有 2 个字符（"报销"）；
        切分成 ``报销时效`` / ``流程`` 后可以命中 ``报销时效``（4 字），
        提升一倍。这正是递归切分的价值所在。
        """
        document = "报销时效与流程：出差结束后 15 个工作日内提交"
        coverage, longest_ratio = span_coverage("报销时效流程", document)
        # 朴素做法只有 2/6；递归切分后应明显更高
        assert longest_ratio >= 0.5, f"递归切分未生效：longest_ratio={longest_ratio}"
        assert coverage > 0.5
