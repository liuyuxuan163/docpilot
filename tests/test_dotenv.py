"""测试：``.env`` 支持（解析器 + 四层优先级 + 与测试隔离）。

为什么这些用例值得写
--------------------
``.env`` 是**唯一一处"文件内容会改变进程行为"**的地方，而且它装着真实 Key。
两类事故都很难查：

1. **解析器太宽松或太严格** —— 多行值、行内注释、带引号、值里带 ``=``，
   任何一种处理错，表现都是"Key 明明填了却说没有"。
2. **优先级搞反** —— 如果 ``.env`` 压过真实环境变量，
   那 CI 里 ``env: DOCPILOT_PROVIDER: mock`` 就会被某个人本地的
   ``.env`` 悄悄盖掉，测试开始打真实接口。这类问题只在别人机器上复现。

还有一个必须锁住的安全性质：**测试套件永不联网**。见
``test_provider_mock_ignores_available_key``。
"""

from __future__ import annotations

import json

import pytest

from docpilot.core import (
    DEFAULT_CONFIG,
    Config,
    load_config,
    parse_dotenv,
    read_dotenv,
)
from docpilot.llm import MockLLM, build_llm

PROVIDER = "DOCPILOT_PROVIDER"


def make_config(dotenv=None, cli=None) -> Config:
    """直接构造 Config（不经 load_config），把关注点隔离到"层"上。"""
    return Config(json.loads(json.dumps(DEFAULT_CONFIG)), dotenv=dotenv, cli=cli)


# --------------------------------------------------------------------------- #
# 解析器
# --------------------------------------------------------------------------- #


class TestParseDotenv:
    def test_basic_key_value(self):
        assert parse_dotenv("DOCPILOT_PROVIDER=deepseek\n") == {PROVIDER: "deepseek"}

    def test_comments_and_blank_lines_ignored(self):
        text = "# 注释\n\n  # 缩进的注释\nA=1\n"
        assert parse_dotenv(text) == {"A": "1"}

    def test_export_prefix_and_quotes(self):
        text = "export A=1\nB='带 空格 的值'\nC=\"x#y\"\n"
        assert parse_dotenv(text) == {"A": "1", "B": "带 空格 的值", "C": "x#y"}

    def test_inline_comment_stripped_only_when_unquoted(self):
        # 未加引号："空白 + #" 之后是注释；加了引号则 # 属于值本身
        assert parse_dotenv("A=deepseek  # 备注") == {"A": "deepseek"}
        assert parse_dotenv('A="deepseek # 不是备注"') == {"A": "deepseek # 不是备注"}

    def test_value_may_contain_equals(self):
        # Key 里偶尔出现 '='，只能按第一个 '=' 切分
        assert parse_dotenv("A=sk-abc==def") == {"A": "sk-abc==def"}

    def test_invalid_lines_ignored(self):
        text = "没有等号的行\n=no_key\n1BAD=x\nGOOD=1\n"
        assert parse_dotenv(text) == {"GOOD": "1"}

    def test_empty_value_is_kept_as_empty_string(self):
        # 这是"显式清空"的表达，交给 _setting 决定当没设（见下面的用例）
        assert parse_dotenv("A=") == {"A": ""}


class TestReadDotenv:
    def test_missing_file_returns_empty(self, workdir):
        assert read_dotenv(workdir / "不存在.env") == {}

    def test_bom_is_tolerated(self, workdir):
        """Windows 记事本会写 BOM；不处理的话第一个键名会带 ``\\ufeff``。"""
        path = workdir / ".env"
        path.write_text("\ufeffA=1\n", encoding="utf-8")
        assert read_dotenv(path) == {"A": "1"}

    def test_directory_is_not_a_file(self, workdir):
        assert read_dotenv(workdir) == {}


# --------------------------------------------------------------------------- #
# 优先级：命令行 > 环境变量 > .env > YAML
# --------------------------------------------------------------------------- #


class TestPrecedence:
    def test_dotenv_overrides_yaml(self, monkeypatch):
        monkeypatch.delenv(PROVIDER, raising=False)
        config = make_config(dotenv={PROVIDER: "deepseek"})
        assert config.get("llm.provider") == "mock", "YAML 默认值应当是 mock"
        assert config.provider == "deepseek"
        assert config.source_of(PROVIDER) == ".env"

    def test_real_env_overrides_dotenv(self, monkeypatch):
        monkeypatch.setenv(PROVIDER, "openai")
        config = make_config(dotenv={PROVIDER: "deepseek"})
        assert config.provider == "openai"
        assert config.source_of(PROVIDER) == "环境变量"

    def test_cli_overrides_everything(self, monkeypatch):
        monkeypatch.setenv(PROVIDER, "openai")
        config = make_config(dotenv={PROVIDER: "deepseek"}, cli={PROVIDER: "mock"})
        assert config.provider == "mock"
        assert config.source_of(PROVIDER) == "命令行"

    def test_empty_value_treated_as_unset(self, monkeypatch):
        """``.env`` 里写 ``DOCPILOT_PROVIDER=`` 应当等于没写，而不是把配置清空。"""
        monkeypatch.delenv(PROVIDER, raising=False)
        config = make_config(dotenv={PROVIDER: ""})
        assert config.provider == "mock", "空值必须回落到 YAML 默认"
        assert config.source_of(PROVIDER) == "配置/默认值"

    def test_cli_none_does_not_clear(self, monkeypatch):
        monkeypatch.setenv(PROVIDER, "deepseek")
        config = make_config(cli={"DOCPILOT_PROVIDER": None})
        assert config.provider == "deepseek"


# --------------------------------------------------------------------------- #
# Key 解析与"来源"报告
# --------------------------------------------------------------------------- #


class TestApiKey:
    def test_key_priority_and_source(self, monkeypatch):
        for name in ("DOCPILOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        config = make_config(dotenv={"OPENAI_API_KEY": "sk-openai", "DEEPSEEK_API_KEY": "sk-deepseek"})
        assert config.api_key() == "sk-deepseek"
        assert "DEEPSEEK_API_KEY" in config.api_key_source()
        assert ".env" in config.api_key_source()

        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")  # 环境变量优先于 .env
        assert config.api_key() == "sk-env"
        assert config.api_key_source() == "OPENAI_API_KEY（环境变量）"

    def test_api_key_source_never_leaks_the_value(self, monkeypatch):
        for name in ("DOCPILOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        config = make_config(dotenv={"DEEPSEEK_API_KEY": "sk-super-secret"})
        assert "sk-super-secret" not in config.api_key_source()
        assert "sk-super-secret" not in config.describe()

    def test_no_key_source(self, monkeypatch):
        for name in ("DOCPILOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert make_config().api_key_source() == "未配置"


# --------------------------------------------------------------------------- #
# 与 LLM 装配的关系（安全性质）
# --------------------------------------------------------------------------- #


class TestBuildLLM:
    def test_provider_mock_ignores_available_key(self, monkeypatch):
        """provider=mock 时即使 Key 就在 .env 里，也绝不构造真实客户端。

        这是"默认离线"的最后一道保险：本地有 Key 是常态，
        但默认不该因此联网。
        """
        monkeypatch.delenv(PROVIDER, raising=False)
        config = make_config(dotenv={PROVIDER: "mock", "DEEPSEEK_API_KEY": "sk-x"})
        assert isinstance(build_llm(config), MockLLM)

    def test_real_provider_without_key_degrades_to_mock(self, monkeypatch):
        for name in ("DOCPILOT_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setenv(PROVIDER, "deepseek")
        assert isinstance(build_llm(make_config()), MockLLM)


# --------------------------------------------------------------------------- #
# load_config 的文件读取
# --------------------------------------------------------------------------- #


class TestLoadConfigReadsFile:
    def test_env_file_is_loaded_from_root(self, workdir, monkeypatch):
        monkeypatch.delenv(PROVIDER, raising=False)
        (workdir / ".env").write_text("DOCPILOT_PROVIDER=deepseek\nDEEPSEEK_API_KEY=sk-x\n", encoding="utf-8")
        config = load_config(root=workdir)
        assert config.provider == "deepseek"
        assert config.api_key() == "sk-x"
        assert config.dotenv_keys() == ["DEEPSEEK_API_KEY", "DOCPILOT_PROVIDER"]

    def test_can_be_disabled(self, workdir, monkeypatch):
        """测试与 CI 用 ``use_dotenv=False`` 保证与本机 .env 隔离。"""
        monkeypatch.delenv(PROVIDER, raising=False)
        (workdir / ".env").write_text("DOCPILOT_PROVIDER=deepseek\n", encoding="utf-8")
        config = load_config(root=workdir, use_dotenv=False)
        assert config.provider == "mock"
        assert config.dotenv_keys() == []

    def test_cli_layer_passed_through_load_config(self, workdir, monkeypatch):
        monkeypatch.delenv(PROVIDER, raising=False)
        (workdir / ".env").write_text("DOCPILOT_PROVIDER=deepseek\n", encoding="utf-8")
        config = load_config(root=workdir, cli={PROVIDER: "openai"})
        assert config.provider == "openai"
        assert config.source_of(PROVIDER) == "命令行"
