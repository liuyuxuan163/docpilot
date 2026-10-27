"""pytest 夹具：统一把临时目录放到项目内，避免 Windows 系统临时目录的权限问题。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 项目内临时目录（某些受限环境下系统 temp 不可写，统一重定向）
RUNTIME = ROOT / ".runtime" / "tests"
RUNTIME.mkdir(parents=True, exist_ok=True)

from docpilot.core import Config, DEFAULT_CONFIG, load_config  # noqa: E402
from docpilot.llm import MockLLM  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture()
def config() -> Config:
    """项目默认配置。

    **刻意不读 ``.env``**：开发者本机可能填着真实 Key，一旦漏进测试，
    `test_api_key_resolution_order` 这类"没有 Key 时应当是 None"的断言会莫名失败，
    而且只在自己的机器上失败 —— 那是最难查的一类测试。
    专门测 ``.env`` 行为的用例在 ``tests/test_dotenv.py`` 里用临时目录自己造。
    """
    return load_config(root=ROOT, use_dotenv=False)


@pytest.fixture()
def kb(config: Config) -> KnowledgeBase:
    """已构建索引的知识库（会话内复用同一实例，构建只做一次）。"""
    knowledge = KnowledgeBase(config)
    knowledge.build()
    return knowledge


@pytest.fixture()
def mock_llm(config: Config) -> MockLLM:
    return MockLLM(min_score=float(config.get("agent.min_evidence_relevance", 0.28)))


@pytest.fixture()
def workdir() -> Path:
    """可写的工作目录（凡是要落盘的测试都用它）。

    两个**踩过的坑**都体现在这几行里：

    1. **不要覆盖 pytest 的 ``tmp_path``**。受限环境里系统 temp 目录会报
       ``PermissionError: [WinError 5]``，而 pytest 的临时目录机制会去遍历
       它自己的 base temp 目录，一样失败；覆盖它还会打断
       ``tmp_path_factory`` 的依赖链。
    2. **不要用 ``tempfile.mkdtemp``**。它创建的目录权限是 ``0o700``，
       在受限的 Windows 沙箱令牌下**无法在其中再建子目录**
       （``os.mkdir`` 报 WinError 5）。

    所以这里用"常规 mkdir + 唯一名字"在项目内建目录，绕开上述两个问题。
    """
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for index in range(1000):
        candidate = RUNTIME / f"case-{os.getpid()}-{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
    raise RuntimeError("无法创建测试工作目录")


@pytest.fixture(autouse=True)
def _isolate_temp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 temp 相关环境变量指向项目内目录，避免测试过程中碰到系统 temp。"""
    monkeypatch.setenv("TMPDIR", str(RUNTIME))
    monkeypatch.setenv("TEMP", str(RUNTIME))
    monkeypatch.setenv("TMP", str(RUNTIME))


@pytest.fixture(autouse=True)
def _never_call_a_real_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """测试绝不联网：把 provider 钉成 mock。

    真实环境变量优先级高于 ``.env``，所以这一行能盖住本机 ``.env`` 里写的
    ``DOCPILOT_PROVIDER=deepseek`` —— 否则某天你在 ``.env`` 里切到真实模型，
    整个测试套件就会开始打真实接口：花钱、慢、还会偶发失败。
    需要测其它 provider 的用例自己再 ``monkeypatch.setenv`` 即可（后设的生效）。
    """
    monkeypatch.setenv("DOCPILOT_PROVIDER", "mock")


@pytest.fixture()
def default_config_dict():
    return DEFAULT_CONFIG
