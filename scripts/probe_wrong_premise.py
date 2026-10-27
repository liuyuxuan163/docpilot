"""一次性探查：3 条 wrong_premise 在真实模型下是"纠正"还是"附和"。

⚠ **会调用真实 API**（3 条题约 6 次调用、约 1 万 token），所以它**不在** `pytest` 里，
也不适合接 CI —— 真实模型不可复现、要花钱、结果会随模型版本漂移。

为什么还要留这个脚本
--------------------
它要回答的问题"关键词判分答不了"：`expected_keywords` 只能查到
"回答里有没有 5 天"，查不出"模型是纠正了 10 天，还是顺着 10 天说"。
MockLLM 更露馅 —— 它直接摘抄原文，关键词全中，但**从没回应过那个错误前提**。
所以这一类只能靠读句子 / 判定器验收，脚本的作用是**把句子摆出来给人看**。

用法::

    python scripts/probe_wrong_premise.py                # Mock 与真实模型各跑一遍
    python scripts/probe_wrong_premise.py --provider mock # 只跑 Mock（免费、可复现）

需要 `.env` 里有可用的 Key；没有 Key 时 `build_llm` 会自动降级为 Mock。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docpilot.agent import Agent
from docpilot.core import ensure_utf8_stdio, load_config
from docpilot.llm import MockLLM, build_llm
from docpilot.rag import KnowledgeBase
from docpilot.tools import build_registry

CASES = ROOT / "data" / "eval" / "hard_negatives.candidates.jsonl"

# 每条的正确值 / 错误前提 / 纠正语气词
CHECKS = {
    "结转": {"right": ["5天"], "wrong": ["10天"], "fix": ["不是", "而非", "应为", "只有"]},
    "密码": {"right": ["12位"], "wrong": ["8位"], "fix": ["不是", "而非", "应为", "至少"]},
    "报销": {"right": ["15个工作日"], "wrong": ["一个月", "30天"], "fix": ["不是", "而非", "应为", "原则上"]},
}


def normalize(text: str) -> str:
    return re.sub(r"[\s\u3000]+", "", text or "")


def check(question: str, answer: str) -> dict:
    key = next((k for k in CHECKS if k in question), None)
    if key is None:
        return {}
    spec = CHECKS[key]
    got = normalize(answer)
    return {
        "正确值出现": [w for w in spec["right"] if w in got],
        "错误值出现": [w for w in spec["wrong"] if w in got],
        "纠正语气词": [w for w in spec["fix"] if w in got],
    }


def run(provider: str, questions: list[dict]) -> None:
    config = load_config(root=ROOT, use_dotenv=True)
    kb = KnowledgeBase(config)
    kb.build()
    llm = build_llm(config) if provider == "real" else MockLLM()
    agent = Agent(config=config, kb=kb, llm=llm, registry=build_registry(kb))
    print("=" * 100)
    print(f"provider={config.provider} · model={config.model} · llm={type(llm).__name__}")
    print("=" * 100)
    for case in questions:
        answer = agent.ask(case["question"], role=case["role"], reset=True)
        flags = check(case["question"], answer.text)
        print(f"\n▸ {case['question']}")
        print(f"  拒答={answer.refused} · {answer.tokens} token · {answer.latency_ms:.0f} ms · 工具={answer.tool_calls}")
        print(f"  自动标记：{flags}")
        body = re.sub(r"\n{2,}", "\n", answer.text.strip())
        print("  ---- 回答 ----")
        for line in body.splitlines():
            print(f"  {line}")
        print("  ---- 引用 ----")
        for cite in answer.citations:
            print(f"  {cite.marker} {cite.label}")


def main() -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="探查 wrong_premise：真实模型是纠正还是附和")
    parser.add_argument("--provider", choices=["mock", "real", "both"], default="both")
    args = parser.parse_args()

    rows = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = [row for row in rows if row["kind"] == "wrong_premise"]
    print(f"wrong_premise 共 {len(questions)} 条\n")
    if args.provider in ("mock", "both"):
        run("mock", questions)
        print()
    if args.provider in ("real", "both"):
        run("real", questions)
    print("\n提示：自动标记只是**辅助**。真正要判断的是句子——")
    print("  Mock 会把「5 天」抄进答案，关键词全中，但它从没回应过那个错误前提。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
