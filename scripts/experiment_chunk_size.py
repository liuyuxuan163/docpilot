"""实验 7：对比不同 chunk_size 对检索指标的影响。

用法::

    python scripts/experiment_chunk_size.py

它会依次把 ``rag.chunk_size`` 设为 360 / 150 / 800，
每次重建索引并跑评测，最后打印对比表，并**自动恢复原值**。

这个脚本存在的意义：**把"改配置跑一次"变成可复现的实验**。
手工改 YAML 容易忘恢复，也可能漏掉"重建索引"这一步。
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "configs" / "docpilot.yaml"
SIZES = [360, 150, 800]           # 基线，太小，太大
METRICS = ["composite", "retrieval_hit", "citation_valid", "answer_coverage", "over_refusal_rate"]


def set_chunk_size(size: int) -> None:
    """改写配置里的 chunk_size（保留其他内容不变）。"""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    new_text, count = re.subn(r"chunk_size: \d+", f"chunk_size: {size}", text, count=1)
    if count == 0:
        raise RuntimeError("配置里没找到 chunk_size，请确认 configs/docpilot.yaml")
    CONFIG_PATH.write_text(new_text, encoding="utf-8")


def current_chunk_size() -> int:
    match = re.search(r"chunk_size: (\d+)", CONFIG_PATH.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else -1


def probe(size: int) -> dict:
    """在指定 chunk_size 下构建索引并跑评测，返回指标与索引规模。"""
    set_chunk_size(size)

    # 每次都在新进程里跑，确保索引按新配置重建（不用管磁盘缓存）
    from docpilot.agent import Agent
    from docpilot.core import load_config
    from docpilot.eval import load_cases, run_eval
    from docpilot.llm import MockLLM
    from docpilot.rag import KnowledgeBase
    from docpilot.tools import build_registry

    config = load_config(root=ROOT)
    kb = KnowledgeBase(config)
    stats = kb.build()

    cases = load_cases(config.path_of("data/eval/cases.jsonl"))
    agent = Agent(config=config, kb=kb, llm=MockLLM(), registry=build_registry(kb))

    buffer = io.StringIO()
    with redirect_stdout(buffer):        # eval 内部可能打印，别污染输出
        report = run_eval(agent, cases, gates=config.get("eval.gates", {}))

    return {
        "size": size,
        "chunks": stats["chunks"],
        "documents": stats["documents"],
        "metrics": {m: report.summary.get(m) for m in METRICS},
        "gate_passed": report.passed,
    }


def main() -> int:
    original = current_chunk_size()
    print(f"当前 chunk_size = {original}（实验结束后会恢复）\n")

    results = []
    try:
        for size in SIZES:
            print(f"--- 正在测试 chunk_size = {size} ---")
            result = probe(size)
            results.append(result)
            print(f"    索引：{result['documents']} 篇文档 / {result['chunks']} 个片段")
            print(f"    指标：{result['metrics']}")
            print(f"    门禁：{'通过' if result['gate_passed'] else '未通过'}\n")
    finally:
        set_chunk_size(original)
        print(f"已恢复 chunk_size = {original}")

    # 对比表
    print("\n" + "=" * 76)
    print("对比结果")
    print("=" * 76)
    header = f"{'chunk_size':>10} {'片段数':>8} " + " ".join(f"{m:>16}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for result in results:
        row = f"{result['size']:>10} {result['chunks']:>8} "
        row += " ".join(f"{result['metrics'][m]!s:>16}" for m in METRICS)
        print(row)

    print("\n观察要点：")
    print("  · chunk 太小（150）→ 索引片段数激增，单块上下文不足，answer_coverage 可能下降")
    print("  · chunk 太大（800）→ 片段数减少，相关性被稀释，retrieval_hit 可能下降")
    print("  · 是否出现 over_refusal_rate 上升？那说明拒答判定被切块粒度影响了")

    out = ROOT / "evals" / "reports" / "chunk-size-experiment.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始数据已写入 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
