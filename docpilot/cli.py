"""命令行入口：``python -m docpilot <命令>``。

命令设计原则：**每条命令都要能直接用在面试演示或 CI 里**，
所以既有给人看的漂亮输出，也有给脚本用的 ``--json``。

::

    python -m docpilot demo                          # 一键演示 5 个典型场景
    python -m docpilot ask "年假能结转到明年吗"        # 单次问答
    python -m docpilot ask "那最多多少天" --history "年假能结转到明年吗"
    python -m docpilot ingest                        # 重建索引并打印统计
    python -m docpilot eval                          # 跑评测 + 质量门禁
    python -m docpilot web                           # 启动 Web 演示
    python -m docpilot info                          # 环境与配置自检
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .agent import Agent
from .core import Config, DocPilotError, load_config
from .eval import load_cases, run_eval
from .llm import build_llm
from .rag import KnowledgeBase

EXIT_OK, EXIT_FAIL, EXIT_GATE = 0, 1, 2


def _build_agent(config: Config) -> Agent:
    kb = KnowledgeBase(config)
    stats = kb.build()
    return Agent(config=config, kb=kb, llm=build_llm(config))


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(payload)


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #


def cmd_demo(args: argparse.Namespace) -> int:
    """跑 5 个典型场景，覆盖这个项目的全部卖点。"""
    config = load_config(root=args.root)
    agent = _build_agent(config)

    scenarios: List[Dict[str, Any]] = [
        {"tag": "① 常规事实问答", "question": "年假没休完可以结转到明年吗？最多结转多少天？", "role": "public"},
        {"tag": "② 口语化提问", "question": "出差住酒店超预算了怎么办？", "role": "public"},
        {
            "tag": "③ 多轮追问（自动改写问题）",
            "question": "那最多能结转到下一年多少天？",
            "role": "public",
            "history": ["年假没休完可以结转到明年吗？"],
        },
        {"tag": "④ 超纲提问（应当拒答，不编造）", "question": "公司的股票代码是多少？", "role": "public"},
        {"tag": "⑤ 权限隔离（public 角色看不到薪酬文档）", "question": "P4 职级的绩效工资占比是多少？", "role": "public"},
        {"tag": "⑥ 权限内提问（hr 角色可以看到同一份文档）", "question": "P4 职级的绩效工资占比是多少？", "role": "hr"},
    ]

    rows: List[Dict[str, Any]] = []
    print("DocPilot 演示：企业知识库问答")
    print("=" * 78)
    for item in scenarios:
        answer = agent.ask(item["question"], role=item["role"], history=item.get("history"), reset=True)
        rows.append(
            {
                "scenario": item["tag"],
                "question": item["question"],
                "role": item["role"],
                "refused": answer.refused,
                "citations": answer.citation_labels,
                "answer": answer.text[:400],
                "tokens": answer.tokens,
                "latency_ms": round(answer.latency_ms, 1),
            }
        )
        print(f"\n▸ {item['tag']}　(角色: {item['role']})")
        print(f"  问：{item['question']}")
        if answer.rewritten_question and answer.rewritten_question != item["question"]:
            print(f"  改写后：{answer.rewritten_question}")
        if answer.refused:
            print(f"  答（拒答）：{answer.text[:120]}")
        else:
            body = " ".join(answer.text.split())
            print(f"  答：{body[:260]}{'…' if len(body) > 260 else ''}")
        if answer.citations:
            print(f"  引用：{', '.join(answer.citation_labels)}")
        print(f"  指标：约 {answer.tokens} token · {answer.latency_ms:.0f} ms · 工具 {answer.tool_calls}")
    print("\n" + "=" * 78)
    if args.json:
        _emit(rows, True)
    return EXIT_OK


def cmd_ask(args: argparse.Namespace) -> int:
    """单次问答（支持多轮历史）。"""
    config = load_config(root=args.root)
    agent = _build_agent(config)
    history = list(args.history or [])
    answer = agent.ask(args.question, role=args.role, history=history or None, reset=not history)
    if args.json:
        _emit(answer.to_dict(), True)
    else:
        print(answer.to_markdown())
    return EXIT_OK


def cmd_ingest(args: argparse.Namespace) -> int:
    """重建索引并打印统计（可视作"知识库体检"）。"""
    config = load_config(root=args.root)
    kb = KnowledgeBase(config)
    stats = kb.build(force=True)
    docs = kb.documents(role=args.role)
    payload = {"build": stats, "visible_documents": docs, "index": kb.stats()}
    if args.json:
        _emit(payload, True)
    else:
        print("知识库索引构建完成")
        print("=" * 70)
        print(f"文档数：{stats['documents']}　切块数：{stats['chunks']}　BM25 词项：{kb.stats()['bm25_terms']}")
        print(f"\n当前角色 `{args.role}` 可见文档（{len(docs)} 篇）：")
        for doc in docs:
            visibility = doc.get("visibility")
            print(f"  · {doc['source']}（{doc['sections']} 个片段，可见范围：{visibility}）")
    return EXIT_OK


def cmd_eval(args: argparse.Namespace) -> int:
    """跑评测并检查质量门禁。"""
    config = load_config(root=args.root)
    cases_path = Path(args.cases) if args.cases else config.path_of(str(config.get("eval.cases_path")))
    cases = load_cases(cases_path)
    if not cases:
        print(f"[FAIL] 评测集为空或不存在：{cases_path}")
        return EXIT_FAIL

    agent = _build_agent(config)
    if not args.json:
        print(f"正在评测 {len(cases)} 条用例…")
    report = run_eval(agent, cases, gates=config.get("eval.gates", {}), verbose=not args.json)

    if args.json:
        _emit(report.to_dict(), True)
    else:
        print()
        print(report.to_markdown())

    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\n[OK] 报告已写入 {path}")
    # 同时写一份最新报告，便于对比
    report_dir = config.path_of(str(config.get("eval.report_dir", "evals/reports")))
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latest.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return EXIT_OK if report.passed else EXIT_GATE


def cmd_info(args: argparse.Namespace) -> int:
    """环境与配置自检。"""
    config = load_config(root=args.root)
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, hint: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail, "hint": hint})

    check("python", sys.version_info >= (3, 9), f"Python {sys.version.split()[0]}", "需要 3.9+")
    check("docs_dir", config.path_of("data/docs").exists(), f"文档目录 {config.path_of('data/docs')}", "请确认 data/docs 存在")
    for module, hint in (("yaml", "未装 PyYAML 时使用内置 mini-YAML，功能不变"), ("openai", "接真实模型需要；Mock 模式无需")):
        try:
            __import__(module)
            check(f"optional:{module}", True, "已安装")
        except Exception:
            check(f"optional:{module}", True, "未安装（可降级）", hint)

    provider = config.provider
    has_key = bool(config.api_key())
    if provider == "mock":
        check("llm", True, "Mock 模式（离线确定性，可完整演示与跑评测）")
    else:
        check("llm", has_key, f"provider={provider} model={config.model}", "未检测到 API Key，将自动降级为 Mock")

    try:
        kb = KnowledgeBase(config)
        stats = kb.build()
        check("knowledge_base", stats["chunks"] > 0, f"{stats['documents']} 篇文档 / {stats['chunks']} 个片段")
    except Exception as exc:  # noqa: BLE001
        check("knowledge_base", False, f"构建失败：{exc}")

    payload = {"ok": all(c["ok"] for c in checks), "config": config.describe(), "checks": checks}
    if args.json:
        _emit(payload, True)
    else:
        print("DocPilot 环境自检")
        print("=" * 66)
        for item in checks:
            print(f"{'✅' if item['ok'] else '❌'} {item['name']:<24} {item['detail']}")
            if item["hint"] and not item["ok"]:
                print(f"   └─ {item['hint']}")
        print("=" * 66)
        print(f"运行配置：{config.describe()}")
    return EXIT_OK if payload["ok"] else EXIT_FAIL


def cmd_web(args: argparse.Namespace) -> int:
    """启动 Web 演示界面。"""
    from .web import serve

    config = load_config(root=args.root)
    return serve(config, host=args.host, port=args.port, open_browser=not args.no_browser)


# --------------------------------------------------------------------------- #
# 解析器
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docpilot",
        description="DocPilot —— 企业知识库问答 Agent（混合检索 + 引用溯源 + 拒答 + 权限隔离）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root", help="项目根目录（默认自动推断）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="一键演示 6 个典型场景")
    p_demo.add_argument("--json", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("question", help="问题文本")
    p_ask.add_argument("--role", default="public", help="用户角色（public/hr/finance/legal）")
    p_ask.add_argument("--history", action="append", help="历史问题，可重复传入以模拟多轮")
    p_ask.add_argument("--json", action="store_true")
    p_ask.set_defaults(func=cmd_ask)

    p_ingest = sub.add_parser("ingest", help="重建索引并查看可见文档")
    p_ingest.add_argument("--role", default="public")
    p_ingest.add_argument("--json", action="store_true")
    p_ingest.set_defaults(func=cmd_ingest)

    p_eval = sub.add_parser("eval", help="跑评测 + 质量门禁（未通过退出码为 2）")
    p_eval.add_argument("--cases", help="评测集路径")
    p_eval.add_argument("--report", help="把 Markdown 报告写入指定路径")
    p_eval.add_argument("--json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_info = sub.add_parser("info", help="环境与配置自检")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=cmd_info)

    p_web = sub.add_parser("web", help="启动 Web 演示界面")
    p_web.add_argument("--host", default=None)
    p_web.add_argument("--port", type=int, default=None)
    p_web.add_argument("--no-browser", action="store_true")
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已中断")
        return 130
    except DocPilotError as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return EXIT_FAIL
    except Exception as exc:  # noqa: BLE001 - CLI 顶层兜底，给出可读错误
        print(f"[FAIL] 未预期的错误：{type(exc).__name__}: {exc}")
        return EXIT_FAIL
