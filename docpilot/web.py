"""Web 演示界面：标准库 ``http.server`` 实现，**零额外依赖**。

为什么不用 FastAPI？
--------------------
这个项目要能"面试现场被要求跑一下"时立刻可用。装 FastAPI + uvicorn 是两分钟的事，
但**两分钟足够让演示冷场**。用标准库的好处是 clone 完就能开，
代价是没有自动文档与流式——对这个体量的演示完全够用。

页面（``static/index.html``）本身也是一个"能讲的东西"：
它把**检索证据、引用、权限、拒答**都摊开显示，而不只是给一段文字。
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from .agent import Agent
from .core import Config
from .llm import build_llm
from .rag import KnowledgeBase

__all__ = ["serve", "create_app", "get_agent", "STATIC_DIR"]

STATIC_DIR = Path(__file__).resolve().parent / "static"

_agent_lock = threading.Lock()
_agent: Optional[Agent] = None


def get_agent(config: Config) -> Agent:
    """进程内复用 Agent（索引构建只做一次）。"""
    global _agent
    with _agent_lock:
        if _agent is None:
            kb = KnowledgeBase(config)
            kb.build()
            _agent = Agent(config=config, kb=kb, llm=build_llm(config))
        return _agent


def handle_ask(config: Config, payload: Dict[str, Any]) -> Dict[str, Any]:
    """处理一次问答请求。"""
    question = str(payload.get("question", "")).strip()
    if not question:
        return {"error": "question 不能为空"}
    role = str(payload.get("role", "public"))
    history = payload.get("history") or None
    agent = get_agent(config)
    answer = agent.ask(question, role=role, history=history, reset=not history)
    return answer.to_dict()


def handle_documents(config: Config, role: str = "public") -> Dict[str, Any]:
    agent = get_agent(config)
    return {"role": role, "documents": agent.kb.documents(role=role), "stats": agent.kb.stats()}


# --------------------------------------------------------------------------- #
# 标准库 HTTP 实现
# --------------------------------------------------------------------------- #


def _make_handler(config: Config) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "DocPilot/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # pragma: no cover
            pass  # 静默：演示时不想刷屏

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                index = STATIC_DIR / "index.html"
                body = index.read_bytes() if index.exists() else "<h1>DocPilot</h1><p>缺少 static/index.html</p>".encode("utf-8")
                self._send(200, body, "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._json({"ok": True, "provider": config.provider, "model": config.model, "config": config.describe()})
                return
            if path == "/api/documents":
                role = "public"
                if "?" in self.path:
                    query = self.path.split("?", 1)[1]
                    for part in query.split("&"):
                        if part.startswith("role="):
                            role = part.split("=", 1)[1]
                self._json(handle_documents(config, role))
                return
            self._json({"error": "not found", "path": path}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._json({"error": "请求体不是合法 JSON"}, status=400)
                return
            if path == "/api/ask":
                try:
                    result = handle_ask(config, payload)
                except Exception as exc:  # noqa: BLE001
                    self._json({"error": f"{type(exc).__name__}: {exc}"}, status=500)
                    return
                self._json(result, status=400 if "error" in result else 200)
                return
            self._json({"error": "not found", "path": path}, status=404)

    return Handler


def create_app(config: Config) -> Any:  # pragma: no cover - 预留 FastAPI 扩展位
    """占位：如需换成 FastAPI，只要实现同名的 ``/api/ask`` 与 ``/api/documents``。"""
    raise NotImplementedError("本项目使用标准库实现，无需 Web 框架")


def serve(config: Config, host: Optional[str] = None, port: Optional[int] = None, open_browser: bool = True) -> int:
    """启动演示服务。"""
    host = host or str(config.get("web.host", "127.0.0.1"))
    port = int(port or config.get("web.port", 8020))
    handler = _make_handler(config)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print("DocPilot Web 演示")
    print(f"  地址：{url}")
    print(f"  配置：{config.describe()}")
    print("  按 Ctrl+C 停止")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover
        print("\n已停止")
    finally:
        httpd.server_close()
    return 0
