"""DocPilot —— 一个精简但完整的 RAG 知识库问答 Agent。

设计目标（刻意与"重装备"相反）
------------------------------
1. **一条命令跑通**：``python -m docpilot demo``，无需 API Key、无需向量数据库。
2. **2~3 天能读透**：源码约 1800 行，单文件 ReAct 主循环，模块不超过 9 个。
3. **但要有真东西**：混合检索（BM25 + 稠密 + RRF）、多轮问题改写、
   引用溯源、**拒答**、**权限隔离**、可评测。这些是"通用"不等于"平庸"的关键。

它解决的真实问题
----------------
企业内部的文档问答有一个共性难点：**模型很会编，但企业答错一句话是要负责的。**
所以本项目的核心不是"答得更花哨"，而是三条底线：

* **不编**：每条结论必须能指回原文片段（引用溯源）；
* **不知道就说不知道**：检索不到依据时明确拒答，而不是硬凑（拒答机制）；
* **不该看的看不到**：按用户角色做文档级权限过滤，检索阶段就隔离（权限隔离）。

架构一览::

    cli.py / web
        │
        └── agent.py        ReAct 主循环（唯一推理引擎）
              ├── tools.py  工具注册表（检索 / 文档列表 / 问题改写）
              ├── rag.py    混合检索：BM25 + 本地嵌入 + RRF + 可选重排
              └── llm.py    Mock 引擎 / OpenAI 兼容网关（双模式）
                    ↑
              core.py      类型契约、配置、mini-YAML、错误（零依赖）
"""

from .core import (  # noqa: F401
    Answer,
    Citation,
    Config,
    DocPilotError,
    Message,
    RetrievedChunk,
    ToolCall,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "Answer",
    "Citation",
    "Config",
    "DocPilotError",
    "Message",
    "RetrievedChunk",
    "ToolCall",
]
