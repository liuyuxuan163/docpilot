"""实验：用"一次模型判定"替换/加固词面判定，能不能把 hard negative 的误答压下来。

用法::

    python scripts/experiment_judge_gate.py                 # 用 dev 集（默认）
    python scripts/experiment_judge_gate.py --split test    # 定稿后才跑，且只跑一次

背景（为什么必须换信号族，而不是继续调阈值）
--------------------------------------------
在 77 条已复核用例上做过阈值扫描：**零误拒约束下，词面 relevance 最多只能挡住
19/59（32%）的误答**；换 `full_ratio` / `longest_full` 这两个词面信号，
甚至会立刻误拒正例（因为正例里有真正的改写，词面对不上）。
三类词面信号都无法分离，说明问题不在阈值，而在信号族。

本实验的判定器只回答一个问题：

    "这段文档**直接回答**了这个问题吗？还是只是提到了相关话题？"

对比三组：

* ``lexical``：现有 gate（relevance ≥ 0.28 且 span ≥ 2）；
* ``judge``：纯判定器；
* ``lexical+judge``（推荐）：词面先兜底（省 token），判定器再复核。

缓存
----
判定结果写进 ``data/eval/judge_cache.json``，键为
``模型 + prompt 版本 + 问题 + chunk_id``。这样**同一次判定不会重复花钱**，
重跑是离线且确定的 —— 和 `rag` 里"向量落盘缓存"是同一个思路。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docpilot.core import ensure_utf8_stdio, load_config  # noqa: E402
from docpilot.rag import KnowledgeBase  # noqa: E402

CASES = ROOT / "data" / "eval" / "hard_negatives.candidates.jsonl"
HOLDOUT = ROOT / "data" / "eval" / "hard_negatives.holdout.jsonl"
CACHE = ROOT / "data" / "eval" / "judge_cache.json"

# 每次改 prompt 都是一次新实验：版本号进缓存键，旧结果不会被覆盖也不会被误用。
PROMPTS = {
    # v3：在第一版基础上授权"否定式回答"和"纠正错误前提"。
    # test 上仍误拒 5/11，诊断显示全是"文档给规则、问题问具体情形"的推理题。
    "v3": """你是一个严格的文档问答判定器。判断给定的文档片段是否**直接回答了**用户的问题。

规则：
- 片段包含回答问题所必需的信息时，答 YES；
- **片段明确否定**（"不可以 / 不需要 / 不另行 / 不得 / 不予"）时，这同样是在回答问题，答 YES；
- **问题中的前提写错、而片段给出了正确事实**时，答 YES（这是纠正，不是没回答）；
- 片段只是"提到了相关话题"、只给了部分前提、或需要额外推测才能得出答案时，答 NO；
- 不要用你自己的常识补全片段里没有的信息。

问题：{question}

文档片段：{chunk}

只输出一个词：YES 或 NO。""",
    # v4：额外授权"由规则推结论"。
    # 针对 test 上被误拒的那 5 条（如文档写「P7 及以上可乘公务舱」，问「P6 能坐吗」）。
    "v4": """你是一个文档问答判定器。判断给定的文档片段是否**足以回答**用户的问题。

规则：
- 片段包含回答问题所必需的信息时，答 YES；
- **片段明确否定**（"不可以 / 不需要 / 不另行 / 不得 / 不予"）时，同样是在回答问题，答 YES；
- **问题中的前提写错、而片段给出了正确事实**时，答 YES（这是纠正，不是没回答）；
- **片段给出了判定规则（条件、阈值、职级范围、名单、时长上限），而问题是在问某个具体情形是否满足该规则时，答 YES。**
  例：片段写「P7 及以上可乘坐公务舱」，问题问「P6 能坐公务舱吗」→ 答 YES
  （可由规则推出"不能"，这属于可回答，不属于没依据）；
  例：片段写「服务台电话工作日 9:00-18:00」，问「周末上班吗」→ 答 YES；
- 只有当片段**完全没有覆盖**问题所需的信息（缺的正是问题问的那个东西）时，才答 NO；
- 不要用你自己的常识补全片段里没有的信息。

问题：{question}

文档片段：{chunk}

只输出一个词：YES 或 NO。""",
}
PROMPT_VERSION = "v4"


def cache_key(model: str, version: str, question: str, chunk_id: str) -> str:
    blob = json.dumps([model, version, question, chunk_id], ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_cache() -> Dict[str, Any]:
    if not CACHE.exists():
        return {}
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(cache: Dict[str, Any]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


class Judge:
    """薄封装：走 OpenAI 兼容接口；无 Key 时明确报错而不是静默降级。

    为什么不复用 MockLLM 的降级逻辑：那会让"判定器没配 Key"看起来像
    "判定器说 NO"，从而把误拒率算错。实验脚本宁可直接失败。
    """

    def __init__(
        self,
        config,
        model: Optional[str] = None,
        max_tokens: int = 256,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        from openai import OpenAI

        key = config.api_key()
        if not key:
            raise SystemExit("没有 API Key：请在 .env 里配置 DEEPSEEK_API_KEY")
        if prompt_version not in PROMPTS:
            raise SystemExit(f"未知的 prompt 版本：{prompt_version}（可选：{', '.join(PROMPTS)}）")
        self.model = str(model or config.model)
        self.max_tokens = int(max_tokens)
        self.prompt_version = prompt_version
        self.prompt = PROMPTS[prompt_version]
        self._client = OpenAI(api_key=key, base_url=config.base_url(), timeout=60)
        self.cache = load_cache()
        self.calls = 0
        self.tokens = 0

    def ask(self, question: str, chunk_id: str, chunk_text: str) -> bool:
        key = cache_key(self.model, self.prompt_version, question, chunk_id)
        cached = self.cache.get(key)
        # 只认非空缓存：v1 那次 max_tokens=4 导致 45 条全部返回空串，
        # 而空串被当成 NO 缓存了下来 —— 这种"坏缓存"必须在读取时识别出来。
        if cached is not None and cached.get("raw"):
            return bool(cached["verdict"])
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt.format(question=question, chunk=chunk_text[:1200])}],
            temperature=0,
            # 4 太小：推理型模型会先花掉内容预算，导致 content 为空。
            # 宁可多给几十个 token，也不能拿到空串还当结果。
            max_tokens=self.max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        usage = getattr(response, "usage", None)
        self.calls += 1
        self.tokens += getattr(usage, "total_tokens", 0) or 0
        if not raw:
            # 静默失败的典型：空串会被解析成"判定为否"，于是"全部拒绝"
            # 看起来像一个漂亮的 0% 误答。实验必须在这里**炸掉**。
            raise RuntimeError(
                f"判定器返回空内容（question={question!r}）。"
                "检查模型是否为推理型、max_tokens 是否够用、或换 --model。"
            )
        verdict = raw.startswith("YES")
        self.cache[key] = {
            "verdict": verdict,
            "raw": raw[:40],
            "question": question,
            "chunk_id": chunk_id,
            "model": self.model,
        }
        return verdict

    def flush(self) -> None:
        save_cache(self.cache)


def main() -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description="判定器 gate 实验")
    parser.add_argument("--split", choices=["dev", "test", "all", "holdout"], default="dev",
                        help="dev 调参、test/holdout 只跑一次")
    parser.add_argument("--model", help="判定用的模型（默认取配置；推理型模型可能返回空 content）")
    parser.add_argument("--max-tokens", type=int, default=256, help="判定器输出预算")
    parser.add_argument("--prompt-version", choices=sorted(PROMPTS), default=PROMPT_VERSION,
                        help="判定 prompt 版本（每次改都是一次新实验，缓存按版本隔离）")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="holdout 还没人工复核，需要显式允许（结果只能当方向性参考）")
    args = parser.parse_args()

    config = load_config(root=ROOT, use_dotenv=True)
    if config.provider == "mock":
        raise SystemExit("需要真实模型：请在 .env 或 --provider 中指定 deepseek/openai")

    source = HOLDOUT if args.split == "holdout" else CASES
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not args.include_unreviewed:
        rows = [r for r in rows if r.get("reviewed") and r.get("verdict", "keep") != "drop"]
    else:
        print("⚠ 含未复核用例：数字只能当方向性参考")
    if args.split not in ("all", "holdout"):
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_hard_negatives import split_of

        rows = [r for r in rows if split_of(r["case_id"]) == args.split]
    print(f"用例 {len(rows)} 条（split={args.split}）· prompt={args.prompt_version} · 模型 {args.model or config.model}")

    kb = KnowledgeBase(config)
    kb.build()
    judge = Judge(config, model=args.model, max_tokens=args.max_tokens, prompt_version=args.prompt_version)

    results = []
    started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        hits = kb.search(row["question"], top_k=1, role=row["role"])
        top = hits[0] if hits else None
        lexical = bool(row["probe"]["has_evidence"])  # 现用 gate 的结论
        verdict = judge.ask(row["question"], top.chunk_id, top.text) if top else False
        results.append(
            {
                "case_id": row["case_id"],
                "question": row["question"],
                "chunk_id": top.chunk_id if top else "",
                "kind": row["kind"],
                "expect_answer": row["expect_answer"],
                "lexical": lexical,
                "judge": verdict,
                "agree": lexical == verdict,
            }
        )
        if index % 10 == 0:
            print(f"  …{index}/{len(rows)}（已花 {judge.tokens} token）")
    judge.flush()

    def rate(subset: List[Dict[str, Any]], field: str) -> str:
        neg = [r for r in subset if not r["expect_answer"]]
        pos = [r for r in subset if r["expect_answer"]]
        if field == "lexical":
            missed = sum(1 for r in neg if r["lexical"])
            over = sum(1 for r in pos if not r["lexical"])
        elif field == "judge":
            missed = sum(1 for r in neg if r["judge"])
            over = sum(1 for r in pos if not r["judge"])
        else:  # and
            missed = sum(1 for r in neg if r["lexical"] and r["judge"])
            over = sum(1 for r in pos if not (r["lexical"] and r["judge"]))
        return f"误答 {missed}/{len(neg)} ({missed/len(neg):.1%}) · 误拒 {over}/{len(pos)}"

    print("\n" + "=" * 88)
    print(f"{'策略':<14}{'负例（该拒却答）':<40}{'正例（该答却拒）'}")
    print("-" * 88)
    for label, field in (("lexical", "lexical"), ("judge", "judge"), ("lexical+judge", "and")):
        neg = [r for r in results if not r["expect_answer"]]
        pos = [r for r in results if r["expect_answer"]]
        if field == "lexical":
            missed = sum(1 for r in neg if r["lexical"])
            over = sum(1 for r in pos if not r["lexical"])
        elif field == "judge":
            missed = sum(1 for r in neg if r["judge"])
            over = sum(1 for r in pos if not r["judge"])
        else:
            missed = sum(1 for r in neg if r["lexical"] and r["judge"])
            over = sum(1 for r in pos if not (r["lexical"] and r["judge"]))
        print(f"{label:<14}{f'{missed}/{len(neg)} ({missed/len(neg):.1%})':<40}{f'{over}/{len(pos)}'}")
    print("-" * 88)
    print(f"本次调用 {judge.calls} 次、{judge.tokens} token、耗时 {time.perf_counter()-started:.0f}s")
    print(f"缓存：{CACHE.relative_to(ROOT)}（重跑免费且确定）")

    disagreed = [r for r in results if not r["agree"]]
    print(f"\n两种判定的分歧 {len(disagreed)} 条（这些就是词面判定的盲区）：")
    for row in disagreed[:10]:
        print(f"  {'词面=答' if row['lexical'] else '词面=拒'} / {'判定=YES' if row['judge'] else '判定=NO'}  {row['case_id']}")

    over = [r for r in results if r["expect_answer"] and not (r["lexical"] and r["judge"])]
    if over:
        print(f"\n⚠ 判定器误拒的正例 {len(over)} 条（zero-over-refusal 是硬约束，必须看）：")
        for row in over:
            raw = judge.cache.get(cache_key(judge.model, judge.prompt_version, row["question"], row["chunk_id"]), {})
            print(f"  {row['case_id']}：{row['question']}")
            print(f"    判定器原话：{raw.get('raw', '(缓存缺失)')}")

    leaked = [r for r in results if not r["expect_answer"] and r["judge"] and r["lexical"]]
    if leaked:
        print(f"\n⚠ 判定器仍然放过的负例 {len(leaked)} 条（这些是下一版要啃的骨头）：")
        for row in leaked:
            print(f"  {row['case_id']}：{row['question']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
