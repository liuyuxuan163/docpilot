"""测试：hard negative 生成器与分层评测的关键性质。

这组测试锁的是**数据管道的正确性**，不是模型效果：

1. **case_id 唯一** —— 踩过一次：主题名几乎全中文，slug 全被清洗成空串，
   57 条候选实际只有几个不同的 id，dev/test 切分变成 47/10。
   症状是"文件看着很多条，其实全是重复"。
2. **dev/test 不重叠且覆盖全部候选** —— 否则评测会重复计分或漏题。
3. **切分稳定且大致均衡** —— 同一条用例必须永远落在同一侧（换机器、换时间都一样），
   否则"调参用 dev、报告用 test"这件事就形同虚设。
4. **Wilson 区间的边界** —— 误拒率恰好是 0 时，正态近似会算出负数区间，
   所以这里必须用 Wilson；这条测试防止有人"顺手"改回正态。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_hard_negatives as bhn  # noqa: E402
import eval_stratified as es  # noqa: E402

CANDIDATES = ROOT / "data" / "eval" / "hard_negatives.candidates.jsonl"
KNOWN_KINDS = {"sibling_slot", "synonym_only", "half_answer", "wrong_premise", "permission", "positive_control"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_topic_table_entries_are_well_formed():
    """主题库本身别写漏字段 —— 少一个 ``absent_reason`` 就等于"没证据的标注"。"""
    for topic in bhn.TOPICS:
        assert topic.get("topic"), topic
        assert topic.get("kind") in KNOWN_KINDS, topic
        assert topic.get("anchor"), topic
        assert topic.get("absent_reason"), topic
        assert topic.get("docs"), topic
        assert topic.get("negatives") or topic.get("positives"), topic


def test_slug_never_produces_empty_string():
    """中文主题名必须仍能产出一个可用的 id 片段（这是撞车事故的根因）。"""
    for name in ("丧假", "P6 期权", "VPN 客户端下载", "正例-年假折现", ""):
        assert bhn.slug(name), name
    assert bhn.slug("丧假") != bhn.slug("产假"), "不同主题必须得到不同 slug 片段"


def test_split_is_deterministic():
    """同一个 case_id 永远落在同一侧。"""
    for case_id in ("hard-abc-1234-n1", "hard-xyz-9999-p2"):
        assert bhn.split_of(case_id) == bhn.split_of(case_id)


def test_split_is_roughly_balanced():
    """400 个结构化 id 应当接近对半。

    以前的实现用 ``hash % 2``，只有哈希最后一位参与运算：
    id 前缀相同、只在结尾变化时分布会偏。这条测试正是为了拦住那种退化。
    """
    ids = [f"hard-topic-{index:04d}-n1" for index in range(400)]
    dev = sum(1 for case_id in ids if bhn.split_of(case_id) == "dev")
    assert 0.4 <= dev / len(ids) <= 0.6, f"切分不均衡：dev {dev}/400"


def test_candidates_are_unique_and_complete():
    rows = load_jsonl(CANDIDATES)
    ids = [row["case_id"] for row in rows]
    assert len(ids) == len(set(ids)), "case_id 必须唯一"
    for row in rows:
        assert row["question"].strip()
        assert isinstance(row["expect_answer"], bool)
        assert row["kind"] in KNOWN_KINDS
        assert row["absence_check"], "每条都要有关键词命中证据（哪怕是空关键词的说明）"
        assert "probe" in row and "decision" in row["probe"]


def test_review_state_overlay_is_authoritative():
    """复核状态独立存放，并且必须与候选一一对应。

    这条测试防的是"人工复核结果被生成器静默清空"——
    生成器每次重写候选文件，若把 reviewed 写死成 False，一小时的复核就没了。
    """
    state = {row["case_id"]: row for row in load_jsonl(ROOT / "data" / "eval" / "hard_negatives.review_state.jsonl") if "case_id" in row}
    candidates = {row["case_id"]: row for row in load_jsonl(CANDIDATES)}
    assert set(state) == set(candidates), "状态文件必须覆盖且只覆盖现有候选"
    for case_id, row in candidates.items():
        assert row["reviewed"] == state[case_id]["reviewed"], case_id
        assert row.get("verdict", "keep") == state[case_id]["verdict"], case_id


def test_review_state_records_corpus_fingerprint():
    """状态文件要能回答"这次复核是针对哪一版语料做的"。"""
    meta = None
    for line in (ROOT / "data" / "eval" / "hard_negatives.review_state.jsonl").read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        if "_meta" in data:
            meta = data["_meta"]
            break
    assert meta, "缺少 _meta 行"
    assert len(meta["fingerprint"]) == 12
    assert meta["documents"] > 0 and meta["chunks"] > 0


def test_negatives_carry_the_evidence_needed_for_review():
    rows = [row for row in load_jsonl(CANDIDATES) if not row["expect_answer"]]
    assert rows, "应当有负例"
    for row in rows:
        assert row["absent_reason"].strip(), row["case_id"]
        assert row["anchor"].strip(), row["case_id"]


def test_dev_and_test_are_disjoint_and_cover_everything():
    candidates = load_jsonl(CANDIDATES)
    kept = {row["case_id"] for row in candidates if row.get("verdict", "keep") != "drop"}
    dev = {row["case_id"] for row in load_jsonl(ROOT / "data" / "eval" / "hard_negatives.dev.jsonl")}
    test = {row["case_id"] for row in load_jsonl(ROOT / "data" / "eval" / "hard_negatives.test.jsonl")}
    assert not dev & test, "dev 与 test 不得重叠"
    assert dev | test == kept, "两半必须覆盖全部**保留**的候选（否则评测漏题）"


class TestWilsonInterval:
    def test_bounds_are_clamped(self):
        """0 和 1 这两个端点不能算出负数或大于 1 的区间。

        注意用 approx：浮点下 60/60 会得到 0.999…9，这不是 bug。
        """
        low, _ = es.wilson(0, 60)
        _, high = es.wilson(60, 60)
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(1.0)
        assert low >= 0.0 and high <= 1.0

    def test_matches_known_values(self):
        """54/60 → 约 [0.799, 0.953]（与分层评测输出一致）。"""
        low, high = es.wilson(54, 60)
        assert low == pytest.approx(0.799, abs=1e-3)
        assert high == pytest.approx(0.953, abs=1e-3)

    def test_interval_contains_the_estimate(self):
        for successes, total in ((0, 10), (1, 60), (30, 60), (59, 60), (10, 10)):
            low, high = es.wilson(successes, total)
            assert low <= successes / total <= high

    def test_empty_sample(self):
        assert es.wilson(0, 0) == (0.0, 0.0)
