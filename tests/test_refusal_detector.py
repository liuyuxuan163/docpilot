"""测试：拒答识别器对**真实模型话术**的覆盖。

背景：``_looks_like_refusal`` 原来是一份照着 Mock 话术写的关键词表，
对真实模型的说法（"没有 X 这个说法 / 未检索到 / 没有直接规定"）**全部漏判** ——
59 条负例里有 9 条正确拒答被当成"作答"，直接把误答率算高了。

改法是：窗口收窄到结论区（前 90 字）+ 短语表换成"否定词 + 缺什么"的正则。
在 129 条真实模型答案上召回 87.8% → 98.6%、正例误判 7.3% → 3.6%。
这组测试把两种方向的边界都钉住，防止以后又退回"全文关键词匹配"。
"""

from __future__ import annotations

import pytest

from docpilot.agent import _looks_like_refusal


class TestRefusalDetector:
    @pytest.mark.parametrize(
        "text",
        [
            # 真实模型最常见的几种拒答说法（旧词表全都漏判）
            "**结论：知识库中没有“13 薪”这个说法**，也没有写明发放时间。",
            "知识库中未检索到“发票丢失后如何补办”的具体处理办法。",
            "**结论：知识库中没有直接规定“延长后的试用期薪资比例”**，因此无法确认。",
            "知识库中没有关于“加急打款”的任何规定，因此无法确认是否支持加急。",
            "你好，这里没有找到相关依据，建议咨询 HR。",
            # 权限类：必须认出来，否则会被当成"答了但答错"
            "你当前角色没有访问《薪酬制度》文档的权限。",
            "这个问题涉及受限文档（hr/compensation.md）。",
            # 追问类
            "请补充一下具体对象，例如是哪份文件。",
        ],
    )
    def test_detects_real_model_refusals(self, text: str):
        assert _looks_like_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            # 正常作答：结论在第一句，后面才出现"未包含"这类补充
            "**结论：不能折现。** 按现行制度，超出结转限额的部分自动作废。",
            "**结论：不可以。** 源代码属于机密级数据，禁止上传至公有云 AI。",
            "结论：P4 职级的绩效工资占比为 30% [1]。",
            "病假工资按当地最低工资标准的 80% 计发 [1]。",
        ],
    )
    def test_does_not_flag_normal_answers_with_caveats(self, text: str):
        assert _looks_like_refusal(text) is False

    def test_empty_is_refusal(self):
        assert _looks_like_refusal("") is True

    def test_mock_wording_still_detected(self):
        """Mock 的固定话术必须继续生效 —— 否则 Mock 基线的指标会整体漂移。"""
        assert _looks_like_refusal(
            "知识库中没有检索到与问题相关的内容，因此无法回答。"
        ) is True
        assert _looks_like_refusal(
            "这个问题涉及受限文档（`hr/compensation.md`），你当前角色（public）没有访问权限。"
        ) is True
