"""
端到端答案的评判：规则 + LLM 混合。

- **规则**判它擅长的：数字是否原样出现、引用是否落在正确的公司/年份。
  这两项确定、免费、可复现，但对措辞和单位换算很脆。
- **LLM**判语义等价：891.75亿元 与 89,175,178,322.70元 是同一个数，
  拒答类问题模型是否真的拒答了——这些规则判不了。

两层各取所长：规则做诊断性子指标，LLM 做权威的正确性信号（见 evaluate.py 的汇总）。
"""
import re
from typing import List, Sequence

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from providers import chat_model

JUDGE_MODEL = "qwen-plus"        # 评判要准，用 plus 而非 flash
# 结构化输出走 json_schema，与 rag/chains.py 同因：function_calling 在 Qwen 上会间歇死循环
STRUCTURED = {"method": "json_schema"}

# 归一化：去掉千分位、空白和「元」，让 "89,175,178,322.70元" 与金标准可比。
# 只去这些，不动小数点和数字本身——单位换算（亿/千）交给 LLM，规则不碰。
_STRIP = re.compile(r"[,，\s元]")


def _norm(text: str) -> str:
    return _STRIP.sub("", text or "")


def figures_hit(generation: str, key_figures: Sequence[str]) -> List[str]:
    """返回 key_figures 中在答案里（归一化后）原样出现的那些。全命中即数字正确。"""
    normalized = _norm(generation)
    return [f for f in key_figures if _norm(f) in normalized]


def citations_hit(cited_meta: Sequence[dict], expected_sources: Sequence[dict]) -> List[dict]:
    """返回 expected_sources 中被引用文档覆盖到的（按公司+年份匹配）。"""
    got = {(m.get("company"), str(m.get("year"))) for m in cited_meta}
    return [s for s in expected_sources
            if (s["company"], str(s["year"])) in got]


# ---------------------------------------------------------------
class Verdict(BaseModel):
    correct: bool = Field(description="模型答案是否与参考答案在关键事实上一致")
    reason: str = Field(description="简短判定理由，一句话")


JUDGE_SYSTEM = """你在评判一个企业年报问答系统的回答是否正确。

给你三样东西：用户问题、参考答案、模型答案。判定模型答案是否与参考答案在关键事实上一致。

判定规则：
- 只看关键事实和数字是否一致，忽略措辞、句式、以及引用标注（[1][2]）的差异。
- 数字单位等价即算一致：891.75亿元 与 89,175,178,322.70元 视为相同；
  22,146,581千元 与 约221.47亿元 视为相同。允许合理的四舍五入。
- 参考答案若是"语料中没有/无法回答"这类拒答，则模型答案也必须是拒答才算正确；
  模型若编造了一个具体数字，判错。
- 模型答案遗漏了参考答案的关键数字，或给出与之矛盾的数字，判错。

只输出结构化判定，不要复述问题。"""

_judge_prompt = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM),
    ("human", "用户问题：{question}\n\n参考答案：{reference}\n\n模型答案：{answer}"),
])

_judge = _judge_prompt | chat_model(
    JUDGE_MODEL, max_tokens=512).with_structured_output(Verdict, **STRUCTURED)


def llm_judge(question: str, reference: str, answer: str) -> Verdict:
    """LLM 语义评判。评不出来时按不正确处理，理由里注明，不让异常中断整轮 benchmark。"""
    try:
        return _judge.invoke(
            {"question": question, "reference": reference, "answer": answer})
    except Exception as exc:
        return Verdict(correct=False, reason=f"评判调用失败: {type(exc).__name__}")
