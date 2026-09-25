"""端到端问答 benchmark：跑一批「问题 → 参考答案」，规则 + LLM 混合评判。"""
from benchmark.judge import citations_hit, figures_hit, llm_judge

__all__ = ["citations_hit", "figures_hit", "llm_judge"]
