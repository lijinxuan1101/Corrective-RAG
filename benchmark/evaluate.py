#!/usr/bin/env python
"""
端到端问答 benchmark 运行器。

把 dataset.jsonl 里每条「问题 → 参考答案」喂给 rag.app，
再用规则 + LLM 混合评判（见 judge.py），最后打一张汇总表。

    python benchmark/evaluate.py                    # 跑全部
    python benchmark/evaluate.py --limit 3          # 只跑前 3 条（调试用）
    python benchmark/evaluate.py --no-llm-judge     # 跳过 LLM，只看规则指标（省钱省时）
    python benchmark/evaluate.py --out results.jsonl  # 逐条结果落盘

依赖：需要 DASHSCOPE_API_KEY（重排 + 生成 + 评判都走它）；in_corpus 用例会加载
本地嵌入模型和索引，请先跑过 `python retrieval/build_index.py`。
"""
import argparse
import json
import sys
import time
from pathlib import Path

# 允许 `python benchmark/evaluate.py` 直接跑——否则 sys.path[0] 是 benchmark/，
# 找不到项目根下的 rag 包。放在其它 import 之前。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag import app

from benchmark import judge

DATASET = Path(__file__).parent / "dataset.jsonl"


def load_cases(path: Path, limit: int | None = None) -> list[dict]:
    cases = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return cases[:limit] if limit else cases


def run_case(case: dict, use_llm: bool = True) -> dict:
    """跑一条用例，返回带各项判定的结果字典。"""
    t0 = time.perf_counter()
    try:
        state = app.invoke({"question": case["question"],
                            "chat_history": case.get("chat_history", [])})
        error = None
    except Exception as exc:
        state, error = {}, f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - t0

    generation = state.get("generation", "")
    docs = state.get("documents", [])
    scope = state.get("scope", "")

    key_figures = case.get("key_figures", [])
    expected_sources = case.get("expected_sources", [])
    figs = judge.figures_hit(generation, key_figures)
    cits = judge.citations_hit([d.metadata for d in docs], expected_sources)

    scope_ok = scope == case["expected_scope"]
    figure_ok = len(figs) == len(key_figures)          # 无数字要求时恒真
    citation_ok = len(cits) == len(expected_sources)   # 无来源要求时恒真

    semantic_ok = None
    reason = ""
    if use_llm and not error:
        verdict = judge.llm_judge(case["question"], case["reference_answer"], generation)
        semantic_ok, reason = verdict.correct, verdict.reason

    # 权威正确性：有 LLM 判定就以它为准，否则退回规则（数字命中）。
    # 无论哪种，scope 走错就算错——越界问题被拉去检索本身就是失败。
    correctness = semantic_ok if semantic_ok is not None else figure_ok
    overall = bool(scope_ok and correctness and not error)

    return {
        "id": case["id"], "category": case.get("category", ""),
        "question": case["question"], "generation": generation,
        "latency": round(latency, 2), "error": error,
        "scope_ok": scope_ok, "scope_got": scope, "scope_want": case["expected_scope"],
        "figure_ok": figure_ok, "figures_hit": figs, "figures_want": key_figures,
        "citation_ok": citation_ok, "citations_hit": cits, "citations_want": expected_sources,
        "semantic_ok": semantic_ok, "semantic_reason": reason,
        "overall": overall,
    }


def _rate(hits: int, total: int) -> str:
    return f"{hits}/{total} ({hits / total:.0%})" if total else "—"


def summarize(results: list[dict], use_llm: bool) -> None:
    n = len(results)
    print("\n" + "=" * 72)
    print(f"{'ID':<22}{'总判':<6}{'范围':<6}{'数字':<6}{'引用':<6}{'语义':<6}{'耗时'}")
    print("-" * 72)
    for r in results:
        mark = lambda ok: "✅" if ok else ("—" if ok is None else "❌")
        print(f"{r['id']:<22}{mark(r['overall']):<5}{mark(r['scope_ok']):<5}"
              f"{mark(r['figure_ok']):<5}{mark(r['citation_ok']):<5}"
              f"{mark(r['semantic_ok']):<5}{r['latency']:>5.1f}s")
        if r["error"]:
            print(f"  ⚠️  {r['error']}")
        elif not r["overall"] and r["semantic_reason"]:
            print(f"  ↳ {r['semantic_reason']}")
    print("-" * 72)

    fig_cases = [r for r in results if r["figures_want"]]
    cit_cases = [r for r in results if r["citations_want"]]
    sem_cases = [r for r in results if r["semantic_ok"] is not None]
    errors = sum(1 for r in results if r["error"])
    avg_latency = sum(r["latency"] for r in results) / n if n else 0

    print(f"总体正确率   {_rate(sum(r['overall'] for r in results), n)}")
    print(f"范围路由     {_rate(sum(r['scope_ok'] for r in results), n)}")
    print(f"数字命中     {_rate(sum(r['figure_ok'] for r in fig_cases), len(fig_cases))}")
    print(f"引用覆盖     {_rate(sum(r['citation_ok'] for r in cit_cases), len(cit_cases))}")
    if use_llm:
        print(f"语义正确     {_rate(sum(bool(r['semantic_ok']) for r in sem_cases), len(sem_cases))}")
    print(f"平均耗时     {avg_latency:.1f}s" + (f"   ⚠️ {errors} 条报错" if errors else ""))
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端问答 benchmark")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--no-llm-judge", action="store_true", help="跳过 LLM 评判")
    parser.add_argument("--out", type=Path, default=None, help="逐条结果写入 JSONL")
    args = parser.parse_args()

    cases = load_cases(args.dataset, args.limit)
    use_llm = not args.no_llm_judge
    print(f"📋 共 {len(cases)} 条用例，LLM 评判：{'开' if use_llm else '关'}\n")

    results = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['id']}: {case['question']}")
        results.append(run_case(case, use_llm))

    summarize(results, use_llm)

    if args.out:
        with args.out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"💾 逐条结果已写入 {args.out}")


if __name__ == "__main__":
    main()
