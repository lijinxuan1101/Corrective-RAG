# 端到端问答 Benchmark

把一批「问题 → 参考答案」喂给整条 `rag.app` 管线，评判答案对不对、越界问题拦没拦住。
评的是**用户实际拿到的答案**，不是中间的召回名次——最贴近 README「🎯 效果」那几个例子。

```
dataset.jsonl ──► rag.app ──► 规则判(数字/引用/范围) ┐
                                                     ├─► 逐条判定 + 汇总
                              LLM判(语义等价)         ┘
```

## 🚀 使用

```bash
python benchmark/evaluate.py                     # 跑全部
python benchmark/evaluate.py --limit 3           # 只跑前 3 条，调试
python benchmark/evaluate.py --no-llm-judge      # 跳过 LLM，只看规则指标（省钱省时）
python benchmark/evaluate.py --out results.jsonl # 逐条结果落盘
```

依赖：`DASHSCOPE_API_KEY`（重排 + 生成 + 评判都走它）。`in_corpus` 用例会加载本地嵌入
模型和索引，先跑过 `python retrieval/build_index.py`；`out_of_scope` 用例在查询分析阶段
就被拦下，不加载索引。

## 📊 五项指标

| 指标 | 判法 | 说明 |
|---|---|---|
| **总体正确率** | 范围对 **且** 语义对 | 最终看这个。无 LLM 时语义退回数字命中 |
| **范围路由** | 规则 | `out_of_corpus` 是否被正确识别，越界问题不该被拉去检索 |
| **数字命中** | 规则 | `key_figures` 归一化后是否原样出现在答案里 |
| **引用覆盖** | 规则 | 引用文档是否落在正确的 (公司, 年份) |
| **语义正确** | LLM | 关键事实是否等价，容忍措辞和单位换算（891.75亿 = 89,175,178,322.70元） |

**规则和 LLM 各司其职**：规则确定、免费、可复现，但对措辞和 `亿/千` 换算很脆；
LLM 判语义等价和拒答意图。所以规则做诊断性子指标，LLM 做权威的正确性信号——
数字规则没命中但语义判对，很可能是模型用了「亿元」而金标准写了完整数字，看 `semantic_reason` 即可分辨。

## 📝 数据集

`dataset.jsonl`，一行一条：

```json
{
  "id": "wly-2024-revenue",
  "category": "fact",
  "question": "五粮液2024年的营业收入是多少",
  "chat_history": [],
  "expected_scope": "in_corpus",
  "reference_answer": "五粮液2024年的营业收入为891.75亿元（89,175,178,322.70元）。",
  "key_figures": ["89,175,178,322.70"],
  "expected_sources": [{"company": "五粮液", "year": "2024"}]
}
```

| 字段 | 作用 |
|---|---|
| `chat_history` | 多轮上下文，格式同 `main.py`：`["用户: ...", "助手: ..."]`。测指代消解 |
| `expected_scope` | `in_corpus` / `out_of_corpus`，判范围路由 |
| `reference_answer` | 参考答案，喂给 LLM 评判；越界用例写成拒答 |
| `key_figures` | 必须出现的数字，规则精确匹配（归一化后）。留空则跳过数字判定 |
| `expected_sources` | 引用应覆盖的 (公司, 年份)。留空则跳过引用判定 |

三类用例：`fact`（单轮取数）、`fact_multiturn`（追问指代，如"那2024年的呢？"）、
`out_of_scope`（问语料外的公司/年份，应拒答）。

### 加用例

种子集里的数字都经过核对（源自年报或 README 已验证的例子）。**加新用例时务必先用
`python main.py "你的问题"` 跑一遍拿到真实数字再填 `key_figures`**——填错金标准会把
对的答案判成错的。没把握的数字宁可留空 `key_figures`，只靠 LLM 语义判。

## ⚠️ 注意

- 本 benchmark 会**真实调用** DashScope（每条用例 1 次生成 + 1 次评判，`in_corpus`
  另加 1 次重排 + 若干次评分），跑全量有成本和延迟。调参时用 `--limit` 或 `--no-llm-judge`。
- LLM 评判有轻微不确定性，同一条可能偶发翻转。关注趋势而非单条，报数字时说明用的哪版评判。
- 换重排模型后 `rag/nodes.py` 的 `IRRELEVANT` 阈值要重标定，否则范围/评分指标会失真。
