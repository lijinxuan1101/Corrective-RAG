# 检索层

把 `doc_parsing` 产出的嵌入单元变成可查询的混合索引。

```
嵌入单元 ──┬─→ Qwen3-Embedding ─→ dense.npy ──┐
           │                                   ├─→ RRF 融合 ─→ 归并父块 ─→ 重排 ─→ Top-K
           └─→ jieba + BM25 ───→ bm25.pkl ─────┘
```

嵌入跑在本地，重排默认走百炼 API（比本地快 4 倍），缺 key 时自动退回本地模型。

## 🚀 使用

```bash
# 建索引（首次会从 HuggingFace 下载模型）
python retrieval/build_index.py

# 换更大的嵌入模型
python retrieval/build_index.py --model Qwen/Qwen3-Embedding-4B
```

```python
from retrieval import HybridRetriever

retriever = HybridRetriever()
for doc in retriever.search("五粮液2024年的营业收入是多少", top_k=5):
    print(doc.metadata["section"], doc.metadata["page_start"])
    print(doc.page_content[:200])
```

### 元数据过滤

语料里同一家公司有多个年度，问"营业收入"时模型分不清你要哪一年。
已知公司或年份时用 `filters` 收窄范围，它会在两路召回之前生效：

```python
retriever.search("营业收入是多少", filters={"company": "五粮液", "year": "2024"})
```

不加过滤时上例返回的第一条是 2025 年报；加上之后前三条全部落在 2024 年报的
利润表和收入构成上。可过滤的字段就是块的元数据：`company`、`year`、`source`、`unit_type`。

## 🧩 三段式设计

### 1. 双路召回

稠密和稀疏的失效模式是互补的，年报查询恰好同时踩中两者：

| 查询类型 | 稠密向量 | BM25 |
|---|---|---|
| "公司现金充裕吗" | ✅ 语义匹配"货币资金" | ❌ 无词面重叠 |
| "应收账款坏账准备" | ⚠️ 易与"存货跌价准备"混淆 | ✅ 术语精确命中 |
| "127,398,915,484.11" | ❌ 数字语义几乎为零 | ✅ 精确匹配 |

中文没有空格，BM25 前必须分词，这里用 `jieba`（MinerU 已带，不引入新依赖）。

### 2. RRF 融合

两路的分数量纲不可比——余弦相似度在 0~1，BM25 分数无上界且随语料变化。
直接加权求和需要反复调标准化参数，而 [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) 只用**名次**：

```
score(d) = Σ 1 / (60 + rank_i(d))
```

常数 60 起平滑作用，避免某一路的第 1 名过度主导。无需调参是它在实践中被广泛采用的原因。

### 3. 归并父块后再重排

召回命中的是**行级单元**（比如表格的某一行），但喂给 LLM 的必须是**完整父块**，
否则 LLM 看不到表头和上下文。所以融合后按 `parent_id` 归并，同一父块取其所有命中单元的最高分。

重排器放在归并之后而不是之前：交叉编码器很贵，对 20 个父块打分远比对 60 个单元打分划算，
而且它看到的是完整上下文，判断更准。

## 🤖 模型

| 用途 | 模型 | 位置 | 备注 |
|---|---|---|---|
| 嵌入 | `Qwen3-Embedding-0.6B` | 本地 | 1024 维 / 32k 上下文，fp16 约 1.2 GB |
| 重排 | `qwen3.7-text-rerank` | 百炼 API | 默认 |
| 重排（备用） | `Qwen3-Reranker-0.6B` | 本地 | 缺 key 时自动启用 |

重排换成 API 是因为它快 4 倍（单次检索 1.54s vs 6.23s）且省 1.2GB 内存，
细节见 [providers/README](../providers/README.md)。
换后端要注意 **`rag/nodes.py` 里的 `IRRELEVANT` 阈值需要重新标定**——
两个模型对无关查询的打分差一个量级。

想强制用本地：

```python
from retrieval import HybridRetriever
from retrieval.rerank import score

retriever = HybridRetriever(reranker=score)
```

嵌入选 Qwen3 系列的关键是 **32k 上下文**。我们的块中位数 530 token、最长 3212 token，
如果用 512-token 的模型（`bge-large-zh-v1.5` 等），超过一半的块会被**静默截断**，
表格的后半部分直接消失且不报错。

`MAX_TOKENS` 设为 4096 而非 32768：模型支持长上下文不代表要按它分配显存，
截到略高于实际最长块即可。

Qwen3-Reranker 不是常规 cross-encoder，而是让因果语言模型回答 yes/no，
取这两个 token 的 logits 做 softmax 当分数。`rerank.py` 里的 prompt 模板必须与官方一致，
包括那个空的 `<think>\n\n</think>` 块，否则分数会明显退化。

**查询侧要加指令前缀**，文档侧不加——这是 Qwen3-Embedding 的非对称训练方式决定的，
漏掉前缀会损失召回质量。前缀定义在 `index.QUERY_INSTRUCT`。

想换 4B：质量更好但嵌入耗时和显存都翻几倍，24GB 内存下嵌入和重排不要同时常驻。

## 💾 为什么不用向量库

1 万量级的向量，`numpy` 点积全量扫描只要几毫秒，比向量库的网络/序列化开销还低。
引入 Chroma 只会增加依赖和状态管理成本，等语料上到百万级再换不迟。

索引产物（`index/` 已被 gitignore）：

| 文件 | 内容 |
|---|---|
| `dense.npy` | 归一化后的向量矩阵，`float32` |
| `bm25.pkl` | 建好的 BM25 索引 |
| `units.jsonl` | 单元文本与元数据，与前两者行序一致 |

三者靠**行序对齐**，所以必须一起重建，不能只更新其中一个。

## ⏱️ 性能

M5 / 24GB，1341 个父块、10610 个嵌入单元：

| 阶段 | 耗时 |
|---|---|
| 建索引（一次性） | 11 分 48 秒 |
| 稠密召回 | 0.07–0.14 s |
| BM25 召回 | 0.001–0.002 s |
| RRF + 归并 | < 0.01 s |
| 重排 API（30 个父块） | 1.5 s 平均 |
| 重排本地（20 个父块） | 3–9 s |

延迟仍几乎全在重排上，召回两路加起来不到 0.2 秒。
`RERANK_POOL` 设成 30 而不是 20，是因为 API 快到足以支撑更大的候选池——
实测 100 篇也只要 2.41 秒。想要更低延迟可以调小它，或 `use_reranker=False`。

下面两个坑是本地模型特有的，走 API 时不涉及：

**MPS 上必须用 fp16。** 默认 fp32 编码全量要 4 小时，fp16 只要 11.5 分钟——快 20 倍，
而向量归一化之后精度差异可以忽略（实测无 NaN，范数正常）。

**重排前要按长度排序再分批。** 一批要 padding 到最长的那条，长短混批浪费巨大。
排序后最慢的查询从 25.4 秒降到 9.0 秒。

另外 MPS 会按张量形状重新编译 kernel，所以基准测试必须充分预热，
否则会测出"截断到 2048 比 4096 还慢"这种反直觉结果。

## ⚙️ 可调参数

`search.py` 顶部：

| 参数 | 默认 | 说明 |
|---|---|---|
| `CANDIDATES` | 60 | 每路召回的单元数，召回不全就调大 |
| `RERANK_POOL` | 30 | 送进重排器的父块数，直接决定延迟 |
| `RRF_K` | 60 | 越小越信任头部排名 |
| `TOP_K` | 5 | 最终返回给 LLM 的父块数 |

调试时可以 `search(query, use_reranker=False)` 跳过重排，用来分辨问题出在召回还是排序。
