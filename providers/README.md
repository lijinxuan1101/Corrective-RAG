# 外部 API 供应商

所有需要密钥的外部服务集中在这里，别处不直接读 `os.environ`，也不直接调 SDK。
这样换供应商或轮换密钥时只有一个地方要改。

```
providers/
├── config.py      读 .env，缺 key 时给出可操作的报错
├── embedding.py   DashScope 文本嵌入
├── rerank.py      百炼文本重排
└── llm.py         DashScope 对话模型
```

## 🔑 配置

密钥放在项目根目录的 `.env`（已在 `.gitignore` 中）：

```
DASHSCOPE_API_KEY=sk-...
```

一个 key 同时覆盖嵌入和对话两类接口。

## 🚀 使用

```python
from providers import embed, embed_query, rerank, chat_model

vectors = embed(["文档一", "文档二"])        # (2, 1024)，已归一化
query = embed_query("用户的问题")             # (1024,)

scores = rerank("用户的问题", ["候选一", "候选二"])   # 顺序与输入一致

llm = chat_model("qwen-plus")
llm = chat_model("qwen-flash", temperature=0)  # 高频小任务用 flash
```

## 🧬 嵌入

模型 `qwen3.7-text-embedding-flash`，1024 维。实测出来的接口边界：

| 项目 | 实测值 |
|---|---|
| 单次批量上限 | **25 条**（26 条即报 `InvalidParameter`） |
| 单条长度 | 至少 2 万 token（我们最长的块 3212，绰绰有余） |
| `dimension` 参数 | 支持，可降到 512 等 |
| `text_type` | 支持 `query` / `document`，且确实生效 |

**`text_type` 必须分清 query 和 document。** 模型是非对称训练的，
同一句"营业收入"用两种 type 编码出来的向量余弦只有 0.756——
查询侧误用 `document` 会实打实损失召回质量。

`embed()` 把分批、重试、并发都包掉了：25 条一批、6 线程并发、429 和 5xx 指数退避重试，
4xx 直接抛错（请求本身有问题，重试没有意义）。全量 10610 条约 10 分钟。

**返回顺序必须按 `index` 字段还原**——接口不保证顺序，而且字段叫 `index` 不是
文档里常见的 `text_index`。

一个不影响使用但值得知道的现象：同一条文本在不同批次里编码，结果会有约 `8e-3` 的偏差
（服务端按批次形状走不同 kernel），余弦相似度 0.9974。单独重复编码则完全一致。

## 🎯 重排

模型 `qwen3.7-text-rerank`。注意它**不在 dashscope.aliyuncs.com 上**，
而是走 `maas.qianwenaiapi.com`，SDK 里没有对应方法，所以 `rerank.py` 直接发 HTTP。

真实负载（父块最长 4899 字符）实测：

| 文档数 | 字符数 | 耗时 |
|---|---|---|
| 20 | 67,058 | 1.28 s |
| 50 | 142,078 | 3.76 s |
| 100 | 243,478 | 2.41 s |

单篇 19,596 字符也能过。`BATCH_SIZE` 取 50 是留余量，超出会自动分批——
交叉编码器对每个 (query, document) 对独立打分，分批不影响分数可比性。

**返回结果按分数降序，必须按 `index` 字段还原成输入顺序**，否则分数会对错文档。

这比本地 `Qwen3-Reranker-0.6B` 快 4 倍（单次检索 1.54s vs 6.23s），
还省下 1.2GB 常驻内存，所以 `retrieval` 默认走它，缺 key 时才退回本地。

## 💬 对话

百炼提供 OpenAI 兼容端点，所以直接复用 `ChatOpenAI` 换 `base_url`，
上层 LangChain 链路不用改。可用型号（同一个 key 全部验证可用）：

| 型号 | 定位 |
|---|---|
| `qwen-flash` | 最快最便宜，适合文档评分这类高频小任务 |
| `qwen-plus` | 均衡，默认值，适合最终生成 |
| `qwen-max` / `qwen3-max` | 最强，贵且慢 |

## ⚠️ 区域

这个 key 属于**国内区**账号，只能走默认的北京端点。
国际站端点 `dashscope-intl.aliyuncs.com` 会返回 401 `InvalidApiKey`。
如果以后换成新加坡区的账号，需要同时改 `embedding.py` 的 `base_http_api_url`
和 `llm.py` 的 `BASE_URL`。

## 🔄 与本地模型的关系

`retrieval/` 目前用的是**本地** `Qwen3-Embedding-0.6B`，不是这里的 API。
两者虽然都是 1024 维，但**向量空间完全不同**，不能混用——
用 API 编码查询去搜本地建的索引，结果会是噪声。

要切换就得整体切换并重建索引。两边的取舍：

| | 本地 0.6B | DashScope API |
|---|---|---|
| 建索引（10610 条） | 11.8 分钟 | ~10 分钟 |
| 单次查询延迟 | 0.07–0.14 s | 网络往返，更慢 |
| 成本 | 免费 | 按 token 计费 |
| 离线可用 | ✅ | ❌ |
