# 外部 API 供应商

所有需要密钥的外部服务集中在这里，别处不直接读 `os.environ`，也不直接调 SDK。
这样换供应商或轮换密钥时只有一个地方要改。

```
providers/
├── config.py      读 .env，缺 key 时给出可操作的报错
├── rerank.py      百炼文本重排
└── llm.py         DashScope 对话模型
```

> 嵌入不在这里——它跑在本地（`retrieval/index.py` 的 `Qwen3-Embedding-0.6B`），
> 不走外部 API，原因见下文「🔄 为什么嵌入留在本地」。

## 🔑 配置

密钥放在项目根目录的 `.env`（已在 `.gitignore` 中）：

```
DASHSCOPE_API_KEY=sk-...
```

一个 key 同时覆盖重排和对话两类接口。

## 🚀 使用

```python
from providers import rerank, chat_model

scores = rerank("用户的问题", ["候选一", "候选二"])   # 顺序与输入一致

llm = chat_model("qwen-plus")
llm = chat_model("qwen-flash", temperature=0)  # 高频小任务用 flash
```

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
还省下 1.2GB 常驻内存，所以 `retrieval` 默认走它。缺 key 时**直接报错**，
不退回本地——两个模型对无关查询的打分差一个量级，静默切换会让 grade 的
`IRRELEVANT` 阈值失去意义。需要本地重排请显式 `HybridRetriever(reranker=...)`。

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
如果以后换成新加坡区的账号，需要改 `llm.py` 的 `BASE_URL`。

## 🔄 为什么嵌入留在本地

重排上了 API，嵌入却留在本地（`retrieval/index.py` 的 `Qwen3-Embedding-0.6B`），
这个不对称是刻意的——两者在管线里的位置不同：

- **嵌入在每次查询的关键路径上**。本地 0.6B 已经很快（内存点积 0.07–0.14s），
  换 API 只会给每次查询平添一个网络往返、按 token 计费、丢掉离线能力，
  而建索引时间基本打平。没有换的理由。
- **重排恰好相反**：本地 cross-encoder 慢（3–9s）又常驻 1.2GB 内存，
  API 快 4 倍还省内存，换过去是净赚。

一句话：**快的那个（嵌入）留本地，慢的那个（重排）上 API。**

| | 本地嵌入 0.6B | 若改用 DashScope API |
|---|---|---|
| 建索引（10610 条） | 11.8 分钟 | ~10 分钟 |
| 单次查询延迟 | 0.07–0.14 s | 网络往返，更慢 |
| 成本 | 免费 | 按 token 计费 |
| 离线可用 | ✅ | ❌ |

真要换成 API 嵌入，不是改个开关的事——两者向量空间不同（即便都是 1024 维），
必须整体切换并**重建索引**，否则用 API 向量搜本地建的索引只会得到噪声。
