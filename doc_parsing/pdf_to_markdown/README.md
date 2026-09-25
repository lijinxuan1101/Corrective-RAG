# ① PDF → Markdown

用 [MinerU](https://github.com/opendatalab/MinerU) 4.x 把 `data/China/*.pdf` 批量解析成 `data/markdown/*.md`。

```
convert.sh        批量转换脚本（幂等）
```

## 🚀 用法

```bash
conda activate byte
./doc_parsing/pdf_to_markdown/convert.sh                    # 只转未处理的
FORCE=1 ./doc_parsing/pdf_to_markdown/convert.sh            # 全部重转
CONDA_ENV=other INPUT_DIR=data/US OUTPUT_DIR=data/md_us \
    ./doc_parsing/pdf_to_markdown/convert.sh
```

脚本会自动检查 `mineru` 是否可用、按需启动本地服务、确认本地解析模式已启用，然后逐个转换。已有 `.md` 的文件默认跳过，所以新增年报时只处理增量。

## ⚙️ 首次环境准备

模型约 2.6GB，存放在 `~/.mineru`，**与 conda 环境无关**，换环境不必重下。

```bash
pip install -U "mineru[core]"
mineru-kit models download --tier standard --source huggingface
mineru server start
mineru config set parse_server.local.mode managed
```

## 📤 输出特征

四份年报都是**文字版 PDF，不需要 OCR**（每份几百个 `/Font` 引用，图像只有零星封面 logo）。转换 749 页耗时 8 分 37 秒，产出 2MB Markdown。

MinerU 的输出有三个对 RAG 很有用的特性：

- **标题还原成 `#` 层级** — 这是下游按结构切块的基础。
- **每页插入 `<!-- page N of M -->` 标记** — 可解析成页码元数据，让答案能标注出处。
- **表格分层处理** — 简单表格转成 Markdown 管道表；带合并单元格的复杂表格保留为 HTML `<table>` 并附 `rowspan`/`colspan`。后者看着不清爽，但对财报的多层表头是正确选择，强行压平会丢失"期末余额横跨 5 列"这类信息。

输出的原始 Markdown 仍有噪声（重复页眉、页码、跨页断表、私有区字符），由下一阶段 [`chunking/`](../chunking/) 统一清洗。
