#!/usr/bin/env bash
#
# 使用 MinerU 将 data/ 下的 PDF 年报批量转换为 Markdown。
#
# 用法:
#   ./doc_parsing/pdf_to_markdown/convert.sh           # 转换所有未处理的 PDF
#   FORCE=1 ./doc_parsing/pdf_to_markdown/convert.sh   # 强制重新转换
#   CONDA_ENV=other ./doc_parsing/pdf_to_markdown/convert.sh
#
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-byte}"
INPUT_DIR="${INPUT_DIR:-data/China}"
OUTPUT_DIR="${OUTPUT_DIR:-data/markdown}"
FORCE="${FORCE:-0}"

cd "$(dirname "$0")/../.."

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
ENV_BIN="$CONDA_BASE/envs/$CONDA_ENV/bin"

if [[ ! -x "$ENV_BIN/mineru" ]]; then
    echo "❌ 在 conda 环境 '$CONDA_ENV' 中找不到 mineru。"
    echo "   💡 请先执行: conda activate $CONDA_ENV && pip install -U 'mineru[core]'"
    exit 1
fi
export PATH="$ENV_BIN:$PATH"

# --- 1. 确保本地解析服务就绪 ---
# MinerU 4.x 默认走 mineru.net 云端，必须显式启用本地托管模式。
if ! mineru server status &>/dev/null; then
    echo "🚀 正在启动 MinerU 本地服务..."
    mineru server start
fi

if [[ "$(mineru config get parse_server.local.mode 2>/dev/null || true)" != *managed* ]]; then
    echo "⚙️  正在启用本地解析模式 (managed)..."
    mineru config set parse_server.local.mode managed
fi

# --- 2. 批量转换 ---
mkdir -p "$OUTPUT_DIR"
shopt -s nullglob
pdfs=("$INPUT_DIR"/*.pdf)
if (( ${#pdfs[@]} == 0 )); then
    echo "❌ 在 '$INPUT_DIR' 中没有找到任何 PDF 文件。"
    exit 1
fi

echo "📚 找到 ${#pdfs[@]} 个 PDF，输出目录: $OUTPUT_DIR"
converted=0
skipped=0

for pdf in "${pdfs[@]}"; do
    name="$(basename "${pdf%.pdf}")"
    out="$OUTPUT_DIR/$name.md"

    if [[ -f "$out" && "$FORCE" != "1" ]]; then
        echo "⏭️  跳过 $name (已存在，FORCE=1 可强制重转)"
        (( ++skipped ))
        continue
    fi

    echo "✂️  [$(date +%H:%M:%S)] 正在转换 $name ..."
    # -p all 不可省略，否则 MinerU 只处理前 10 页
    mineru parse "$pdf" -p all -o "$out" --wait 3600
    echo "    ✅ $(wc -c < "$out" | awk '{printf "%.0f KB", $1/1024}')"
    (( ++converted ))
done

echo "🎉 完成: 转换 $converted 个，跳过 $skipped 个。"
