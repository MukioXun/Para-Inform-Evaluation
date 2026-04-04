#!/bin/bash
# 完整音频评测流程
#
# 分三步运行:
#   Step 1: 基础标注 (ASR, AGE, GND) - audio_paraling 环境
#   Step 2: EMO 标注 - audio_paraling 环境
#   Step 3: TONE 标注 - Audio-Reasoner 环境
#
# 使用方法:
#   chmod +x run_eval.sh
#   ./run_eval.sh
#
# 可选环境变量:
#   SKIP_BASE=1    跳过基础标注
#   SKIP_EMO=1     跳过EMO标注
#   SKIP_TONE=1    跳过TONE标注
#   WORKERS=8      设置并行线程数 (默认4)

set -e

SCRIPT_DIR=$(dirname "$0")
AUDIO_DIR="/home/u2023112559/qix/Project/Final_Project/Audio_Captior/audio"
OUTPUT_DIR="/home/u2023112559/qix/Project/Final_Project/Audio_Captior/evaluation_results"

# 参数
WORKERS=${WORKERS:-4}
LIMIT=${LIMIT:-0}

echo "============================================"
echo "Audio Evaluation Pipeline (Full)"
echo "============================================"
echo "Audio:   $AUDIO_DIR"
echo "Output:  $OUTPUT_DIR"
echo "Workers: $WORKERS"
echo ""

# ================= Step 1: 基础标注 =================
if [ "${SKIP_BASE:-0}" != "1" ]; then
    echo "[Step 1] Running base annotations (ASR, AGE, GND)..."

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate audio_paraling

    python "$SCRIPT_DIR/run_evaluation.py" \
        --input "$AUDIO_DIR" \
        --output "$OUTPUT_DIR" \
        --device cuda \
        --workers $WORKERS \
        --skip-tone

    echo ""
    echo "[Step 1] Base annotations complete!"
    echo ""
else
    echo "[Step 1] Skipped (SKIP_BASE=1)"
    echo ""
fi

# ================= Step 2: EMO 标注 =================
if [ "${SKIP_EMO:-0}" != "1" ]; then
    echo "[Step 2] Running EMO annotations..."

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate audio_paraling

    EMO_ARGS="--input $OUTPUT_DIR --audio $AUDIO_DIR --workers $WORKERS"

    if [ "$LIMIT" -gt 0 ]; then
        EMO_ARGS="$EMO_ARGS --limit $LIMIT"
    fi

    python "$SCRIPT_DIR/run_emo_annotation.py" $EMO_ARGS

    echo ""
    echo "[Step 2] EMO annotations complete!"
    echo ""
else
    echo "[Step 2] Skipped (SKIP_EMO=1)"
    echo ""
fi

# ================= Step 3: TONE 标注 =================
if [ "${SKIP_TONE:-0}" != "1" ]; then
    echo "[Step 3] Running TONE annotations..."

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate Audio-Reasoner

    TONE_ARGS="--input $OUTPUT_DIR --audio $AUDIO_DIR --workers $WORKERS"

    if [ "$LIMIT" -gt 0 ]; then
        TONE_ARGS="$TONE_ARGS --limit $LIMIT"
    fi

    python "$SCRIPT_DIR/run_tone_annotation.py" $TONE_ARGS

    echo ""
    echo "[Step 3] TONE annotations complete!"
    echo ""
else
    echo "[Step 3] Skipped (SKIP_TONE=1)"
    echo ""
fi

# ================= 完成 =================
echo "============================================"
echo "Evaluation Complete!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""

# 统计
if [ -f "$OUTPUT_DIR/summary.json" ]; then
    echo "Statistics:"
    python -c "
import json
with open('$OUTPUT_DIR/summary.json', 'r') as f:
    data = json.load(f)
print(f'  Total directories: {data[\"total_directories\"]}')
for cat, info in data['categories'].items():
    print(f'  {cat}: {info[\"count\"]} directories')
"
fi
