#!/bin/bash
#
# Audio Pipeline 主启动脚本 (重写版)
#
# 功能:
#   阶段1: 在audio_paraling环境运行主pipeline
#   阶段2: 在Audio-Reasoner环境运行Age+Tone标注
#
# 使用方法:
#   ./run_pipeline_with_tone.sh --input /path/to/audio --output /path/to/output
#

set -e

# ============================================================
# 1. 初始化配置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
INPUT=""
OUTPUT=""
DEVICE="cuda"
LIMIT=0

# 模型路径
AGE_MODEL_PATH="/home/u2023112559/qix/Models/Models/age-classification"
TONE_MODEL_PATH="/home/u2023112559/qix/Models/Models/Audio-Reasoner"

# Conda环境
MAIN_ENV="audio_paraling"           # 主流程环境
AR_ENV="Audio-Reasoner"              # Age+Tone环境

# ============================================================
# 2. 帮助信息
# ============================================================
usage() {
    cat << EOF
Audio Pipeline - 双环境执行脚本

使用:
  $0 --input <音频目录> --output <输出目录> [选项]

选项:
  --input PATH         音频输入目录 (必填)
  --output PATH        输出目录 (必填)
  --device DEVICE      cuda或cpu (默认: cuda)
  --limit N            限制处理数量 (默认: 全部)
  --skip-ar            跳过Age+Tone标注
  --ar-only            仅运行Age+Tone标注 (需要先运行主流程)
  -h, --help           显示帮助

环境说明:
  - audio_paraling: LowLevel, ER, SED, Gender, SCR, SpER
  - Audio-Reasoner: Age, Tone

示例:
  # 完整流程
  $0 --input /datasets/PASM_Lite --output ./output

  # 仅主流程
  $0 --input /datasets/PASM_Lite --output ./output --skip-ar

  # 仅Age+Tone
  $0 --output ./output --ar-only

EOF
    exit 0
}

# ============================================================
# 3. 解析参数
# ============================================================
SKIP_AR=false
AR_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)  INPUT="$2";  shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --limit)  LIMIT="$2";  shift 2 ;;
        --skip-ar) SKIP_AR=true; shift ;;
        --ar-only) AR_ONLY=true; shift ;;
        -h|--help) usage ;;
        *) echo "未知选项: $1"; usage ;;
    esac
done

# ============================================================
# 4. 参数验证
# ============================================================
if [[ -z "$INPUT" ]] && [[ "$AR_ONLY" != "true" ]]; then
    echo "错误: --input 是必填参数"
    usage
fi

if [[ -z "$OUTPUT" ]]; then
    echo "错误: --output 是必填参数"
    usage
fi

# 转换为绝对路径
if [[ "$INPUT" != /* ]] && [[ -n "$INPUT" ]]; then
    INPUT="$(cd "$SCRIPT_DIR" && pwd)/$INPUT"
fi

if [[ "$OUTPUT" != /* ]]; then
    OUTPUT="$(cd "$SCRIPT_DIR" && pwd)/$OUTPUT"
fi

# ============================================================
# 5. 辅助函数
# ============================================================
print_header() {
    echo ""
    echo "============================================"
    echo "$1"
    echo "============================================"
}

print_info() {
    echo "[INFO] $1"
}

print_warn() {
    echo "[WARN] $1"
}

print_error() {
    echo "[ERROR] $1"
}

check_file() {
    local file="$1"
    local name="$2"
    if [[ ! -e "$file" ]]; then
        print_error "$name 不存在: $file"
        exit 1
    fi
    print_info "$name: $file"
}

check_dir() {
    local dir="$1"
    local name="$2"
    if [[ ! -d "$dir" ]]; then
        print_error "$name 目录不存在: $dir"
        exit 1
    fi
    print_info "$name: $dir"
}

# ============================================================
# 6. 主流程：阶段1 - 运行主Pipeline
# ============================================================
run_stage1_main_pipeline() {
    print_header "阶段1: 主Pipeline (环境: $MAIN_ENV)"

    # 验证输入
    check_dir "$INPUT" "输入音频目录"
    check_dir "$(dirname "$AGE_MODEL_PATH")" "Age模型目录"
    check_dir "$(dirname "$TONE_MODEL_PATH")" "Tone模型目录"

    print_info "输入目录: $INPUT"
    print_info "输出目录: $OUTPUT"
    print_info "处理数量: ${LIMIT:-全部}"
    print_info "设备: $DEVICE"

    # 创建输出目录
    print_info "创建输出目录..."
    mkdir -p "$OUTPUT/merged"

    # 构建命令
    CMD="cd $PROJECT_ROOT && python scripts/run_test_batch.py"
    CMD+=" --input $INPUT"
    CMD+=" --output $OUTPUT"
    CMD+=" --device $DEVICE"
    CMD+=" --skip-tone"
    CMD+=" --skip-age"

    if [[ -n "$LIMIT" ]] && [[ "$LIMIT" -gt 0 ]]; then
        CMD+=" --limit $LIMIT"
    fi

    echo ""
    print_info "执行命令:"
    echo "  $CMD"
    echo ""

    # 确认继续
    read -p "确认继续? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        print_warn "已取消"
        exit 0
    fi

    # 激活环境并运行
    print_info "激活环境: $MAIN_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$MAIN_ENV"

    # 执行
    echo ""
    print_info "开始执行..."
    eval "$CMD"

    print_info "阶段1完成!"
}

# ============================================================
# 7. 主流程：阶段2 - 运行Age+Tone标注
# ============================================================
run_stage2_ar_annotation() {
    print_header "阶段2: Age+Tone标注 (环境: $AR_ENV)"

    local merged_dir="$OUTPUT/merged"

    # 验证阶段1输出
    check_dir "$merged_dir" "Merged结果目录"

    # 查找一个样本文件验证
    local sample_file=$(ls "$merged_dir"/*.json 2>/dev/null | head -1)
    if [[ -z "$sample_file" ]]; then
        print_error "merged目录中没有结果文件"
        exit 1
    fi

    print_info "输出目录: $OUTPUT"
    print_info "Merged目录: $merged_dir"
    print_info "样本文件: $(basename $sample_file)"

    # 构建命令
    CMD="cd $PROJECT_ROOT && python scripts/run_sar_audio_reasoner.py"
    CMD+=" --input $merged_dir"
    CMD+=" --audio-dirs $INPUT"
    CMD+=" --age-model $AGE_MODEL_PATH"
    CMD+=" --tone-model $TONE_MODEL_PATH"
    CMD+=" --device $DEVICE"

    if [[ -n "$LIMIT" ]] && [[ "$LIMIT" -gt 0 ]]; then
        CMD+=" --limit $LIMIT"
    fi

    echo ""
    print_info "执行命令:"
    echo "  $CMD"
    echo ""

    # 确认继续
    read -p "确认继续? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        print_warn "已取消"
        exit 0
    fi

    # 激活环境并运行
    print_info "激活环境: $AR_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$AR_ENV"

    # 执行
    echo ""
    print_info "开始执行..."
    eval "$CMD"

    print_info "阶段2完成!"
}

# ============================================================
# 8. 仅运行Age+Tone标注
# ============================================================
run_ar_only() {
    print_header "仅运行Age+Tone标注 (环境: $AR_ENV)"

    local merged_dir="$OUTPUT/merged"

    # 验证
    if [[ -z "$OUTPUT" ]]; then
        print_error "--output 是必填参数"
        exit 1
    fi

    check_dir "$merged_dir" "Merged结果目录"

    print_info "输出目录: $OUTPUT"
    print_info "Merged目录: $merged_dir"

    # 构建命令
    CMD="cd $PROJECT_ROOT && python scripts/run_sar_audio_reasoner.py"
    CMD+=" --input $merged_dir"

    # 如果输入存在，添加音频目录
    if [[ -n "$INPUT" ]] && [[ -d "$INPUT" ]]; then
        CMD+=" --audio-dirs $INPUT"
    fi

    CMD+=" --age-model $AGE_MODEL_PATH"
    CMD+=" --tone-model $TONE_MODEL_PATH"
    CMD+=" --device $DEVICE"

    if [[ -n "$LIMIT" ]] && [[ "$LIMIT" -gt 0 ]]; then
        CMD+=" --limit $LIMIT"
    fi

    echo ""
    print_info "执行命令:"
    echo "  $CMD"
    echo ""

    # 确认
    read -p "确认继续? (y/n): " confirm
    if [[ "$confirm" != "y" ]]; then
        print_warn "已取消"
        exit 0
    fi

    # 激活环境并运行
    print_info "激活环境: $AR_ENV"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$AR_ENV"

    eval "$CMD"

    print_info "Age+Tone标注完成!"
}

# ============================================================
# 9. 主入口
# ============================================================
main() {
    echo ""
    echo "============================================"
    echo "  Audio Pipeline - 双环境执行"
    echo "============================================"
    echo ""
    print_info "项目根目录: $PROJECT_ROOT"
    print_info "主流程环境: $MAIN_ENV"
    print_info "Age+Tone环境: $AR_ENV"

    if [[ "$AR_ONLY" == "true" ]]; then
        # 仅运行Age+Tone
        run_ar_only
    else
        # 阶段1: 主Pipeline
        run_stage1_main_pipeline

        # 阶段2: Age+Tone (如果未跳过)
        if [[ "$SKIP_AR" != "true" ]]; then
            run_stage2_ar_annotation
        else
            print_warn "跳过Age+Tone标注"
        fi
    fi

    echo ""
    echo "============================================"
    echo "  全部完成!"
    echo "============================================"
    echo ""
    print_info "结果保存在: $OUTPUT/merged"
    echo ""
}

# 运行
main
