#!/usr/bin/env python3
"""
语音多任务标注系统 - 主入口

支持任务:
- LowLevel: 低级声学特征提取 (Spectral, Prosody, Energy, Temporal, Timbre)
- Embeddings: 深度表征 (wav2vec2, HuBERT, CLAP)
- SCR: Speech Content Reasoning (ASR)
- SpER: Speech Entity Recognition
- SED: Sound Event Detection
- ER: Emotion Recognition
- SAR: Speaker Attribute Recognition

输出结构 (三层嵌套):
- Top Level: audio_id, file_path, content_metadata, acoustic_features
- acoustic_features:
  - low_level: 基础物理特征
  - embeddings: 模型中层表示
  - high_level: 任务标签 (ER/SED/SAR)

使用方法:
    # 单音频标注
    python main.py --mode single --input audio.wav --output ./output/

    # 批量标注
    python main.py --mode batch --input ./audio/ --output ./output/

    # 指定任务
    python main.py --mode single --input audio.wav --tasks ER SED

    # 测试模式
    python main.py --mode test --input audio.wav
"""
import argparse
import os
import sys
import json
from pathlib import Path

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from audio_paralinguistic.config.model_config import MODEL_CONFIGS
from audio_paralinguistic.core.pipeline import create_pipeline, AnnotationPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio Multi-Task Annotation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mode",
        choices=["single", "batch", "test", "list"],
        default="test",
        help="运行模式"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="输入音频文件或目录"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./data/annotations",
        help="输出目录"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["LowLevel", "Embeddings", "SCR", "SpER", "SED", "ER", "SAR", "all"],
        default=["all"],
        help="指定任务"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="计算设备"
    )

    parser.add_argument(
        "--no-load",
        action="store_true",
        help="不加载模型（仅测试流程）"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 处理任务列表
    if args.tasks == ["all"]:
        tasks = ["LowLevel", "Embeddings", "SCR", "SpER", "SED", "ER", "SAR"]
    else:
        tasks = args.tasks

    print("=" * 60)
    print("Audio Multi-Task Annotation System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Tasks: {tasks}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print("=" * 60)

    if args.mode == "list":
        # 列出可用模型
        print("\nAvailable Tasks and Models:")
        for task, config in MODEL_CONFIGS.items():
            print(f"  {task}: {config.get('model_name', 'unknown')}")
            print(f"    Path: {config.get('model_path', 'N/A')}")
        return

    # 创建Pipeline
    print("\n[Step 1] Creating pipeline...")
    pipeline = create_pipeline(
        tasks=tasks,
        output_dir=args.output,
        device=args.device
    )

    # 加载模型
    if not args.no_load:
        print("\n[Step 2] Loading models...")
        pipeline.load_all_models()
    else:
        print("\n[Step 2] Skipping model loading (--no-load)")

    # 执行标注
    print("\n[Step 3] Running annotation...")

    if args.mode == "test":
        # 测试模式：使用默认测试音频或指定音频
        if args.input:
            audio_path = args.input
        else:
            # 查找测试音频
            test_audio = PROJECT_ROOT / "data" / "input" / "test.wav"
            if test_audio.exists():
                audio_path = str(test_audio)
            else:
                print("No test audio found. Please specify --input")
                return

        result = pipeline.annotate_single(audio_path, tasks)
        print("\n" + "=" * 60)
        print("Result Summary:")
        print(f"  audio_id: {result.get('audio_id', 'N/A')}")
        print(f"  file_path: {result.get('file_path', 'N/A')}")
        print(f"  acoustic_features keys: {list(result.get('acoustic_features', {}).keys())}")

    elif args.mode == "single":
        if not args.input:
            print("Error: --input required for single mode")
            return

        result = pipeline.annotate_single(args.input, tasks)
        print("\n" + "=" * 60)
        print("Result saved!")

    elif args.mode == "batch":
        if not args.input:
            print("Error: --input required for batch mode")
            return

        pipeline.annotate_batch(args.input, args.output, tasks)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
