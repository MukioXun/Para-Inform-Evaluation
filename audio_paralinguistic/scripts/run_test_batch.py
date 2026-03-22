#!/usr/bin/env python3
"""
批量测试脚本 - 支持限制处理数量和时间戳输出目录

使用方法:
    python run_test_batch.py --input /datasets/PASM_Lite --limit 20
    python run_test_batch.py --input /datasets/PASM_Lite --limit 100 --output ./custom_output
"""
import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from audio_paralinguistic.config.model_config import MODEL_CONFIGS
from audio_paralinguistic.core.pipeline import create_pipeline, AnnotationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Test with Timestamp")

    parser.add_argument(
        "--input",
        type=str,
        default="/datasets/PASM_Lite",
        help="输入音频目录"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录 (默认: data/experiments/exp_TIMESTAMP)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="限制处理数量 (0=全部)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="计算设备"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["LowLevel", "ER", "SED", "SAR", "SCR"],
        help="任务列表"
    )

    parser.add_argument(
        "--skip-tone",
        action="store_true",
        help="跳过Tone标注 (SAR中)"
    )

    parser.add_argument(
        "--skip-age",
        action="store_true",
        help="跳过Age标注 (SAR中)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置输出目录
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = PROJECT_ROOT / "data" / "experiments" / f"exp_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存实验配置
    experiment_config = {
        "timestamp": timestamp,
        "input": args.input,
        "output": str(output_dir),
        "limit": args.limit,
        "device": args.device,
        "tasks": args.tasks,
        "skip_tone": args.skip_tone,
        "skip_age": args.skip_age
    }

    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print("Audio Multi-Task Annotation - Batch Test")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print(f"Limit: {args.limit}")
    print(f"Device: {args.device}")
    print(f"Tasks: {args.tasks}")
    print(f"Skip Tone: {args.skip_tone}")
    print(f"Skip Age: {args.skip_age}")
    print("=" * 70)

    # 获取音频文件
    audio_dir = Path(args.input)
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
        audio_files.extend(audio_dir.glob(ext))

    audio_files = sorted(audio_files)

    if args.limit > 0:
        audio_files = audio_files[:args.limit]

    print(f"\nFound {len(audio_files)} audio files to process")

    # 修改SAR配置，跳过tone和age
    if args.skip_tone and "SAR" in args.tasks:
        MODEL_CONFIGS["SAR"]["enable_tone"] = False

    if args.skip_age and "SAR" in args.tasks:
        MODEL_CONFIGS["SAR"]["enable_age"] = False

    # 创建Pipeline
    print("\n[Step 1] Creating pipeline...")
    pipeline = create_pipeline(
        tasks=args.tasks,
        output_dir=str(output_dir),
        device=args.device
    )

    # 加载模型
    print("\n[Step 2] Loading models...")
    start_load = time.time()
    pipeline.load_all_models()
    load_time = time.time() - start_load
    print(f"Model loading time: {load_time:.2f}s")

    # 执行标注
    print("\n[Step 3] Running annotation...")
    start_annotate = time.time()

    results_summary = []
    success_count = 0
    fail_count = 0

    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")

        try:
            result = pipeline.annotate_single(str(audio_file), args.tasks)

            # 统计
            success_count += 1
            summary = {
                "audio_id": result.get("audio_id"),
                "status": "success"
            }
            results_summary.append(summary)

        except Exception as e:
            fail_count += 1
            print(f"  Error: {e}")
            results_summary.append({
                "audio_id": audio_file.stem,
                "status": "failed",
                "error": str(e)
            })

    annotate_time = time.time() - start_annotate

    # 保存统计结果
    stats = {
        "timestamp": timestamp,
        "total_files": len(audio_files),
        "success": success_count,
        "failed": fail_count,
        "load_time_seconds": round(load_time, 2),
        "annotate_time_seconds": round(annotate_time, 2),
        "avg_time_per_file": round(annotate_time / len(audio_files), 2) if audio_files else 0,
        "tasks": args.tasks
    }

    stats_path = output_dir / "experiment_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # 保存详细结果列表
    summary_path = output_dir / "results_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    # 打印统计
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Output: {output_dir}")
    print(f"Total: {len(audio_files)}, Success: {success_count}, Failed: {fail_count}")
    print(f"Model Loading: {load_time:.2f}s")
    avg_time = annotate_time / len(audio_files) if audio_files else 0
    print(f"Annotation: {annotate_time:.2f}s ({avg_time:.2f}s/file)")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    main()
