#!/usr/bin/env python3
"""
批量测试Pipeline
从PASM_Lite数据集随机选取100条数据进行测试
"""
import os
import sys
import json
import time
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

# 设置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from audio_paralinguistic.core.pipeline import create_pipeline


def run_batch_test(
    data_dir: str,
    output_dir: str,
    num_samples: int = 100,
    tasks: list = None,
    device: str = "cuda"
):
    """运行批量测试"""

    # 获取所有音频文件
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob("*.mp3"))
    print(f"Total files in dataset: {len(all_files)}")

    # 随机选取
    random.seed(42)
    selected_files = random.sample(all_files, min(num_samples, len(all_files)))
    print(f"Selected {len(selected_files)} files for testing")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建pipeline
    tasks = tasks or ["LowLevel", "ER", "SED", "SAR"]
    print(f"\nTasks: {tasks}")

    pipeline = create_pipeline(
        tasks=tasks,
        output_dir=str(output_dir),
        device=device
    )

    # 加载模型
    print("\nLoading models...")
    pipeline.load_all_models()

    # 批量处理
    print(f"\nProcessing {len(selected_files)} files...")
    start_time = time.time()

    results = []
    errors = []

    for i, audio_file in enumerate(selected_files, 1):
        try:
            print(f"[{i}/{len(selected_files)}] {audio_file.name}", end=" ")
            result = pipeline.annotate_single(str(audio_file), tasks, save_individual=False)
            results.append(result)

            # 提取关键信息
            emotion = result.get("acoustic_features", {}).get("high_level", {}).get("emotion", {})
            speaker = result.get("acoustic_features", {}).get("high_level", {}).get("speaker", {})

            print(f"emotion={emotion.get('primary_emotion', 'N/A')} "
                  f"gender={speaker.get('gender', {}).get('label', 'N/A')}")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({"file": str(audio_file), "error": str(e)})

    total_time = time.time() - start_time

    # 统计分析
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    analysis = analyze_results(results, total_time)

    # 保存结果
    report = {
        "test_info": {
            "dataset": str(data_dir),
            "num_samples": len(selected_files),
            "tasks": tasks,
            "total_time": round(total_time, 2),
            "avg_time_per_file": round(total_time / len(selected_files), 2),
            "timestamp": datetime.now().isoformat()
        },
        "errors": errors,
        "analysis": analysis
    }

    # 保存报告
    report_path = output_dir / "batch_test_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {report_path}")

    # 保存所有结果
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"All results saved to: {all_results_path}")

    return report


def analyze_results(results: list, total_time: float) -> dict:
    """分析结果统计"""

    analysis = {
        "total_files": len(results),
        "success_rate": 0,
        "emotion": {
            "distribution": {},
            "confidence_stats": {"mean": 0, "min": 0, "max": 0}
        },
        "speaker": {
            "gender_distribution": {},
            "age_distribution": {}
        },
        "events": {
            "speech_detected_rate": 0,
            "top_events": {}
        },
        "low_level": {
            "f0_stats": {"mean": 0, "std": 0, "min": 0, "max": 0},
            "duration_stats": {"mean": 0, "min": 0, "max": 0}
        }
    }

    if not results:
        return analysis

    # 情感统计
    emotions = []
    emotion_confidences = []
    emotion_ids = []

    # 说话人统计
    genders = []
    ages = []

    # 事件统计
    speech_detected_count = 0
    event_counter = Counter()

    # Low-level统计
    f0_means = []
    durations = []

    for r in results:
        hl = r.get("acoustic_features", {}).get("high_level", {})
        ll = r.get("acoustic_features", {}).get("low_level", {})

        # 情感
        emotion = hl.get("emotion", {})
        if emotion.get("primary_emotion"):
            emotions.append(emotion["primary_emotion"])
            emotion_confidences.append(emotion.get("confidence", 0))
            emotion_ids.append(emotion.get("emotion_id", 8))

        # 说话人
        speaker = hl.get("speaker", {})
        gender = speaker.get("gender", {})
        if gender.get("label"):
            genders.append(gender["label"])
        age = speaker.get("age", {})
        if age.get("label"):
            ages.append(age["label"])

        # 事件
        events = hl.get("events", {})
        if events.get("prob_summary", {}).get("speech_detected"):
            speech_detected_count += 1
        primary_event = events.get("primary_event")
        if primary_event and primary_event != "unknown":
            event_counter[primary_event] += 1

        # Low-level
        prosody = ll.get("prosody", {})
        f0 = prosody.get("f0", {})
        if f0.get("mean_hz"):
            f0_means.append(f0["mean_hz"])

        temporal = ll.get("temporal", {})
        duration = temporal.get("duration", {})
        if duration.get("total_seconds"):
            durations.append(duration["total_seconds"])

    # 计算统计
    analysis["success_rate"] = len(results) / analysis["total_files"] * 100

    # 情感分布
    emotion_counter = Counter(emotions)
    analysis["emotion"]["distribution"] = dict(emotion_counter)
    if emotion_confidences:
        analysis["emotion"]["confidence_stats"] = {
            "mean": round(sum(emotion_confidences) / len(emotion_confidences), 3),
            "min": round(min(emotion_confidences), 3),
            "max": round(max(emotion_confidences), 3)
        }

    # 说话人分布
    analysis["speaker"]["gender_distribution"] = dict(Counter(genders))
    analysis["speaker"]["age_distribution"] = dict(Counter(ages))

    # 事件分布
    analysis["events"]["speech_detected_rate"] = round(speech_detected_count / len(results) * 100, 1)
    analysis["events"]["top_events"] = dict(event_counter.most_common(10))

    # Low-level统计
    if f0_means:
        import numpy as np
        analysis["low_level"]["f0_stats"] = {
            "mean": round(np.mean(f0_means), 1),
            "std": round(np.std(f0_means), 1),
            "min": round(min(f0_means), 1),
            "max": round(max(f0_means), 1)
        }
    if durations:
        analysis["low_level"]["duration_stats"] = {
            "mean": round(sum(durations) / len(durations), 2),
            "min": round(min(durations), 2),
            "max": round(max(durations), 2)
        }

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/home/u2023112559/qix/datasets/PASM_Lite")
    parser.add_argument("--output_dir", type=str, default="/home/u2023112559/qix/Project/Audio_Captior/data/batch_test_results")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    run_batch_test(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        device=args.device
    )
