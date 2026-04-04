#!/usr/bin/env python
"""
音频评测脚本 - 多任务并行标注
对 audio/ 目录下的音频进行 ASR/EMO/AGE/GND/TONE 标注

运行环境:
- 基础标注 (ASR, EMO, AGE, GND): conda activate audio_paraling
- TONE 标注需要: conda activate Audio-Reasoner (或安装 swift)

使用方法:
    # 完整标注 (包含TONE，需要Audio-Reasoner环境)
    conda activate Audio-Reasoner
    python run_evaluation.py --input /path/to/audio --output /path/to/output

    # 仅基础标注 (跳过TONE)
    conda activate audio_paraling
    python run_evaluation.py --input /path/to/audio --output /path/to/output --skip-tone
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from audio_paralinguistic.config.model_config import MODEL_CONFIGS


# ================= 数据结构 =================

@dataclass
class AudioAnnotation:
    """单个音频的标注结果"""
    file: str
    model: str  # 'user' 或模型名称
    annotation: Dict[str, Any]


@dataclass
class AudioPair:
    """输入-输出配对"""
    input_audio: AudioAnnotation
    output_audio: AudioAnnotation


@dataclass
class DirectoryResult:
    """单个目录的评测结果"""
    category: str          # age/emotion/gender/sarcasm
    label: str             # adult/happy/male/sincere 等
    dir_name: str          # 目录名
    pairs: List[Dict[str, Any]]  # 配对列表


# ================= 标注器管理 =================

class AnnotatorManager:
    """标注器管理器 - 支持延迟加载"""

    def __init__(self, device: str = "cuda", skip_tone: bool = False):
        self.device = device
        self.skip_tone = skip_tone
        self._annotators = {}
        self._locks = {
            "SCR": threading.Lock(),
            "ER": threading.Lock(),
            "Age": threading.Lock(),
            "Gender": threading.Lock(),
            "Tone": threading.Lock(),
        }

    def _load_scr(self):
        """加载 ASR 标注器"""
        if "SCR" not in self._annotators:
            from audio_paralinguistic.annotators.scr.whisper_asr import WhisperASRAnnotator
            config = MODEL_CONFIGS["SCR"].copy()
            config['device'] = self.device
            self._annotators["SCR"] = WhisperASRAnnotator(config)
            self._annotators["SCR"].load_model()
        return self._annotators["SCR"]

    def _load_er(self):
        """加载情感识别标注器"""
        if "ER" not in self._annotators:
            from audio_paralinguistic.annotators.er.hubert_emotion import Emotion2VecAnnotator
            config = MODEL_CONFIGS["ER"].copy()
            config['device'] = self.device
            self._annotators["ER"] = Emotion2VecAnnotator(config)
            self._annotators["ER"].load_model()
        return self._annotators["ER"]

    def _load_age(self):
        """加载年龄分类器"""
        if "Age" not in self._annotators:
            from audio_paralinguistic.annotators.sar.age_classifier import AgeClassifier
            config = MODEL_CONFIGS["Age"].copy()
            config['device'] = self.device
            self._annotators["Age"] = AgeClassifier(config)
            self._annotators["Age"].load_model()
        return self._annotators["Age"]

    def _load_gender(self):
        """加载性别分类器"""
        if "Gender" not in self._annotators:
            from audio_paralinguistic.annotators.sar.gender_classifier import GenderClassifier
            config = MODEL_CONFIGS["Gender"].copy()
            config['device'] = self.device
            self._annotators["Gender"] = GenderClassifier(config)
            self._annotators["Gender"].load_model()
        return self._annotators["Gender"]

    def _load_tone(self):
        """加载语气分析器"""
        if "Tone" not in self._annotators:
            from audio_paralinguistic.annotators.sar.tone_annotator import ToneAnnotator
            config = MODEL_CONFIGS["Tone"].copy()
            config['device'] = self.device
            self._annotators["Tone"] = ToneAnnotator(config)
            self._annotators["Tone"].load_model()
        return self._annotators["Tone"]

    def load_all(self):
        """加载所有标注器"""
        print("\n[AnnotatorManager] Loading all annotators...")
        self._load_scr()
        self._load_er()
        self._load_age()
        self._load_gender()
        if not self.skip_tone:
            self._load_tone()
        print("[AnnotatorManager] All annotators loaded!")

    def annotate_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        对单个音频进行完整标注

        Returns:
            {
                "ASR": "文本内容",
                "EMO": {"emotion": "happy", "confidence": 0.9},
                "AGE": {"age_group": "adult", "confidence": 0.8},
                "GND": {"gender": "male", "confidence": 0.95},
                "TONE": {"description": "..."}
            }
        """
        result = {
            "ASR": "",
            "EMO": {},
            "AGE": {},
            "GND": {},
            "TONE": {}
        }

        # ASR (SCR)
        try:
            with self._locks["SCR"]:
                scr = self._load_scr()
                scr_result = scr.annotate(audio_path)
                if "predictions" in scr_result:
                    trans = scr_result["predictions"].get("transcription", {})
                    result["ASR"] = trans.get("text", "")
        except Exception as e:
            print(f"  [SCR Error] {e}")
            result["ASR"] = f"[Error: {e}]"

        # Emotion (ER)
        try:
            with self._locks["ER"]:
                er = self._load_er()
                er_result = er.annotate(audio_path)
                if "predictions" in er_result:
                    discrete = er_result["predictions"].get("discrete", {})
                    result["EMO"] = {
                        "emotion": discrete.get("primary_emotion", "unknown"),
                        "confidence": round(discrete.get("confidence", 0), 3),
                        "distribution": discrete.get("emotion_distribution", {})
                    }
        except Exception as e:
            print(f"  [ER Error] {e}")
            result["EMO"] = {"error": str(e)}

        # Age
        try:
            with self._locks["Age"]:
                age = self._load_age()
                age_result = age.annotate(audio_path)
                if "predictions" in age_result:
                    pred = age_result["predictions"]
                    result["AGE"] = {
                        "age_group": pred.get("age_group", "unknown"),
                        "confidence": round(pred.get("confidence", 0), 3)
                    }
        except Exception as e:
            print(f"  [Age Error] {e}")
            result["AGE"] = {"error": str(e)}

        # Gender
        try:
            with self._locks["Gender"]:
                gender = self._load_gender()
                gender_result = gender.annotate(audio_path)
                if "predictions" in gender_result:
                    pred = gender_result["predictions"]
                    result["GND"] = {
                        "gender": pred.get("gender", "unknown"),
                        "confidence": round(pred.get("confidence", 0), 3)
                    }
        except Exception as e:
            print(f"  [Gender Error] {e}")
            result["GND"] = {"error": str(e)}

        # Tone
        if self.skip_tone:
            result["TONE"] = {"description": "skipped"}
        else:
            try:
                with self._locks["Tone"]:
                    tone = self._load_tone()
                    tone_result = tone.annotate(audio_path)
                    if "predictions" in tone_result:
                        pred = tone_result["predictions"]
                        result["TONE"] = {
                            "description": pred.get("tone_description", "")
                        }
            except Exception as e:
                print(f"  [Tone Error] {e}")
                result["TONE"] = {"error": str(e)}

        return result


# ================= 目录扫描与配对 =================

def parse_dir_name(dir_name: str) -> Tuple[str, str]:
    """
    解析目录名，提取 category 和 label

    Examples:
        "04-18-05-07_84_adult" -> ("age", "adult")
        "04-24-22-59_15_happy" -> ("emotion", "happy")
        "04-19-04-42_10_female" -> ("gender", "female")
        "04-23-03-22_27_sincere" -> ("sarcasm", "sincere")
    """
    parts = dir_name.split("_")
    if len(parts) >= 3:
        label = parts[-1]  # 最后一部分是标签
    else:
        label = parts[-1]

    # 根据 label 推断 category
    age_labels = ["adult", "littlekid", "child", "elderly", "teenager"]
    emotion_labels = ["happy", "sad", "angry", "fearful", "disgust", "surprised", "neutral"]
    gender_labels = ["male", "female"]
    sarcasm_labels = ["sincere", "sarcastic"]

    if label in age_labels:
        return "age", label
    elif label in emotion_labels:
        return "emotion", label
    elif label in gender_labels:
        return "gender", label
    elif label in sarcasm_labels:
        return "sarcasm", label
    else:
        return "unknown", label


def scan_audio_directory(audio_dir: Path) -> List[Dict[str, Any]]:
    """
    扫描音频目录，返回配对列表

    Returns:
        [
            {
                "category": "age",
                "label": "adult",
                "dir_name": "04-18-05-07_84_adult",
                "dir_path": "/path/to/dir",
                "user_audio": "user.wav",
                "model_audios": ["glm4.wav", "gpt-4o-voice-mode.wav", ...]
            },
            ...
        ]
    """
    results = []

    for category_dir in audio_dir.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name  # age/emotion/gender/sarcasm

        for sample_dir in category_dir.iterdir():
            if not sample_dir.is_dir():
                continue

            dir_name = sample_dir.name
            _, label = parse_dir_name(dir_name)

            # 查找音频文件
            audio_files = list(sample_dir.glob("*.wav"))

            user_audio = None
            model_audios = []

            for af in audio_files:
                if af.name == "user.wav":
                    user_audio = af.name
                else:
                    model_audios.append(af.name)

            if user_audio and model_audios:
                results.append({
                    "category": category_name,
                    "label": label,
                    "dir_name": dir_name,
                    "dir_path": str(sample_dir),
                    "user_audio": user_audio,
                    "model_audios": sorted(model_audios)
                })

    return results


# ================= 并行处理 =================

def process_single_pair(
    dir_info: Dict[str, Any],
    model_audio: str,
    annotator_manager: AnnotatorManager
) -> Dict[str, Any]:
    """处理单个配对 (user + model_output)"""
    dir_path = Path(dir_info["dir_path"])

    # 标注 user 音频
    user_audio_path = dir_path / dir_info["user_audio"]
    user_annotation = annotator_manager.annotate_audio(str(user_audio_path))

    # 标注模型输出音频
    model_audio_path = dir_path / model_audio
    model_name = model_audio.replace(".wav", "")
    model_annotation = annotator_manager.annotate_audio(str(model_audio_path))

    return {
        "input": {
            "file": dir_info["user_audio"],
            "model": "user",
            "annotation": user_annotation
        },
        "output": {
            "file": model_audio,
            "model": model_name,
            "annotation": model_annotation
        }
    }


def process_directory(
    dir_info: Dict[str, Any],
    annotator_manager: AnnotatorManager,
    num_workers: int = 4
) -> DirectoryResult:
    """处理单个目录的所有配对"""
    pairs = []

    print(f"\n[Processing] {dir_info['dir_name']} ({dir_info['category']}/{dir_info['label']})")
    print(f"  Model audios: {len(dir_info['model_audios'])}")

    # 并行处理每个模型配对
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_pair,
                dir_info,
                model_audio,
                annotator_manager
            ): model_audio
            for model_audio in dir_info["model_audios"]
        }

        for future in as_completed(futures):
            model_audio = futures[future]
            try:
                pair_result = future.result()
                pairs.append(pair_result)
                print(f"  ✓ Completed: {model_audio}")
            except Exception as e:
                print(f"  ✗ Failed: {model_audio} - {e}")

    return DirectoryResult(
        category=dir_info["category"],
        label=dir_info["label"],
        dir_name=dir_info["dir_name"],
        pairs=pairs
    )


# ================= 主流程 =================

def run_evaluation(
    audio_dir: str,
    output_dir: str,
    device: str = "cuda",
    num_workers: int = 4,
    skip_tone: bool = False
):
    """
    运行完整评测流程

    Args:
        audio_dir: 音频根目录
        output_dir: 输出目录
        device: 计算设备
        num_workers: 并行worker数
        skip_tone: 是否跳过TONE标注
    """
    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Audio Evaluation Pipeline")
    print("=" * 60)
    print(f"Input:  {audio_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print(f"Workers: {num_workers}")
    print(f"Skip TONE: {skip_tone}")

    # 1. 扫描目录
    print("\n[Step 1] Scanning audio directories...")
    dir_list = scan_audio_directory(audio_dir)
    print(f"Found {len(dir_list)} directories to process")

    # 2. 加载标注器
    print("\n[Step 2] Loading annotators...")
    annotator_manager = AnnotatorManager(device=device, skip_tone=skip_tone)
    annotator_manager.load_all()

    # 3. 处理每个目录
    print("\n[Step 3] Processing directories...")
    all_results = []

    for dir_info in dir_list:
        result = process_directory(dir_info, annotator_manager, num_workers)
        all_results.append(asdict(result))

        # 保存单个目录结果
        category_dir = output_dir / result.category
        category_dir.mkdir(parents=True, exist_ok=True)

        result_path = category_dir / f"{result.dir_name}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"  Saved: {result_path}")

    # 4. 保存汇总结果
    print("\n[Step 4] Saving summary...")
    summary_path = output_dir / "summary.json"

    # 统计信息
    summary = {
        "total_directories": len(all_results),
        "categories": {},
        "results": all_results
    }

    for result in all_results:
        cat = result["category"]
        if cat not in summary["categories"]:
            summary["categories"][cat] = {"count": 0, "labels": {}}
        summary["categories"][cat]["count"] += 1

        label = result["label"]
        if label not in summary["categories"][cat]["labels"]:
            summary["categories"][cat]["labels"][label] = 0
        summary["categories"][cat]["labels"][label] += 1

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"{'=' * 60}")
    print(f"Total directories: {len(all_results)}")
    print(f"Summary saved to: {summary_path}")

    # 打印统计
    print("\nStatistics:")
    for cat, info in summary["categories"].items():
        print(f"  {cat}: {info['count']} directories")
        for label, count in info["labels"].items():
            print(f"    - {label}: {count}")


# ================= 入口 =================

def main():
    parser = argparse.ArgumentParser(description="Audio Evaluation Pipeline")
    parser.add_argument(
        "--input", "-i",
        default="/home/u2023112559/qix/Project/Final_Project/Audio_Captior/audio",
        help="Input audio directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="/home/u2023112559/qix/Project/Final_Project/Audio_Captior/evaluation_results",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--skip-tone",
        action="store_true",
        help="Skip TONE annotation (requires Audio-Reasoner environment)"
    )

    args = parser.parse_args()

    run_evaluation(
        audio_dir=args.input,
        output_dir=args.output,
        device=args.device,
        num_workers=args.workers,
        skip_tone=args.skip_tone
    )


if __name__ == "__main__":
    main()
