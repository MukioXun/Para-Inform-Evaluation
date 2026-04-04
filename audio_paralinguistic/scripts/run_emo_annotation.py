#!/usr/bin/env python3
"""
EMO 情感标注补充脚本 - 多线程版本
在 audio_paraling 环境中运行: conda activate audio_paraling

功能:
1. 读取已有的评测结果 (EMO 为 unknown)
2. 使用 emotion2vec 多线程标注 EMO
3. 更新结果文件

使用方法:
    conda activate audio_paraling
    python run_emo_annotation.py --input ./evaluation_results --audio /path/to/audio

    # 指定线程数
    python run_emo_annotation.py --input ./evaluation_results --audio /path/to/audio --workers 4

    # 限制处理数量 (测试用)
    python run_emo_annotation.py --input ./evaluation_results --audio /path/to/audio --limit 10
"""
import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

# 设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ================= EMO 标注器 =================

class EmotionAnnotator:
    """情感标注器 - 使用 emotion2vec_plus_large"""

    EMOTION_MAP = {
        0: "angry",
        1: "disgusted",
        2: "fearful",
        3: "happy",
        4: "neutral",
        5: "other",
        6: "sad",
        7: "surprised",
        8: "unknown"
    }

    EMOTION_CN_MAP = {
        "生气": "angry", "愤怒": "angry",
        "高兴": "happy", "开心": "happy", "快乐": "happy",
        "中性": "neutral", "平静": "neutral",
        "悲伤": "sad", "伤心": "sad", "难过": "sad",
        "恐惧": "fearful", "害怕": "fearful",
        "厌恶": "disgusted", "讨厌": "disgusted",
        "惊讶": "surprised", "吃惊": "surprised",
        "其他": "other", "未知": "unknown"
    }

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self._lock = threading.Lock()
        self._loaded = False

    def load_model(self) -> bool:
        """加载 emotion2vec 模型"""
        try:
            from funasr import AutoModel
            import librosa
            import torch
            import numpy as np

            print(f"[EMO] Loading emotion2vec from: {self.model_path}")

            self.librosa = librosa
            self.torch = torch
            self.np = np

            self.model = AutoModel(
                model=self.model_path,
                device=self.device,
                disable_update=True,
                disable_log=True
            )

            self._loaded = True
            print(f"[EMO] emotion2vec loaded successfully")
            return True

        except Exception as e:
            print(f"[EMO] Failed to load model: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行情感识别"""
        if not self.is_loaded():
            return {"emotion": "unknown", "confidence": 0.0, "error": "model not loaded"}

        try:
            # 加载音频
            wav, sr = self.librosa.load(audio_path, sr=16000)
            wav_tensor = self.torch.from_numpy(wav).unsqueeze(0).float()

            # 推理
            with self._lock:
                result = self.model.generate(input=wav_tensor, output_dir=None)

            # 解析结果
            primary_emotion = "unknown"
            confidence = 0.5
            emotion_distribution = {}

            if result and len(result) > 0:
                res = result[0]
                labels = res.get('labels', [])
                scores = res.get('scores', [])

                if labels and scores:
                    max_idx = self.np.argmax(scores)
                    raw_label = labels[max_idx]
                    confidence = float(scores[max_idx])

                    primary_emotion = self._parse_emotion_label(raw_label)

                    if primary_emotion in ['<unk>', 'unk']:
                        primary_emotion = "unknown"

                    # 构建分布
                    for i, label in enumerate(labels):
                        if i < len(scores):
                            emo = self._parse_emotion_label(label)
                            if emo not in ['<unk>', 'unk']:
                                emotion_distribution[emo] = float(scores[i])

            if not emotion_distribution:
                emotion_distribution = {primary_emotion: confidence}

            return {
                "emotion": primary_emotion,
                "confidence": round(confidence, 3),
                "distribution": {k: round(v, 3) for k, v in emotion_distribution.items()}
            }

        except Exception as e:
            print(f"[EMO] Inference failed for {audio_path}: {e}")
            return {"emotion": "error", "confidence": 0.0, "error": str(e)}

    def _parse_emotion_label(self, raw_label: str) -> str:
        """解析情感标签"""
        if '/' in raw_label:
            _, en_part = raw_label.split('/', 1)
            return en_part.lower().strip()
        else:
            label = raw_label.strip()
            if label in self.EMOTION_CN_MAP:
                return self.EMOTION_CN_MAP[label]
            return label.lower()


# ================= 数据处理函数 =================

@dataclass
class AudioTask:
    """单个音频标注任务"""
    audio_path: str
    is_user: bool


@dataclass
class PairMapping:
    """配对映射关系"""
    result_file: Path
    pair_index: int
    is_input: bool


def scan_evaluation_results(input_dir: Path, audio_base_dir: Path) -> Tuple[List[AudioTask], Dict[str, List[PairMapping]]]:
    """
    扫描评测结果，提取需要标注 EMO 的音频任务（去重）

    Args:
        input_dir: 评测结果目录
        audio_base_dir: 音频文件基础目录

    Returns:
        (tasks, mapping)
    """
    tasks = []
    audio_seen = set()
    mapping: Dict[str, List[PairMapping]] = {}

    for category_dir in input_dir.iterdir():
        if not category_dir.is_dir():
            continue

        for result_file in category_dir.glob("*.json"):
            if result_file.name == "summary.json":
                continue

            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read {result_file}: {e}")
                continue

            dir_name = data.get("dir_name", result_file.stem)
            category = data.get("category", category_dir.name)
            audio_dir = audio_base_dir / category / dir_name

            pairs = data.get("pairs", [])
            for pair_idx, pair in enumerate(pairs):
                # 输入音频
                input_info = pair.get("input", {})
                input_file = input_info.get("file", "")
                if input_file:
                    input_path = audio_dir / input_file
                    input_path_str = str(input_path)

                    if input_path.exists():
                        if input_path_str not in mapping:
                            mapping[input_path_str] = []
                        mapping[input_path_str].append(PairMapping(
                            result_file=result_file,
                            pair_index=pair_idx,
                            is_input=True
                        ))

                        if input_path_str not in audio_seen:
                            audio_seen.add(input_path_str)
                            tasks.append(AudioTask(audio_path=input_path_str, is_user=True))

                # 输出音频
                output_info = pair.get("output", {})
                output_file = output_info.get("file", "")
                if output_file:
                    output_path = audio_dir / output_file
                    output_path_str = str(output_path)

                    if output_path.exists():
                        if output_path_str not in mapping:
                            mapping[output_path_str] = []
                        mapping[output_path_str].append(PairMapping(
                            result_file=result_file,
                            pair_index=pair_idx,
                            is_input=False
                        ))

                        if output_path_str not in audio_seen:
                            audio_seen.add(output_path_str)
                            tasks.append(AudioTask(audio_path=output_path_str, is_user=False))

    return tasks, mapping


def process_task(task: AudioTask, annotator: EmotionAnnotator) -> Tuple[str, Dict]:
    """处理单个任务"""
    annotation = annotator.annotate(task.audio_path)
    return task.audio_path, annotation


def update_result_file(result_file: Path, emo_results: Dict[int, Dict]) -> bool:
    """更新结果文件，添加 EMO 标注"""
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pairs = data.get("pairs", [])
        for pair_idx, emo_data in emo_results.items():
            if pair_idx >= len(pairs):
                continue

            pair = pairs[pair_idx]

            if "input" in emo_data:
                pair.setdefault("input", {}).setdefault("annotation", {})["EMO"] = emo_data["input"]

            if "output" in emo_data:
                pair.setdefault("output", {}).setdefault("annotation", {})["EMO"] = emo_data["output"]

        tmp_path = result_file.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(result_file)

        return True

    except Exception as e:
        print(f"[Error] Failed to update {result_file}: {e}")
        return False


# ================= 主流程 =================

def run_emo_annotation(
    input_dir: str,
    audio_dir: str,
    output_dir: Optional[str],
    model_path: str,
    device: str,
    num_workers: int,
    limit: int
):
    """运行 EMO 标注"""
    input_path = Path(input_dir)
    audio_path = Path(audio_dir)

    if not input_path.exists():
        print(f"[Error] Input directory not found: {input_path}")
        return

    if not audio_path.exists():
        print(f"[Error] Audio directory not found: {audio_path}")
        return

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    print("=" * 60)
    print("EMO Annotation Pipeline (Multi-threaded)")
    print("=" * 60)
    print(f"Input:    {input_path}")
    print(f"Audio:    {audio_path}")
    print(f"Output:   {output_path}")
    print(f"Model:    {model_path}")
    print(f"Device:   {device}")
    print(f"Workers:  {num_workers}")
    print(f"Limit:    {limit if limit > 0 else 'unlimited'}")

    # 1. 加载模型
    print("\n[Step 1] Loading emotion2vec...")
    annotator = EmotionAnnotator(model_path, device)
    if not annotator.load_model():
        print("[Error] Failed to load model, exiting")
        return

    # 2. 扫描任务
    print("\n[Step 2] Scanning evaluation results...")
    tasks, mapping = scan_evaluation_results(input_path, audio_path)

    user_count = sum(1 for t in tasks if t.is_user)
    model_count = sum(1 for t in tasks if not t.is_user)
    print(f"Found {len(tasks)} unique audio files to annotate:")
    print(f"  - User audio (shared): {user_count}")
    print(f"  - Model output audio:  {model_count}")

    if limit > 0:
        tasks = tasks[:limit]
        print(f"Limited to {len(tasks)} tasks")

    # 3. 多线程标注
    print("\n[Step 3] Running EMO annotation...")

    results_cache: Dict[str, Dict] = {}

    start_time = time.time()
    completed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_task, task, annotator): task
            for task in tasks
        }

        for future in as_completed(futures):
            task = futures[future]
            try:
                audio_path, annotation = future.result()
                results_cache[audio_path] = annotation
                completed += 1

                if completed % 20 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    eta = avg_time * (len(tasks) - completed)
                    print(f"  Progress: {completed}/{len(tasks)} | "
                          f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

            except Exception as e:
                print(f"[Error] Task failed: {task.audio_path} - {e}")

    # 4. 组织结果
    print("\n[Step 4] Organizing results...")
    file_emo_results: Dict[Path, Dict[int, Dict]] = {}

    for audio_path, pair_mappings in mapping.items():
        annotation = results_cache.get(audio_path, {"emotion": "error"})

        for pm in pair_mappings:
            file_key = pm.result_file
            if file_key not in file_emo_results:
                file_emo_results[file_key] = {}

            if pm.pair_index not in file_emo_results[file_key]:
                file_emo_results[file_key][pm.pair_index] = {}

            key = "input" if pm.is_input else "output"
            file_emo_results[file_key][pm.pair_index][key] = annotation

    # 5. 更新文件
    print("\n[Step 5] Updating result files...")
    updated = 0
    for result_file, emo_data in file_emo_results.items():
        if output_path != input_path:
            import shutil
            rel_path = result_file.relative_to(input_path)
            dest_file = output_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(result_file, dest_file)
            result_file = dest_file

        if update_result_file(result_file, emo_data):
            updated += 1

    # 6. 统计
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EMO Annotation Complete!")
    print("=" * 60)
    print(f"Total tasks:     {len(tasks)}")
    print(f"Completed:       {completed}")
    print(f"Files updated:   {updated}")
    print(f"Total time:      {elapsed:.1f}s")
    print(f"Avg time/audio:  {elapsed/max(completed,1):.2f}s")


# ================= 入口 =================

def main():
    parser = argparse.ArgumentParser(description="EMO Annotation Pipeline")

    parser.add_argument("--input", "-i", required=True, help="Input evaluation results directory")
    parser.add_argument("--audio", "-a", required=True, help="Audio files directory")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument(
        "--model",
        default="/home/u2023112559/qix/Models/Models/emotion2vec_plus_large",
        help="emotion2vec model path"
    )
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()

    run_emo_annotation(
        input_dir=args.input,
        audio_dir=args.audio,
        output_dir=args.output,
        model_path=args.model,
        device=args.device,
        num_workers=args.workers,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
