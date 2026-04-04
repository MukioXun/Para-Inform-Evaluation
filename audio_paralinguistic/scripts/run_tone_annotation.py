#!/usr/bin/env python3
"""
TONE 标注补充脚本 - 多线程版本
在 Audio-Reasoner 环境中运行: conda activate Audio-Reasoner

功能:
1. 读取已有的评测结果 (不含 TONE)
2. 使用 Audio-Reasoner 多线程标注 TONE
3. 更新结果文件

使用方法:
    conda activate Audio-Reasoner
    python run_tone_annotation.py --input ./evaluation_results --output ./evaluation_results_with_tone

    # 指定线程数
    python run_tone_annotation.py --input ./evaluation_results --workers 4

    # 限制处理数量 (测试用)
    python run_tone_annotation.py --input ./evaluation_results --limit 10
"""
import argparse
import json
import os
import sys
import re
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


# ================= Tone 标注器 =================

class ToneAnnotator:
    """Tone 标注器 - 使用 Audio-Reasoner (Qwen2-Audio)"""

    TONE_PROMPT = """Analyze the speaker's vocal delivery in this audio while adhering to the following strict constraints:
    Zero Semantic Content: Treat the audio as if it were in a language you do not understand. Do not transcribe or summarize the verbal content.
    Focus on Prosody: Analyze only the non-verbal cues and acoustic features.

    Dimensions for Analysis:
    Affective Base: The underlying emotional mood (e.g., solemn, lighthearted, agitated).
    Speech Rate: Tempo, rhythmic regularity, and use of pauses.
    Inflection & Pitch: Pitch range, contours (rising/falling), and melodic variation.
    Vocal Intensity: Breath support, projection, and dynamic range (strong vs. frail).
    Psychological Profile: Perceived confidence, uncertainty, urgency, or composure.

    Output: Provide a precise and professional summary of these vocal traits."""

    SYSTEM_PROMPT = 'You are an audio deep-thinking model. Upon receiving a question, please respond in two parts: <THINK> and <RESPONSE>. The <THINK> section should be further divided into four parts: <PLANNING>, <CAPTION>, <REASONING>, and <SUMMARY>.'

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.engine = None
        self._lock = threading.Lock()
        self._loaded = False

    def load_model(self) -> bool:
        """加载 Audio-Reasoner 模型"""
        try:
            from swift.llm import PtEngine

            print(f"[Tone] Loading Audio-Reasoner from: {self.model_path}")

            self.engine = PtEngine(
                self.model_path,
                max_batch_size=1,
                model_type='qwen2_audio'
            )

            self._loaded = True
            print(f"[Tone] Audio-Reasoner loaded successfully")
            return True

        except ImportError as e:
            print(f"[Tone] Error: swift module not found!")
            print(f"[Tone] Please run in Audio-Reasoner environment:")
            print(f"[Tone]   conda activate Audio-Reasoner")
            return False
        except Exception as e:
            print(f"[Tone] Failed to load model: {e}")
            return False

    def is_loaded(self) -> bool:
        return self._loaded and self.engine is not None

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行语气标注"""
        if not self.is_loaded():
            return {"description": "unavailable", "error": "model not loaded"}

        from swift.llm import InferRequest, RequestConfig
        from swift.plugin import InferStats

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": self.TONE_PROMPT}
                ]
            }
        ]

        request_config = RequestConfig(
            max_tokens=512,
            temperature=0,
            stream=False
        )

        try:
            with self._lock:
                metric = InferStats()
                results = self.engine.infer(
                    [InferRequest(messages=messages)],
                    request_config,
                    metrics=[metric]
                )

            if results and len(results) > 0 and results[0] is not None:
                full_response = results[0].choices[0].message.content
                description = self._extract_caption(full_response)
            else:
                description = "unknown"

            return {"description": description}

        except Exception as e:
            print(f"[Tone] Inference failed for {audio_path}: {e}")
            return {"description": "error", "error": str(e)}

    def _extract_caption(self, response: str) -> str:
        """提取 CAPTION 内容"""
        # 尝试提取 <CAPTION> 标签
        caption_match = re.search(r'<CAPTION>(.*?)</CAPTION>', response, re.DOTALL)
        if caption_match:
            return caption_match.group(1).strip()

        # 尝试提取 <RESPONSE> 标签
        response_match = re.search(r'<RESPONSE>(.*?)</RESPONSE>', response, re.DOTALL)
        if response_match:
            return response_match.group(1).strip()

        # 清理并返回原始响应
        cleaned = re.sub(r'<THINK>.*?</THINK>', '', response, flags=re.DOTALL)
        cleaned = re.sub(r'<[^>]+>', '', cleaned)
        return cleaned.strip() if cleaned.strip() else response[:200]


# ================= 数据处理函数 =================

@dataclass
class AudioTask:
    """单个音频标注任务"""
    audio_path: str
    is_user: bool  # True=user音频, False=模型输出音频


@dataclass
class PairMapping:
    """配对映射关系"""
    result_file: Path
    pair_index: int
    is_input: bool  # True=输入音频, False=输出音频


def scan_evaluation_results(input_dir: Path, audio_base_dir: Path) -> Tuple[List[AudioTask], Dict[str, List[PairMapping]]]:
    """
    扫描评测结果，提取需要标注 TONE 的音频任务（去重）

    Args:
        input_dir: 评测结果目录
        audio_base_dir: 音频文件基础目录

    Returns:
        (tasks, mapping)
        - tasks: 去重后的音频任务列表
        - mapping: audio_path -> PairMapping 列表 的映射
    """
    tasks = []
    audio_seen = set()  # 去重用
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

            # 获取目录路径
            dir_name = data.get("dir_name", result_file.stem)
            category = data.get("category", category_dir.name)
            audio_dir = audio_base_dir / category / dir_name

            # 处理每个配对
            pairs = data.get("pairs", [])
            for pair_idx, pair in enumerate(pairs):
                # 输入音频 (user.wav)
                input_info = pair.get("input", {})
                input_file = input_info.get("file", "")
                if input_file:
                    input_path = audio_dir / input_file
                    input_path_str = str(input_path)

                    if input_path.exists():
                        # 添加映射关系
                        if input_path_str not in mapping:
                            mapping[input_path_str] = []
                        mapping[input_path_str].append(PairMapping(
                            result_file=result_file,
                            pair_index=pair_idx,
                            is_input=True
                        ))

                        # 去重添加任务
                        if input_path_str not in audio_seen:
                            audio_seen.add(input_path_str)
                            tasks.append(AudioTask(
                                audio_path=input_path_str,
                                is_user=True
                            ))

                # 输出音频 (模型生成)
                output_info = pair.get("output", {})
                output_file = output_info.get("file", "")
                if output_file:
                    output_path = audio_dir / output_file
                    output_path_str = str(output_path)

                    if output_path.exists():
                        # 添加映射关系
                        if output_path_str not in mapping:
                            mapping[output_path_str] = []
                        mapping[output_path_str].append(PairMapping(
                            result_file=result_file,
                            pair_index=pair_idx,
                            is_input=False
                        ))

                        # 去重添加任务（模型输出通常不重复，但还是检查）
                        if output_path_str not in audio_seen:
                            audio_seen.add(output_path_str)
                            tasks.append(AudioTask(
                                audio_path=output_path_str,
                                is_user=False
                            ))

    return tasks, mapping


def process_task(
    task: AudioTask,
    annotator: ToneAnnotator
) -> Tuple[str, Dict]:
    """
    处理单个任务

    Args:
        task: 音频任务
        annotator: Tone 标注器

    Returns:
        (audio_path, annotation)
    """
    annotation = annotator.annotate(task.audio_path)
    return task.audio_path, annotation


def update_result_file(
    result_file: Path,
    tone_results: Dict[int, Dict[str, Dict]]
) -> bool:
    """
    更新结果文件，添加 TONE 标注

    Args:
        result_file: 结果文件路径
        tone_results: {pair_index: {"input": {...}, "output": {...}}}

    Returns:
        是否成功
    """
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pairs = data.get("pairs", [])
        for pair_idx, tone_data in tone_results.items():
            if pair_idx >= len(pairs):
                continue

            pair = pairs[pair_idx]

            # 更新 input
            if "input" in tone_data:
                input_tone = tone_data["input"]
                pair.setdefault("input", {}).setdefault("annotation", {})["TONE"] = input_tone

            # 更新 output
            if "output" in tone_data:
                output_tone = tone_data["output"]
                pair.setdefault("output", {}).setdefault("annotation", {})["TONE"] = output_tone

        # 原子写入
        tmp_path = result_file.with_suffix('.tmp')
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(result_file)

        return True

    except Exception as e:
        print(f"[Error] Failed to update {result_file}: {e}")
        return False


# ================= 主流程 =================

def run_tone_annotation(
    input_dir: str,
    audio_dir: str,
    output_dir: Optional[str],
    model_path: str,
    device: str,
    num_workers: int,
    limit: int
):
    """
    运行 TONE 标注

    Args:
        input_dir: 输入评测结果目录
        audio_dir: 音频文件目录
        output_dir: 输出目录 (None 表示原地更新)
        model_path: Audio-Reasoner 模型路径
        device: 计算设备
        num_workers: 并行线程数
        limit: 限制处理数量 (0=不限制)
    """
    input_path = Path(input_dir)
    audio_path = Path(audio_dir)

    if not input_path.exists():
        print(f"[Error] Input directory not found: {input_path}")
        return

    if not audio_path.exists():
        print(f"[Error] Audio directory not found: {audio_path}")
        return

    # 输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = input_path

    print("=" * 60)
    print("TONE Annotation Pipeline (Multi-threaded)")
    print("=" * 60)
    print(f"Input:    {input_path}")
    print(f"Audio:    {audio_path}")
    print(f"Output:   {output_path}")
    print(f"Model:    {model_path}")
    print(f"Device:   {device}")
    print(f"Workers:  {num_workers}")
    print(f"Limit:    {limit if limit > 0 else 'unlimited'}")

    # 1. 加载模型
    print("\n[Step 1] Loading Audio-Reasoner...")
    annotator = ToneAnnotator(model_path, device)
    if not annotator.load_model():
        print("[Error] Failed to load model, exiting")
        return

    # 2. 扫描任务
    print("\n[Step 2] Scanning evaluation results...")
    tasks, mapping = scan_evaluation_results(input_path, audio_path)

    # 统计
    user_count = sum(1 for t in tasks if t.is_user)
    model_count = sum(1 for t in tasks if not t.is_user)
    print(f"Found {len(tasks)} unique audio files to annotate:")
    print(f"  - User audio (shared): {user_count}")
    print(f"  - Model output audio:  {model_count}")

    if limit > 0:
        tasks = tasks[:limit]
        print(f"Limited to {len(tasks)} tasks")

    # 3. 多线程标注
    print("\n[Step 3] Running TONE annotation...")

    results_cache: Dict[str, Dict] = {}  # audio_path -> annotation

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

                # 进度显示
                if completed % 10 == 0 or completed == len(tasks):
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    eta = avg_time * (len(tasks) - completed)
                    print(f"  Progress: {completed}/{len(tasks)} | "
                          f"Elapsed: {elapsed:.1f}s | ETA: {eta:.1f}s")

            except Exception as e:
                print(f"[Error] Task failed: {task.audio_path} - {e}")

    # 4. 根据映射关系组织结果
    print("\n[Step 4] Organizing results...")
    file_tone_results: Dict[Path, Dict[int, Dict]] = {}  # result_file -> {pair_idx -> tone_data}

    for audio_path, pair_mappings in mapping.items():
        annotation = results_cache.get(audio_path, {"description": "error"})

        for pm in pair_mappings:
            file_key = pm.result_file
            if file_key not in file_tone_results:
                file_tone_results[file_key] = {}

            if pm.pair_index not in file_tone_results[file_key]:
                file_tone_results[file_key][pm.pair_index] = {}

            key = "input" if pm.is_input else "output"
            file_tone_results[file_key][pm.pair_index][key] = annotation

    # 5. 更新文件
    print("\n[Step 5] Updating result files...")
    updated = 0
    for result_file, tone_data in file_tone_results.items():
        # 如果输出目录不同，复制文件
        if output_path != input_path:
            import shutil
            rel_path = result_file.relative_to(input_path)
            dest_file = output_path / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(result_file, dest_file)
            result_file = dest_file

        if update_result_file(result_file, tone_data):
            updated += 1

    # 5. 统计
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TONE Annotation Complete!")
    print("=" * 60)
    print(f"Total tasks:     {len(tasks)}")
    print(f"Completed:       {completed}")
    print(f"Files updated:   {updated}")
    print(f"Total time:      {elapsed:.1f}s")
    print(f"Avg time/audio:  {elapsed/max(completed,1):.2f}s")


# ================= 入口 =================

def main():
    parser = argparse.ArgumentParser(
        description="TONE Annotation Pipeline (Multi-threaded)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 基本用法
    conda activate Audio-Reasoner
    python run_tone_annotation.py --input ./evaluation_results --audio /path/to/audio

    # 指定输出目录
    python run_tone_annotation.py --input ./evaluation_results --audio /path/to/audio --output ./results_with_tone

    # 多线程
    python run_tone_annotation.py --input ./evaluation_results --audio /path/to/audio --workers 4

    # 测试模式
    python run_tone_annotation.py --input ./evaluation_results --audio /path/to/audio --limit 5
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input evaluation results directory"
    )
    parser.add_argument(
        "--audio", "-a",
        required=True,
        help="Audio files directory"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: update in place)"
    )
    parser.add_argument(
        "--model",
        default="/home/u2023112559/qix/Models/Models/Audio-Reasoner",
        help="Audio-Reasoner model path"
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
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks (0=unlimited)"
    )

    args = parser.parse_args()

    run_tone_annotation(
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
