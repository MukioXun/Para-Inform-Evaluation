"""
Pipeline主流程控制
协调各任务标注器执行
"""
import os
import json
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..annotators.base_annotator import BaseAnnotator


class AnnotationPipeline:
    """多任务标注流水线"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.annotators: Dict[str, BaseAnnotator] = {}
        self.output_dir = Path(self.config.get('output_dir', 'data/annotations'))

    def register_annotator(self, task_name: str, annotator: BaseAnnotator):
        """注册标注器"""
        self.annotators[task_name] = annotator
        print(f"[Pipeline] Registered annotator: {task_name}")

    def load_all_models(self):
        """加载所有已注册的模型"""
        print("\n[Pipeline] Loading all models...")
        for task_name, annotator in self.annotators.items():
            print(f"[Pipeline] Loading {task_name}...")
            try:
                annotator.load_model()
            except Exception as e:
                print(f"[Pipeline] Failed to load {task_name}: {e}")

    def annotate_single(
        self,
        audio_path: str,
        tasks: Optional[List[str]] = None,
        save_individual: bool = True
    ) -> Dict[str, Any]:
        """
        单音频标注

        Args:
            audio_path: 音频文件路径
            tasks: 指定任务列表，None表示所有任务
            save_individual: 是否保存各任务单独结果

        Returns:
            合并后的标注结果
        """
        tasks = tasks or list(self.annotators.keys())
        audio_id = Path(audio_path).stem

        print(f"\n[Pipeline] Processing: {audio_id}")
        print(f"[Pipeline] Tasks: {tasks}")

        results = {}
        start_time = time.time()

        for task in tasks:
            if task not in self.annotators:
                print(f"[Pipeline] Warning: Task {task} not registered, skipping")
                continue

            try:
                result = self.annotators[task].process(audio_path)
                results[task] = result

                # 保存单独结果
                if save_individual:
                    task_output_dir = self.output_dir / task.lower()
                    task_output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = task_output_dir / f"{audio_id}.json"
                    self.annotators[task].save(result, str(output_path))
                    print(f"[Pipeline]   {task}: saved to {output_path}")

            except Exception as e:
                print(f"[Pipeline]   {task}: failed - {e}")
                results[task] = {
                    "audio_id": audio_id,
                    "task": task,
                    "error": str(e),
                    "metadata": {"status": "failed"}
                }

        total_time = time.time() - start_time

        # 合并结果
        merged = self._merge_results(audio_path, results, total_time)

        return merged

    def annotate_batch(
        self,
        audio_dir: str,
        output_dir: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        num_workers: int = 1
    ):
        """
        批量标注

        Args:
            audio_dir: 音频目录
            output_dir: 输出目录
            tasks: 指定任务
            num_workers: 并行worker数（暂不支持多进程）
        """
        if output_dir:
            self.output_dir = Path(output_dir)

        # 获取音频文件
        audio_dir = Path(audio_dir)
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
            audio_files.extend(audio_dir.glob(ext))

        print(f"\n[Pipeline] Found {len(audio_files)} audio files")
        print(f"[Pipeline] Output directory: {self.output_dir}")

        # 处理每个文件
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}] {audio_file.name}")
            self.annotate_single(str(audio_file), tasks)

        print(f"\n[Pipeline] Batch processing complete!")

    def _merge_results(
        self,
        audio_path: str,
        results: Dict[str, Any],
        total_time: float
    ) -> Dict[str, Any]:
        """合并各任务结果"""
        audio_id = Path(audio_path).stem

        merged = {
            "audio_id": audio_id,
            "file_path": str(audio_path),
            "annotations": {},
            "processing_info": {
                "total_time": round(total_time, 3),
                "tasks_completed": [],
                "tasks_failed": [],
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
        }

        for task, result in results.items():
            if result.get("metadata", {}).get("status") == "failed":
                merged["processing_info"]["tasks_failed"].append(task)
            else:
                merged["processing_info"]["tasks_completed"].append(task)

            merged["annotations"][task] = result

        # 保存合并结果
        merged_dir = self.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_path = merged_dir / f"{audio_id}_merged.json"

        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"[Pipeline] Merged result saved to: {merged_path}")

        return merged


def create_pipeline(
    tasks: List[str] = None,
    output_dir: str = None,
    device: str = "cuda"
) -> AnnotationPipeline:
    """
    创建并初始化Pipeline

    Args:
        tasks: 要加载的任务列表
        output_dir: 输出目录
        device: 设备

    Returns:
        初始化好的Pipeline
    """
    from ..config.model_config import MODEL_CONFIGS

    # 动态导入标注器
    from ..annotators.scr.whisper_asr import WhisperASRAnnotator
    from ..annotators.sper.funasr_ner import FunASRNERAnnotator
    from ..annotators.sed.panns_detector import PANNsDetector
    from ..annotators.er.hubert_emotion import HuBERTEmotionAnnotator
    from ..annotators.sar.ecapa_attribute import ECAPAAttributeAnnotator

    # 任务到标注器类的映射
    TASK_ANNOTATORS = {
        "SCR": WhisperASRAnnotator,
        "SpER": FunASRNERAnnotator,
        "SED": PANNsDetector,
        "ER": HuBERTEmotionAnnotator,
        "SAR": ECAPAAttributeAnnotator,
    }

    # 创建Pipeline
    pipeline = AnnotationPipeline({
        'output_dir': output_dir or 'data/annotations'
    })

    # 注册标注器
    tasks = tasks or list(TASK_ANNOTATORS.keys())

    for task in tasks:
        if task in TASK_ANNOTATORS and task in MODEL_CONFIGS:
            config = MODEL_CONFIGS[task].copy()
            config['device'] = device
            annotator = TASK_ANNOTATORS[task](config)
            pipeline.register_annotator(task, annotator)
        else:
            print(f"[Warning] Unknown task: {task}")

    return pipeline
