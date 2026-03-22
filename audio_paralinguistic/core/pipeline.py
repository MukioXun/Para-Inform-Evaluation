"""
Pipeline主流程控制
协调各任务标注器执行

输出结构：
- Top Level: audio_id, file_path, content_metadata, acoustic_features
- acoustic_features:
  - low_level: 基础物理特征
  - high_level: 任务标签 (ER/SED/SAR)

执行顺序：按标注维度依次加载模型，全部数据标注完该维度再切换
"""
import os
import json
import time
import hashlib
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
        save_individual: bool = False
    ) -> Dict[str, Any]:
        """
        单音频标注 - 返回三层嵌套结构

        Args:
            audio_path: 音频文件路径
            tasks: 指定任务列表，None表示所有任务
            save_individual: 是否保存各任务单独结果

        Returns:
            三层嵌套的标注结果
        """
        tasks = tasks or list(self.annotators.keys())
        audio_id = Path(audio_path).stem

        print(f"\n[Pipeline] Processing: {audio_id}")
        print(f"[Pipeline] Tasks: {tasks}")

        results = {}
        start_time = time.time()

        # 按层级顺序处理
        # 1. Low-level features
        if "LowLevel" in tasks:
            results["LowLevel"] = self._run_task("LowLevel", audio_path)

        # 2. High-level tasks (ER, SED, SAR, SCR, SpER)
        high_level_tasks = ["ER", "SED", "SAR", "SCR", "SpER"]
        for task in high_level_tasks:
            if task in tasks:
                results[task] = self._run_task(task, audio_path)

        total_time = time.time() - start_time

        # 构建三层嵌套结构
        merged = self._build_nested_structure(audio_path, results, total_time)

        # 保存合并结果
        merged_dir = self.output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_path = merged_dir / f"{audio_id}_merged.json"

        with open(merged_path, 'w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        print(f"[Pipeline] Merged result saved to: {merged_path}")

        return merged

    def _run_task(self, task_name: str, audio_path: str) -> Dict[str, Any]:
        """运行单个任务"""
        if task_name not in self.annotators:
            print(f"[Pipeline] Warning: Task {task_name} not registered, skipping")
            return {"error": "Task not registered"}

        try:
            result = self.annotators[task_name].process(audio_path)
            return result
        except Exception as e:
            print(f"[Pipeline]   {task_name}: failed - {e}")
            return {"error": str(e)}

    def _build_nested_structure(
        self,
        audio_path: str,
        results: Dict[str, Any],
        total_time: float
    ) -> Dict[str, Any]:
        """
        构建三层嵌套结构
        删除 task, status, timestamp 等冗余字段
        """
        audio_id = Path(audio_path).stem

        # === Top Level ===
        structure = {
            "audio_id": audio_id,
            "file_path": str(audio_path),
            "content_metadata": self._extract_content_metadata(audio_path, results),
            "acoustic_features": {
                "low_level": {},
                "high_level": {}
            }
        }

        # === acoustic_features.low_level ===
        if "LowLevel" in results and "predictions" in results["LowLevel"]:
            low_level_data = results["LowLevel"]["predictions"]
            structure["acoustic_features"]["low_level"] = {
                "spectral": low_level_data.get("spectral", {}),
                "prosody": low_level_data.get("prosody", {}),
                "energy": low_level_data.get("energy", {}),
                "temporal": low_level_data.get("temporal", {}),
                "timbre": low_level_data.get("timbre", {})
            }

        # === acoustic_features.high_level ===
        high_level = {}

        # ER: 情感识别
        if "ER" in results and "predictions" in results["ER"]:
            er_data = results["ER"]["predictions"]
            high_level["emotion"] = {
                "emotion_id": er_data.get("discrete", {}).get("emotion_id", 8),
                "primary_emotion": er_data.get("discrete", {}).get("primary_emotion", "unknown"),
                "confidence": er_data.get("discrete", {}).get("confidence", 0.0),
                "distribution": er_data.get("discrete", {}).get("emotion_distribution", {}),
                "valence": er_data.get("dimensional", {}).get("valence", 0.0),
                "arousal": er_data.get("dimensional", {}).get("arousal", 0.0)
            }

        # SED: 声学事件检测
        if "SED" in results and "predictions" in results["SED"]:
            sed_data = results["SED"]["predictions"]
            high_level["events"] = {
                "top_events": sed_data.get("top_events", []),
                "prob_summary": sed_data.get("prob_summary", {}),
                "primary_event": sed_data.get("primary_event", "unknown")
            }

        # SAR: 说话人属性
        if "SAR" in results and "predictions" in results["SAR"]:
            sar_data = results["SAR"]["predictions"]
            high_level["speaker"] = {
                "gender": sar_data.get("attributes", {}).get("gender", {}),
                "age": sar_data.get("attributes", {}).get("age", {}),
                "tone": sar_data.get("attributes", {}).get("tone", {})
            }

        # SCR: 语音内容
        if "SCR" in results and "predictions" in results["SCR"]:
            scr_data = results["SCR"]["predictions"]
            high_level["transcription"] = {
                "text": scr_data.get("text", ""),
                "language": scr_data.get("language", "unknown")
            }

        # SpER: 语音实体
        if "SpER" in results and "predictions" in results["SpER"]:
            sper_data = results["SpER"]["predictions"]
            high_level["entities"] = sper_data.get("entities", [])

        structure["acoustic_features"]["high_level"] = high_level

        return structure

    def _extract_content_metadata(self, audio_path: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """提取内容元数据"""
        import soundfile as sf

        metadata = {
            "duration_seconds": 0.0,
            "sample_rate": 0,
            "channels": 1,
            "format": Path(audio_path).suffix.lstrip('.')
        }

        try:
            info = sf.info(audio_path)
            metadata["duration_seconds"] = round(info.duration, 3)
            metadata["sample_rate"] = info.samplerate
            metadata["channels"] = info.channels
        except Exception:
            pass

        # 从LowLevel获取更精确的时长
        if "LowLevel" in results:
            temporal = results["LowLevel"].get("predictions", {}).get("temporal", {})
            if temporal.get("duration", {}).get("total_seconds"):
                metadata["duration_seconds"] = temporal["duration"]["total_seconds"]

        return metadata

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
    from ..annotators.er.hubert_emotion import Emotion2VecAnnotator
    from ..annotators.sar.sar_annotator import SARAnnotator
    from ..annotators.lowlevel.feature_extractor import LowLevelFeatureExtractor

    # 任务到标注器类的映射
    TASK_ANNOTATORS = {
        "LowLevel": LowLevelFeatureExtractor,
        "SCR": WhisperASRAnnotator,
        "SpER": FunASRNERAnnotator,
        "SED": PANNsDetector,
        "ER": Emotion2VecAnnotator,
        "SAR": SARAnnotator,
    }

    # 创建Pipeline
    pipeline = AnnotationPipeline({
        'output_dir': output_dir or 'data/annotations'
    })

    # 注册标注器
    tasks = tasks or list(TASK_ANNOTATORS.keys())

    for task in tasks:
        if task in TASK_ANNOTATORS:
            config = MODEL_CONFIGS.get(task, {}).copy()
            config['device'] = device
            annotator = TASK_ANNOTATORS[task](config)
            pipeline.register_annotator(task, annotator)
        else:
            print(f"[Warning] Unknown task: {task}")

    return pipeline
