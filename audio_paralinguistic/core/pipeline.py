"""
主流程控制模块
协调各模块完成完整的处理流程
"""
import os
import json
import argparse
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import threading

from .audio_processor import AudioProcessor, AudioSegment
from .feature_extractor import FeatureExtractor
from ..config import ModelConfig, FeatureConfig


class Pipeline:
    """主流程控制器"""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        num_workers: int = 4
    ):
        """
        初始化Pipeline

        Args:
            model_config: 模型配置
            feature_config: 特征配置
            num_workers: 并行工作线程数
        """
        self.model_config = model_config or ModelConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.num_workers = num_workers

        # 初始化各模块
        self.audio_processor = AudioProcessor(
            sample_rate=self.model_config.sample_rate,
            max_segment_length=self.model_config.max_audio_length
        )
        self.feature_extractor = FeatureExtractor()

        # 注册标注器（在子类或运行时注册）
        self.annotators = {}

        # 写锁（用于并行写入）
        self.write_lock = threading.Lock()

    def register_annotator(self, name: str, annotator) -> None:
        """注册标注器"""
        self.annotators[name] = annotator

    def process_single(
        self,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        处理单个音频文件

        Args:
            audio_path: 音频文件路径
            output_path: 输出路径（可选）

        Returns:
            处理结果字典
        """
        # 1. 音频预处理
        segments = self.audio_processor.process_file(audio_path)

        results = {
            "audio_path": audio_path,
            "segments": []
        }

        # 2. 对每个片段进行标注
        for segment in segments:
            segment_result = {
                "segment_id": segment.audio_id,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration,
                "annotations": {}
            }

            # 使用所有注册的标注器进行标注
            for name, annotator in self.annotators.items():
                try:
                    if segment.segment_path:
                        annotation = annotator.annotate(segment.segment_path)
                    else:
                        annotation = annotator.annotate(audio_path)
                    segment_result["annotations"][name] = annotation
                except Exception as e:
                    print(f"Error with annotator {name} for segment {segment.audio_id}: {e}")
                    segment_result["annotations"][name] = None

            results["segments"].append(segment_result)

        # 3. 保存结果
        if output_path:
            self._save_result(results, output_path)

        return results

    def process_batch(
        self,
        input_path: str,
        output_path: str,
        continue_from: bool = True
    ) -> List[Dict]:
        """
        批量处理音频文件

        Args:
            input_path: 输入路径（JSONL文件或目录）
            output_path: 输出路径（JSONL文件）
            continue_from: 是否从上次中断处继续

        Returns:
            处理结果列表
        """
        # 读取输入
        if input_path.endswith('.jsonl'):
            audio_paths = self._read_jsonl_input(input_path)
        else:
            audio_paths = self._scan_audio_directory(input_path)

        # 检查已处理的项目
        processed_ids = set()
        if continue_from and os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        processed_ids.add(item.get("audio_path", ""))
                    except:
                        pass

        # 过滤已处理项
        audio_paths = [p for p in audio_paths if p not in processed_ids]

        # 并行处理
        results = []
        with open(output_path, 'a', encoding='utf-8') as fout:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self.process_single, path): path
                    for path in audio_paths
                }

                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        result = future.result()
                        results.append(result)

                        # 写入结果
                        with self.write_lock:
                            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                            fout.flush()
                    except Exception as e:
                        print(f"Error processing {futures[future]}: {e}")

        return results

    def _read_jsonl_input(self, jsonl_path: str) -> List[str]:
        """读取JSONL格式的输入文件"""
        paths = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'path' in item:
                        paths.append(item['path'])
                    elif 'audio_path' in item:
                        paths.append(item['audio_path'])
                except:
                    pass
        return paths

    def _scan_audio_directory(self, dir_path: str) -> List[str]:
        """扫描目录获取音频文件列表"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        paths = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    paths.append(os.path.join(root, file))

        return sorted(paths)

    def _save_result(self, result: Dict, output_path: str) -> None:
        """保存结果到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


def create_pipeline(config_path: Optional[str] = None) -> Pipeline:
    """
    创建Pipeline实例

    Args:
        config_path: 配置文件路径

    Returns:
        Pipeline实例
    """
    # TODO: 从配置文件加载配置
    return Pipeline()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="Audio Paralinguistic Analysis Pipeline")
    parser.add_argument("--mode", type=str, choices=["annotate", "cluster", "pipeline"],
                        default="pipeline", help="运行模式")
    parser.add_argument("--input", type=str, required=True, help="输入路径")
    parser.add_argument("--output", type=str, required=True, help="输出路径")
    parser.add_argument("--num_workers", type=int, default=4, help="并行线程数")
    parser.add_argument("--annotators", type=str, nargs='+',
                        help="使用的标注器列表")

    args = parser.parse_args()

    pipeline = create_pipeline()
    pipeline.process_batch(args.input, args.output)


if __name__ == "__main__":
    main()
