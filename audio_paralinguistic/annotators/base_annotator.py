"""
标注器基类
所有任务标注器的父类，定义统一接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time


class BaseAnnotator(ABC):
    """标注器基类"""

    TASK_NAME: str = "base"

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.device = config.get('device', 'cuda')
        self.sample_rate = config.get('sample_rate', 16000)

    @abstractmethod
    def load_model(self):
        """加载模型"""
        pass

    @abstractmethod
    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """执行标注，返回 predictions 和 logits"""
        pass

    def process(self, audio_path: str) -> Dict[str, Any]:
        """
        完整处理流程，输出标准格式
        """
        start_time = time.time()

        try:
            # 执行标注
            result = self.annotate(audio_path)
            inference_time = time.time() - start_time

            # 构建标准输出
            output = {
                "audio_id": Path(audio_path).stem,
                "task": self.TASK_NAME,
                "predictions": result.get("predictions", {}),
                "logits": result.get("logits", {}),
                "metadata": {
                    "model": self.config.get("model_name", "unknown"),
                    "inference_time": round(inference_time, 3),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "success"
                }
            }

        except Exception as e:
            inference_time = time.time() - start_time
            output = {
                "audio_id": Path(audio_path).stem,
                "task": self.TASK_NAME,
                "predictions": {},
                "logits": {},
                "metadata": {
                    "model": self.config.get("model_name", "unknown"),
                    "inference_time": round(inference_time, 3),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "status": "failed",
                    "error": str(e)
                }
            }

        return output

    def save(self, output: Dict[str, Any], save_path: str):
        """保存结果到JSON文件"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None
