"""
标注器基类
定义所有标注器的统一接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time


class BaseAnnotator(ABC):
    """标注器基类"""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        max_retry: int = 3,
        retry_sleep: float = 0.5
    ):
        """
        初始化标注器

        Args:
            model_name: 模型名称或路径
            device: 运行设备
            max_retry: 最大重试次数
            retry_sleep: 重试间隔
        """
        self.model_name = model_name
        self.device = device
        self.max_retry = max_retry
        self.retry_sleep = retry_sleep
        self.model = None

    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """
        对音频进行标注

        Args:
            audio_path: 音频文件路径

        Returns:
            标注结果字典
        """
        pass

    def annotate_with_retry(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        带重试机制的标注

        Args:
            audio_path: 音频文件路径

        Returns:
            标注结果字典，失败返回None
        """
        for attempt in range(self.max_retry):
            try:
                return self.annotate(audio_path)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {audio_path}: {e}")
                if attempt < self.max_retry - 1:
                    time.sleep(self.retry_sleep)
        return None

    def annotate_batch(self, audio_paths: List[str]) -> List[Optional[Dict[str, Any]]]:
        """
        批量标注

        Args:
            audio_paths: 音频文件路径列表

        Returns:
            标注结果列表
        """
        results = []
        for path in audio_paths:
            result = self.annotate_with_retry(path)
            results.append(result)
        return results

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """返回此标注器能提取的特征名称"""
        pass

    @property
    def annotator_name(self) -> str:
        """返回标注器名称"""
        return self.__class__.__name__

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
