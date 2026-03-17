"""
特征提取器基类
定义特征提取的统一接口
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""

    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        """
        初始化特征提取器

        Args:
            model_name: 模型名称或路径
            device: 运行设备
            **kwargs: 其他参数
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.kwargs = kwargs

    @abstractmethod
    def load_model(self) -> None:
        """加载模型"""
        pass

    @abstractmethod
    def extract(self, audio_path: str) -> Dict[str, Any]:
        """
        提取特征

        Args:
            audio_path: 音频文件路径

        Returns:
            特征字典
        """
        pass

    def extract_batch(self, audio_paths: list) -> list:
        """
        批量提取特征

        Args:
            audio_paths: 音频文件路径列表

        Returns:
            特征字典列表
        """
        results = []
        for path in audio_paths:
            try:
                result = self.extract(path)
                results.append(result)
            except Exception as e:
                print(f"Error extracting features from {path}: {e}")
                results.append(None)
        return results

    @property
    @abstractmethod
    def feature_names(self) -> list:
        """返回特征名称列表"""
        pass

    def to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        将特征字典转换为向量

        Args:
            features: 特征字典

        Returns:
            特征向量
        """
        raise NotImplementedError("Subclass should implement this method")


class FeatureExtractor:
    """特征提取管理器"""

    def __init__(self):
        self.extractors = {}

    def register(self, name: str, extractor: BaseFeatureExtractor) -> None:
        """
        注册特征提取器

        Args:
            name: 提取器名称
            extractor: 提取器实例
        """
        self.extractors[name] = extractor

    def get(self, name: str) -> Optional[BaseFeatureExtractor]:
        """获取特征提取器"""
        return self.extractors.get(name)

    def extract_all(self, audio_path: str) -> Dict[str, Dict]:
        """
        使用所有注册的提取器提取特征

        Args:
            audio_path: 音频文件路径

        Returns:
            所有特征的字典
        """
        all_features = {}
        for name, extractor in self.extractors.items():
            try:
                features = extractor.extract(audio_path)
                all_features[name] = features
            except Exception as e:
                print(f"Error with extractor {name}: {e}")
                all_features[name] = None
        return all_features
