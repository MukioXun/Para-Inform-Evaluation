"""
说话人识别标注器
使用 CAM++ / ECAPA-TDNN 进行说话人识别
"""
from typing import Dict, Any, List, Optional
import numpy as np
from ..base_annotator import BaseAnnotator


class SpeakerAnnotator(BaseAnnotator):
    """
    说话人识别标注器

    使用模型: CAM++ / ECAPA-TDNN (WeSpeaker)
    输出: 说话人embedding、身份聚类ID
    """

    def __init__(
        self,
        model_name: str = "cam++",
        device: str = "cuda",
        pretrained_model: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, device, **kwargs)
        self.pretrained_model = pretrained_model
        self._feature_names = [
            "speaker_embedding",   # 说话人特征向量
            "speaker_id",          # 说话人ID（聚类后）
            "speaker_confidence"   # 置信度
        ]

    def load_model(self) -> None:
        """
        加载模型

        依赖: wespeaker, torch
        """
        # TODO: 实现模型加载
        # import wespeaker
        # self.model = wespeaker.load_model(self.model_name)
        raise NotImplementedError(
            "Model loading not implemented. "
            "Please install wespeaker: pip install wespeaker\n"
            f"Model: {self.model_name}"
        )

    def annotate(self, audio_path: str) -> Dict[str, Any]:
        """
        执行说话人识别

        Args:
            audio_path: 音频文件路径

        Returns:
            {
                "speaker_embedding": [0.1, 0.2, ...],  # 192维或更高
                "speaker_id": None,  # 聚类后分配
                "speaker_confidence": 0.95
            }
        """
        if not self.is_loaded():
            self.load_model()

        # TODO: 实现说话人识别

        return {
            "speaker_embedding": None,
            "speaker_id": None,
            "speaker_confidence": 0.0
        }

    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        提取说话人embedding

        Args:
            audio_path: 音频文件路径

        Returns:
            embedding向量
        """
        result = self.annotate(audio_path)
        return result.get("speaker_embedding")

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        计算两个embedding的相似度（余弦相似度）

        Args:
            embedding1: 第一个embedding
            embedding2: 第二个embedding

        Returns:
            相似度分数 [-1, 1]
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names
