"""
特征融合模块
合并多个标注器的输出特征
"""
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class MergedFeatures:
    """融合后的特征数据类"""
    audio_id: str
    feature_vector: np.ndarray
    feature_dict: Dict[str, Any]
    metadata: Dict[str, Any]


class FeatureMerger:
    """特征融合器"""

    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        include_features: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None
    ):
        """
        初始化特征融合器

        Args:
            feature_weights: 各维度的权重
            include_features: 只包含的特征
            exclude_features: 排除的特征
        """
        self.feature_weights = feature_weights or {}
        self.include_features = include_features
        self.exclude_features = exclude_features or []

    def merge(
        self,
        audio_id: str,
        annotations: Dict[str, Dict[str, Any]]
    ) -> MergedFeatures:
        """
        合并多个标注器的特征

        Args:
            audio_id: 音频ID
            annotations: 各标注器的输出 {annotator_name: {feature: value}}

        Returns:
            MergedFeatures对象
        """
        feature_dict = {}
        feature_vectors = []
        metadata = {
            "annotators_used": list(annotations.keys()),
            "feature_counts": {}
        }

        for annotator_name, annotator_output in annotations.items():
            if annotator_output is None:
                continue

            # 提取数值特征
            for feature_name, feature_value in annotator_output.items():
                if feature_name in self.exclude_features:
                    continue

                if self.include_features and feature_name not in self.include_features:
                    continue

                feature_key = f"{annotator_name}.{feature_name}"

                if isinstance(feature_value, (int, float)):
                    feature_dict[feature_key] = feature_value
                    feature_vectors.append(float(feature_value))

                elif isinstance(feature_value, np.ndarray):
                    feature_dict[feature_key] = feature_value.tolist()
                    feature_vectors.extend(feature_value.flatten().tolist())

                elif isinstance(feature_value, list):
                    feature_dict[feature_key] = feature_value
                    feature_vectors.extend([float(v) for v in feature_value if isinstance(v, (int, float))])

            metadata["feature_counts"][annotator_name] = len(annotator_output)

        # 应用权重
        if self.feature_weights:
            feature_vectors = self._apply_weights(feature_vectors, annotations)

        return MergedFeatures(
            audio_id=audio_id,
            feature_vector=np.array(feature_vectors),
            feature_dict=feature_dict,
            metadata=metadata
        )

    def _apply_weights(
        self,
        feature_vector: List[float],
        annotations: Dict[str, Dict]
    ) -> List[float]:
        """应用特征权重"""
        # TODO: 实现加权逻辑
        return feature_vector

    def merge_batch(
        self,
        annotations_list: List[Dict[str, Any]]
    ) -> List[MergedFeatures]:
        """
        批量合并特征

        Args:
            annotations_list: 标注结果列表

        Returns:
            融合后的特征列表
        """
        results = []
        for item in annotations_list:
            audio_id = item.get("audio_id", item.get("id", "unknown"))
            annotations = item.get("annotations", {})
            merged = self.merge(audio_id, annotations)
            results.append(merged)
        return results

    @staticmethod
    def to_feature_matrix(merged_features: List[MergedFeatures]) -> np.ndarray:
        """
        将融合后的特征列表转换为特征矩阵

        Args:
            merged_features: 融合后的特征列表

        Returns:
            特征矩阵 [N_samples, N_features]
        """
        vectors = [mf.feature_vector for mf in merged_features]

        # 处理不等长向量
        max_len = max(len(v) for v in vectors)
        padded_vectors = []
        for v in vectors:
            if len(v) < max_len:
                v = np.pad(v, (0, max_len - len(v)), mode='constant')
            padded_vectors.append(v)

        return np.array(padded_vectors)
