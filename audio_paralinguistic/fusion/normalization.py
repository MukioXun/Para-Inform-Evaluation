"""
特征标准化模块
对特征进行标准化处理
"""
from typing import Dict, Optional, List
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Normalizer:
    """特征标准化器"""

    def __init__(
        self,
        method: str = "z-score",
        feature_ranges: Optional[Dict[str, tuple]] = None
    ):
        """
        初始化标准化器

        Args:
            method: 标准化方法 ("z-score", "min-max", "none")
            feature_ranges: 各特征的范围 {feature_name: (min, max)}
        """
        self.method = method
        self.feature_ranges = feature_ranges or {}
        self.scalers: Dict[str, object] = {}

    def fit(self, feature_matrix: np.ndarray) -> "Normalizer":
        """
        拟合标准化参数

        Args:
            feature_matrix: 特征矩阵 [N_samples, N_features]

        Returns:
            self
        """
        if self.method == "z-score":
            self.scalers["global"] = StandardScaler()
            self.scalers["global"].fit(feature_matrix)
        elif self.method == "min-max":
            self.scalers["global"] = MinMaxScaler()
            self.scalers["global"].fit(feature_matrix)

        return self

    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        转换特征

        Args:
            feature_matrix: 特征矩阵

        Returns:
            标准化后的特征矩阵
        """
        if self.method == "none":
            return feature_matrix

        if "global" not in self.scalers:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        return self.scalers["global"].transform(feature_matrix)

    def fit_transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        拟合并转换

        Args:
            feature_matrix: 特征矩阵

        Returns:
            标准化后的特征矩阵
        """
        self.fit(feature_matrix)
        return self.transform(feature_matrix)

    def inverse_transform(self, normalized_matrix: np.ndarray) -> np.ndarray:
        """
        反向转换

        Args:
            normalized_matrix: 标准化后的矩阵

        Returns:
            原始尺度的矩阵
        """
        if self.method == "none":
            return normalized_matrix

        return self.scalers["global"].inverse_transform(normalized_matrix)

    def normalize_feature(
        self,
        feature_name: str,
        values: np.ndarray
    ) -> np.ndarray:
        """
        对单个特征进行标准化

        Args:
            feature_name: 特征名称
            values: 特征值

        Returns:
            标准化后的值
        """
        if feature_name in self.feature_ranges:
            min_val, max_val = self.feature_ranges[feature_name]
            return (values - min_val) / (max_val - min_val)

        return values

    def get_statistics(self) -> Dict:
        """
        获取标准化统计信息

        Returns:
            统计信息字典
        """
        stats = {}

        if "global" in self.scalers:
            scaler = self.scalers["global"]
            if isinstance(scaler, StandardScaler):
                stats["mean"] = scaler.mean_.tolist()
                stats["std"] = scaler.scale_.tolist()
            elif isinstance(scaler, MinMaxScaler):
                stats["min"] = scaler.data_min_.tolist()
                stats["max"] = scaler.data_max_.tolist()

        return stats

    def save(self, path: str) -> None:
        """保存标准化参数"""
        import json
        params = {
            "method": self.method,
            "statistics": self.get_statistics()
        }
        with open(path, 'w') as f:
            json.dump(params, f, indent=2)

    def load(self, path: str) -> None:
        """加载标准化参数"""
        import json
        with open(path, 'r') as f:
            params = json.load(f)

        self.method = params["method"]
        # TODO: 根据统计信息重建scaler
