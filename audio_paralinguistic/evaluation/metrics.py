"""
评估指标计算模块
"""
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class MetricResult:
    """指标结果"""
    name: str
    value: float
    description: str
    details: Dict[str, Any]


class MetricsCalculator:
    """评估指标计算器"""

    def __init__(self):
        self.results = {}

    def calculate_all(
        self,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        计算所有可用指标

        Args:
            predictions: 预测结果
            ground_truth: 真实标签（可选）

        Returns:
            指标结果字典
        """
        self.results = {}

        # 无监督指标
        self.results["cluster_count"] = self._cluster_count(predictions)
        self.results["noise_ratio"] = self._noise_ratio(predictions)
        self.results["cluster_balance"] = self._cluster_balance(predictions)

        # 如果有真实标签
        if ground_truth is not None:
            self.results["adjusted_rand"] = self._adjusted_rand_index(predictions, ground_truth)
            self.results["normalized_mutual_info"] = self._normalized_mutual_info(predictions, ground_truth)
            self.results["homogeneity"] = self._homogeneity(predictions, ground_truth)

        return self.results

    def _cluster_count(self, labels: np.ndarray) -> MetricResult:
        """计算聚类数量"""
        n_clusters = len(set(labels) - {-1})
        return MetricResult(
            name="cluster_count",
            value=float(n_clusters),
            description="有效聚类数量",
            details={"unique_labels": list(set(labels))}
        )

    def _noise_ratio(self, labels: np.ndarray) -> MetricResult:
        """计算噪声比例"""
        noise_count = np.sum(labels == -1)
        ratio = noise_count / len(labels) if len(labels) > 0 else 0.0
        return MetricResult(
            name="noise_ratio",
            value=ratio,
            description="噪声点比例",
            details={"noise_count": int(noise_count), "total_count": len(labels)}
        )

    def _cluster_balance(self, labels: np.ndarray) -> MetricResult:
        """计算聚类平衡度"""
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            return MetricResult(
                name="cluster_balance",
                value=0.0,
                description="聚类大小分布的熵值",
                details={}
            )

        unique, counts = np.unique(valid_labels, return_counts=True)
        probs = counts / len(valid_labels)

        # 计算熵
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(unique))

        # 归一化熵（越接近1越平衡）
        balance = entropy / max_entropy if max_entropy > 0 else 0.0

        return MetricResult(
            name="cluster_balance",
            value=balance,
            description="聚类平衡度（归一化熵）",
            details={"cluster_sizes": counts.tolist()}
        )

    def _adjusted_rand_index(self, labels_pred: np.ndarray, labels_true: np.ndarray) -> MetricResult:
        """计算调整兰德指数"""
        from sklearn.metrics import adjusted_rand_score
        score = adjusted_rand_score(labels_true, labels_pred)
        return MetricResult(
            name="adjusted_rand",
            value=score,
            description="调整兰德指数 (ARI)",
            details={}
        )

    def _normalized_mutual_info(self, labels_pred: np.ndarray, labels_true: np.ndarray) -> MetricResult:
        """计算标准化互信息"""
        from sklearn.metrics import normalized_mutual_info_score
        score = normalized_mutual_info_score(labels_true, labels_pred)
        return MetricResult(
            name="normalized_mutual_info",
            value=score,
            description="标准化互信息 (NMI)",
            details={}
        )

    def _homogeneity(self, labels_pred: np.ndarray, labels_true: np.ndarray) -> MetricResult:
        """计算同质性"""
        from sklearn.metrics import homogeneity_score
        score = homogeneity_score(labels_true, labels_pred)
        return MetricResult(
            name="homogeneity",
            value=score,
            description="聚类同质性",
            details={}
        )

    def summary(self) -> str:
        """生成指标摘要"""
        lines = ["评估指标摘要", "-" * 40]
        for name, result in self.results.items():
            lines.append(f"{result.name}: {result.value:.4f} ({result.description})")
        return "\n".join(lines)
