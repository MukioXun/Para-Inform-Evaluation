"""
聚类评估模块
计算聚类效果的各项指标
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """评估结果数据类"""
    silhouette_score: float       # 轮廓系数
    davies_bouldin_score: float   # DB指数（越小越好）
    calinski_harabasz_score: float # CH指数（越大越好）
    n_clusters: int               # 聚类数量
    noise_ratio: float            # 噪声点比例
    metadata: Dict                # 其他信息


class ClusterEvaluator:
    """聚类评估器"""

    def __init__(self):
        self.scores = {}

    def evaluate(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> EvaluationResult:
        """
        评估聚类效果

        Args:
            X: 特征矩阵
            labels: 聚类标签

        Returns:
            EvaluationResult对象
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        # 过滤噪声点
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        n_clusters = len(set(labels_valid))
        n_noise = np.sum(labels == -1)
        noise_ratio = n_noise / len(labels)

        # 如果有效聚类太少，返回默认值
        if n_clusters < 2:
            return EvaluationResult(
                silhouette_score=-1.0,
                davies_bouldin_score=float('inf'),
                calinski_harabasz_score=0.0,
                n_clusters=n_clusters,
                noise_ratio=noise_ratio,
                metadata={"error": "Less than 2 valid clusters"}
            )

        # 计算各项指标
        sil_score = silhouette_score(X_valid, labels_valid)
        db_score = davies_bouldin_score(X_valid, labels_valid)
        ch_score = calinski_harabasz_score(X_valid, labels_valid)

        return EvaluationResult(
            silhouette_score=sil_score,
            davies_bouldin_score=db_score,
            calinski_harabasz_score=ch_score,
            n_clusters=n_clusters,
            noise_ratio=noise_ratio,
            metadata={}
        )

    def compare_clusterings(
        self,
        X: np.ndarray,
        results: Dict[str, "ClusterResult"]
    ) -> Dict[str, EvaluationResult]:
        """
        比较不同聚类结果

        Args:
            X: 特征矩阵
            results: 各方法的聚类结果

        Returns:
            各方法的评估结果
        """
        evaluations = {}

        for method_name, cluster_result in results.items():
            if cluster_result is None:
                continue

            eval_result = self.evaluate(X, cluster_result.labels)
            evaluations[method_name] = eval_result

        return evaluations

    def find_optimal_k(
        self,
        X: np.ndarray,
        k_range: Tuple[int, int] = (2, 10),
        method: str = "silhouette"
    ) -> Tuple[int, Dict[int, float]]:
        """
        使用Elbow/Silhouette方法找到最优K

        Args:
            X: 特征矩阵
            k_range: K的范围
            method: 评估方法 ("silhouette", "elbow")

        Returns:
            (最优K, 各K对应的分数)
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        scores = {}
        best_k = k_range[0]
        best_score = -float('inf')

        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)

            if method == "silhouette":
                score = silhouette_score(X, labels)
            elif method == "elbow":
                score = -kmeans.inertia_  # 负的inertia，便于统一寻找最大值
            else:
                raise ValueError(f"Unknown method: {method}")

            scores[k] = score

            if score > best_score:
                best_score = score
                best_k = k

        return best_k, scores

    def get_cluster_statistics(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        获取各聚类的统计信息

        Args:
            X: 特征矩阵
            labels: 聚类标签

        Returns:
            各聚类的统计信息
        """
        unique_labels = set(labels)
        stats = {}

        for label in unique_labels:
            if label == -1:
                continue

            mask = labels == label
            cluster_data = X[mask]

            stats[int(label)] = {
                "size": int(np.sum(mask)),
                "center": np.mean(cluster_data, axis=0).tolist(),
                "std": np.std(cluster_data, axis=0).tolist(),
                "min": np.min(cluster_data, axis=0).tolist(),
                "max": np.max(cluster_data, axis=0).tolist(),
            }

        return stats

    def summary(self, evaluations: Dict[str, EvaluationResult]) -> str:
        """
        生成评估摘要报告

        Args:
            evaluations: 各方法的评估结果

        Returns:
            摘要文本
        """
        lines = ["=" * 60]
        lines.append("聚类评估报告")
        lines.append("=" * 60)

        for method, eval_result in evaluations.items():
            lines.append(f"\n{method}:")
            lines.append(f"  聚类数量: {eval_result.n_clusters}")
            lines.append(f"  轮廓系数: {eval_result.silhouette_score:.4f}")
            lines.append(f"  DB指数:   {eval_result.davies_bouldin_score:.4f}")
            lines.append(f"  CH指数:   {eval_result.calinski_harabasz_score:.4f}")
            lines.append(f"  噪声比例: {eval_result.noise_ratio:.2%}")

        lines.append("\n" + "=" * 60)

        # 推荐最佳方法
        best_method = max(
            evaluations.items(),
            key=lambda x: x[1].silhouette_score
        )[0]
        lines.append(f"推荐方法: {best_method}")

        return "\n".join(lines)
