"""
聚类引擎模块
支持多种聚类算法
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ClusterResult:
    """聚类结果数据类"""
    labels: np.ndarray           # 每个样本的聚类标签
    n_clusters: int              # 聚类数量
    cluster_centers: Optional[np.ndarray]  # 聚类中心
    cluster_sizes: Dict[int, int]          # 各聚类的大小
    metadata: Dict[str, Any]                # 其他元数据


class BaseClusterer(ABC):
    """聚类器基类"""

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseClusterer":
        """拟合聚类模型"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        pass

    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """拟合并预测"""
        pass


class KMeansClusterer(BaseClusterer):
    """K-Means聚类器"""

    def __init__(self, n_clusters: int = 8, **kwargs):
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.model = None

    def fit(self, X: np.ndarray) -> "KMeansClusterer":
        from sklearn.cluster import KMeans
        self.model = KMeans(n_clusters=self.n_clusters, **self.kwargs)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.model.labels_


class DBSCANClusterer(BaseClusterer):
    """DBSCAN聚类器"""

    def __init__(self, eps: float = 0.5, min_samples: int = 5, **kwargs):
        self.eps = eps
        self.min_samples = min_samples
        self.kwargs = kwargs
        self.model = None

    def fit(self, X: np.ndarray) -> "DBSCANClusterer":
        from sklearn.cluster import DBSCAN
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples, **self.kwargs)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # DBSCAN不支持predict，返回已拟合的标签
        return self.model.labels_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.model.labels_


class HDBSCANClusterer(BaseClusterer):
    """HDBSCAN聚类器"""

    def __init__(self, min_cluster_size: int = 5, **kwargs):
        self.min_cluster_size = min_cluster_size
        self.kwargs = kwargs
        self.model = None

    def fit(self, X: np.ndarray) -> "HDBSCANClusterer":
        import hdbscan
        self.model = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, **self.kwargs)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # HDBSCAN的predict需要额外处理
        return self.model.labels_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.model.labels_


class AgglomerativeClusterer(BaseClusterer):
    """层次聚类器"""

    def __init__(self, n_clusters: int = 8, linkage: str = "ward", **kwargs):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.kwargs = kwargs
        self.model = None

    def fit(self, X: np.ndarray) -> "AgglomerativeClusterer":
        from sklearn.cluster import AgglomerativeClustering
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            **self.kwargs
        )
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.labels_

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.model.labels_


class ClusterEngine:
    """聚类引擎"""

    CLUSTERER_MAP = {
        "kmeans": KMeansClusterer,
        "dbscan": DBSCANClusterer,
        "hdbscan": HDBSCANClusterer,
        "agglomerative": AgglomerativeClusterer,
    }

    def __init__(
        self,
        method: str = "hdbscan",
        **kwargs
    ):
        """
        初始化聚类引擎

        Args:
            method: 聚类方法
            **kwargs: 聚类参数
        """
        self.method = method
        self.kwargs = kwargs
        self.clusterer: Optional[BaseClusterer] = None

    def fit(self, X: np.ndarray) -> "ClusterEngine":
        """
        拟合聚类模型

        Args:
            X: 特征矩阵 [N_samples, N_features]

        Returns:
            self
        """
        if self.method not in self.CLUSTERER_MAP:
            raise ValueError(f"Unknown method: {self.method}. "
                            f"Available: {list(self.CLUSTERER_MAP.keys())}")

        self.clusterer = self.CLUSTERER_MAP[self.method](**self.kwargs)
        self.clusterer.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测聚类标签"""
        return self.clusterer.predict(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """拟合并预测"""
        self.fit(X)
        return self.clusterer.model.labels_

    def get_result(self) -> ClusterResult:
        """
        获取聚类结果

        Returns:
            ClusterResult对象
        """
        if self.clusterer is None:
            raise ValueError("Model not fitted. Call fit() first.")

        labels = self.clusterer.model.labels_

        # 计算聚类数量（排除噪声点 label=-1）
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        # 计算聚类中心
        cluster_centers = None
        if hasattr(self.clusterer.model, 'cluster_centers_'):
            cluster_centers = self.clusterer.model.cluster_centers_

        # 计算各聚类大小
        cluster_sizes = {}
        for label in unique_labels:
            cluster_sizes[int(label)] = int(np.sum(labels == label))

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            cluster_sizes=cluster_sizes,
            metadata={
                "method": self.method,
                "params": self.kwargs
            }
        )

    def compare_methods(
        self,
        X: np.ndarray,
        methods: List[str] = None
    ) -> Dict[str, ClusterResult]:
        """
        比较不同聚类方法的效果

        Args:
            X: 特征矩阵
            methods: 要比较的方法列表

        Returns:
            各方法的聚类结果
        """
        methods = methods or list(self.CLUSTERER_MAP.keys())
        results = {}

        for method in methods:
            try:
                engine = ClusterEngine(method=method, **self.kwargs)
                engine.fit(X)
                results[method] = engine.get_result()
            except Exception as e:
                print(f"Error with method {method}: {e}")
                results[method] = None

        return results
