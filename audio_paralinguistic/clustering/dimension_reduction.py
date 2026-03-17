"""
降维模块
支持PCA、t-SNE、UMAP等降维方法
"""
from typing import Optional, Tuple, Dict
import numpy as np
from dataclasses import dataclass


@dataclass
class ReductionResult:
    """降维结果数据类"""
    embedding: np.ndarray     # 降维后的嵌入
    method: str               # 使用的方法
    n_components: int         # 降维后维度
    metadata: Dict            # 其他元数据


class DimensionReducer:
    """降维器"""

    def __init__(
        self,
        method: str = "umap",
        n_components: int = 2,
        **kwargs
    ):
        """
        初始化降维器

        Args:
            method: 降维方法 ("pca", "tsne", "umap")
            n_components: 降维后的维度
            **kwargs: 其他参数
        """
        self.method = method
        self.n_components = n_components
        self.kwargs = kwargs
        self.reducer = None

    def fit(self, X: np.ndarray) -> "DimensionReducer":
        """
        拟合降维模型

        Args:
            X: 特征矩阵 [N_samples, N_features]

        Returns:
            self
        """
        if self.method == "pca":
            from sklearn.decomposition import PCA
            self.reducer = PCA(n_components=self.n_components, **self.kwargs)

        elif self.method == "tsne":
            from sklearn.manifold import TSNE
            self.reducer = TSNE(n_components=self.n_components, **self.kwargs)

        elif self.method == "umap":
            import umap
            self.reducer = umap.UMAP(n_components=self.n_components, **self.kwargs)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.reducer.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        转换数据

        Args:
            X: 特征矩阵

        Returns:
            降维后的数据
        """
        if self.method == "tsne":
            # t-SNE不支持transform
            raise ValueError("t-SNE does not support transform. Use fit_transform instead.")

        return self.reducer.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        拟合并转换

        Args:
            X: 特征矩阵

        Returns:
            降维后的数据
        """
        self.fit(X)
        if self.method == "tsne":
            return self.reducer.embedding_
        return self.reducer.transform(X)

    def get_result(self, X: Optional[np.ndarray] = None) -> ReductionResult:
        """
        获取降维结果

        Args:
            X: 如果需要transform，提供数据

        Returns:
            ReductionResult对象
        """
        if self.reducer is None:
            raise ValueError("Reducer not fitted. Call fit() first.")

        if self.method == "tsne":
            embedding = self.reducer.embedding_
        elif X is not None:
            embedding = self.reducer.transform(X)
        else:
            embedding = self.reducer.embedding_

        metadata = {}
        if self.method == "pca":
            metadata["explained_variance_ratio"] = self.reducer.explained_variance_ratio_.tolist()
            metadata["total_variance_explained"] = sum(self.reducer.explained_variance_ratio_)

        return ReductionResult(
            embedding=embedding,
            method=self.method,
            n_components=self.n_components,
            metadata=metadata
        )

    def find_optimal_components(
        self,
        X: np.ndarray,
        variance_threshold: float = 0.95
    ) -> int:
        """
        使用PCA找到保留指定方差比例的最小维度

        Args:
            X: 特征矩阵
            variance_threshold: 方差阈值

        Returns:
            推荐的维度数
        """
        from sklearn.decomposition import PCA

        # 使用全部成分
        pca = PCA()
        pca.fit(X)

        # 找到满足阈值的最小维度
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        return n_components
