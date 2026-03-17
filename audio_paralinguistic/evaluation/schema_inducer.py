"""
标注体系推导模块
从聚类结果反向推导标注体系
"""
from typing import Dict, Any, List, Optional
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CategoryDefinition:
    """类别定义"""
    category_id: int
    category_name: str
    feature_characteristics: Dict[str, Any]
    sample_count: int
    representative_samples: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class AnnotationSchema:
    """标注体系"""
    schema_version: str
    categories: List[CategoryDefinition]
    feature_importance: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class SchemaInducer:
    """标注体系推导器"""

    def __init__(self, use_llm: bool = False, llm_client=None):
        """
        初始化推导器

        Args:
            use_llm: 是否使用LLM辅助命名
            llm_client: LLM客户端实例
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

    def induce(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        sample_ids: Optional[List[str]] = None
    ) -> AnnotationSchema:
        """
        从聚类结果推导标注体系

        Args:
            X: 特征矩阵
            labels: 聚类标签
            feature_names: 特征名称列表
            sample_ids: 样本ID列表

        Returns:
            AnnotationSchema对象
        """
        categories = []
        unique_labels = sorted(set(labels) - {-1})  # 排除噪声点

        # 计算全局特征重要性
        feature_importance = self._compute_feature_importance(X, labels, feature_names)

        for label in unique_labels:
            mask = labels == label
            cluster_data = X[mask]
            cluster_ids = [sample_ids[i] for i in range(len(labels)) if labels[i] == label] if sample_ids else []

            # 提取特征特征
            characteristics = self._extract_characteristics(
                cluster_data, feature_names
            )

            # 生成类别名称
            category_name = self._generate_category_name(characteristics, label)

            # 选择代表性样本
            representative = cluster_ids[:5] if cluster_ids else []

            category = CategoryDefinition(
                category_id=int(label),
                category_name=category_name,
                feature_characteristics=characteristics,
                sample_count=int(np.sum(mask)),
                representative_samples=representative,
                description=self._generate_description(characteristics, category_name)
            )
            categories.append(category)

        return AnnotationSchema(
            schema_version="1.0",
            categories=categories,
            feature_importance=feature_importance,
            metadata={
                "n_samples": len(labels),
                "n_features": X.shape[1],
                "noise_count": int(np.sum(labels == -1))
            }
        )

    def _compute_feature_importance(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        计算特征重要性（基于类间方差）

        Args:
            X: 特征矩阵
            labels: 聚类标签
            feature_names: 特征名称

        Returns:
            特征重要性字典
        """
        from sklearn.preprocessing import normalize

        importance = {}
        n_features = X.shape[1]

        # 计算每个特征在不同类别间的方差
        global_mean = np.mean(X, axis=0)
        cluster_means = {}

        for label in set(labels):
            if label == -1:
                continue
            mask = labels == label
            cluster_means[label] = np.mean(X[mask], axis=0)

        # 计算类间方差
        between_class_var = np.zeros(n_features)
        for label, mean in cluster_means.items():
            between_class_var += (mean - global_mean) ** 2

        # 归一化
        if np.sum(between_class_var) > 0:
            between_class_var = between_class_var / np.sum(between_class_var)

        for i, name in enumerate(feature_names[:n_features]):
            importance[name] = float(between_class_var[i])

        return importance

    def _extract_characteristics(
        self,
        cluster_data: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """
        提取聚类特征

        Args:
            cluster_data: 聚类内的数据
            feature_names: 特征名称

        Returns:
            特征统计字典
        """
        characteristics = {}
        n_features = min(cluster_data.shape[1], len(feature_names))

        for i in range(n_features):
            name = feature_names[i]
            values = cluster_data[:, i]
            characteristics[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }

        return characteristics

    def _generate_category_name(
        self,
        characteristics: Dict[str, Any],
        label: int
    ) -> str:
        """
        生成类别名称

        Args:
            characteristics: 特征统计
            label: 聚类标签

        Returns:
            类别名称
        """
        if self.use_llm and self.llm_client:
            return self._llm_generate_name(characteristics)

        # 基于特征的简单命名
        # 找出最显著的特征
        significant_features = []
        for name, stats in characteristics.items():
            if stats["std"] < 0.3:  # 方差较小的特征更稳定
                significant_features.append((name, stats["mean"]))

        if significant_features:
            # 按均值排序
            significant_features.sort(key=lambda x: abs(x[1]), reverse=True)
            top_feature = significant_features[0][0]
            return f"Cluster_{label}_{top_feature}"

        return f"Cluster_{label}"

    def _llm_generate_name(self, characteristics: Dict[str, Any]) -> str:
        """使用LLM生成类别名称"""
        # TODO: 实现LLM命名
        if self.llm_client:
            # prompt = f"根据以下特征统计生成一个简洁的类别名称：{characteristics}"
            # name = self.llm_client.generate(prompt)
            # return name
            pass
        return "LLM_Category"

    def _generate_description(
        self,
        characteristics: Dict[str, Any],
        category_name: str
    ) -> str:
        """
        生成类别描述

        Args:
            characteristics: 特征统计
            category_name: 类别名称

        Returns:
            描述文本
        """
        desc_parts = [f"类别 '{category_name}' 的特征描述："]

        for name, stats in list(characteristics.items())[:5]:  # 只取前5个特征
            desc_parts.append(
                f"- {name}: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}"
            )

        return "\n".join(desc_parts)

    def to_dict(self, schema: AnnotationSchema) -> Dict[str, Any]:
        """将标注体系转换为字典"""
        return {
            "schema_version": schema.schema_version,
            "categories": [
                {
                    "category_id": cat.category_id,
                    "category_name": cat.category_name,
                    "feature_characteristics": cat.feature_characteristics,
                    "sample_count": cat.sample_count,
                    "representative_samples": cat.representative_samples,
                    "description": cat.description
                }
                for cat in schema.categories
            ],
            "feature_importance": schema.feature_importance,
            "metadata": schema.metadata
        }

    def save(self, schema: AnnotationSchema, path: str) -> None:
        """保存标注体系"""
        import json
        data = self.to_dict(schema)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
