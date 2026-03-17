"""
报告生成模块
生成评估报告
"""
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import json


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: str = "./data/output"):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate(
        self,
        schema: Any,
        clustering_result: Any,
        evaluation_result: Any,
        metrics: Optional[Dict] = None,
        save_format: str = "json"
    ) -> str:
        """
        生成完整报告

        Args:
            schema: 标注体系
            clustering_result: 聚类结果
            evaluation_result: 评估结果
            metrics: 指标字典
            save_format: 保存格式 ("json", "markdown")

        Returns:
            报告文件路径
        """
        report_data = {
            "report_time": datetime.now().isoformat(),
            "schema": self._serialize_schema(schema),
            "clustering": self._serialize_clustering(clustering_result),
            "evaluation": self._serialize_evaluation(evaluation_result),
            "metrics": metrics or {}
        }

        if save_format == "json":
            return self._save_json(report_data)
        elif save_format == "markdown":
            return self._save_markdown(report_data)
        else:
            raise ValueError(f"Unknown format: {save_format}")

    def _serialize_schema(self, schema) -> Dict:
        """序列化标注体系"""
        if hasattr(schema, '__dict__'):
            return {
                "schema_version": getattr(schema, 'schema_version', '1.0'),
                "categories": [
                    {
                        "id": cat.category_id,
                        "name": cat.category_name,
                        "count": cat.sample_count,
                        "description": cat.description
                    }
                    for cat in getattr(schema, 'categories', [])
                ]
            }
        return schema if isinstance(schema, dict) else {}

    def _serialize_clustering(self, result) -> Dict:
        """序列化聚类结果"""
        if hasattr(result, '__dict__'):
            return {
                "n_clusters": result.n_clusters,
                "cluster_sizes": result.cluster_sizes,
                "metadata": result.metadata
            }
        return result if isinstance(result, dict) else {}

    def _serialize_evaluation(self, result) -> Dict:
        """序列化评估结果"""
        if hasattr(result, '__dict__'):
            return {
                "silhouette_score": result.silhouette_score,
                "davies_bouldin_score": result.davies_bouldin_score,
                "calinski_harabasz_score": result.calinski_harabasz_score,
                "n_clusters": result.n_clusters,
                "noise_ratio": result.noise_ratio
            }
        return result if isinstance(result, dict) else {}

    def _save_json(self, data: Dict) -> str:
        """保存为JSON"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def _save_markdown(self, data: Dict) -> str:
        """保存为Markdown"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(self.output_dir, filename)

        lines = [
            "# 语音副语言感知评估报告",
            f"\n生成时间: {data['report_time']}",
            "\n---\n",
            "## 1. 聚类结果概览\n",
            f"- 聚类数量: {data['clustering'].get('n_clusters', 'N/A')}",
            f"- 噪声比例: {data['evaluation'].get('noise_ratio', 0):.2%}",
            "\n## 2. 聚类质量评估\n",
            f"- 轮廓系数: {data['evaluation'].get('silhouette_score', 0):.4f}",
            f"- DB指数: {data['evaluation'].get('davies_bouldin_score', 0):.4f}",
            f"- CH指数: {data['evaluation'].get('calinski_harabasz_score', 0):.4f}",
            "\n## 3. 标注体系\n",
        ]

        for cat in data['schema'].get('categories', []):
            lines.append(f"### {cat.get('name', 'Unknown')}")
            lines.append(f"- 样本数量: {cat.get('count', 0)}")
            lines.append(f"- 描述: {cat.get('description', 'N/A')}\n")

        lines.append("\n---\n")
        lines.append("*报告由 Audio Paralinguistic Analyzer 自动生成*")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return filepath

    def generate_visualization_report(
        self,
        embedding: Any,
        labels: Any,
        output_path: Optional[str] = None
    ) -> str:
        """
        生成可视化报告

        Args:
            embedding: 降维后的嵌入
            labels: 聚类标签
            output_path: 输出路径

        Returns:
            图片文件路径
        """
        import matplotlib.pyplot as plt

        if output_path is None:
            output_path = os.path.join(
                self.output_dir,
                f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )

        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                color = 'gray'

            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                label=f'Cluster {label}' if label != -1 else 'Noise',
                alpha=0.6
            )

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('Clustering Visualization')
        ax.legend()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path
