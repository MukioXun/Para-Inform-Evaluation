"""
可视化工具函数
"""
from typing import Optional, List, Dict, Any
import numpy as np


def plot_clusters(
    embedding: np.ndarray,
    labels: np.ndarray,
    title: str = "Clustering Visualization",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
    alpha: float = 0.6
) -> None:
    """
    绘制聚类结果可视化

    Args:
        embedding: 降维后的嵌入 [N, 2]
        labels: 聚类标签 [N]
        title: 图表标题
        save_path: 保存路径（None则显示）
        figsize: 图表大小
        alpha: 透明度
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'gray'
            label_name = 'Noise'
        else:
            label_name = f'Cluster {label}'

        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[color],
            label=label_name,
            alpha=alpha
        )

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.legend(loc='best')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_feature_distribution(
    data: np.ndarray,
    feature_names: List[str],
    labels: Optional[np.ndarray] = None,
    title: str = "Feature Distribution",
    save_path: Optional[str] = None,
    max_features: int = 10
) -> None:
    """
    绘制特征分布图

    Args:
        data: 特征矩阵 [N, M]
        feature_names: 特征名称
        labels: 聚类标签（可选，用于区分不同类别）
        title: 图表标题
        save_path: 保存路径
        max_features: 最大显示特征数
    """
    import matplotlib.pyplot as plt

    n_features = min(data.shape[1], len(feature_names), max_features)

    if labels is not None:
        # 按类别绘制
        unique_labels = sorted(set(labels) - {-1})
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()

        for i in range(n_features):
            ax = axes[i]
            feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"

            for label in unique_labels:
                mask = labels == label
                ax.hist(data[mask, i], bins=30, alpha=0.5, label=f'Cluster {label}')

            ax.set_xlabel(feature_name)
            ax.set_ylabel('Count')
            ax.legend()

        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

    else:
        # 整体分布
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for i in range(n_features):
            ax = axes[i]
            feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
            ax.hist(data[:, i], bins=30, alpha=0.7)
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Count')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(
    data: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Correlation",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 10)
) -> None:
    """
    绘制特征相关性矩阵

    Args:
        data: 特征矩阵
        feature_names: 特征名称
        title: 图表标题
        save_path: 保存路径
        figsize: 图表大小
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 计算相关矩阵
    corr_matrix = np.corrcoef(data.T)

    # 截取特征名称
    n_features = min(len(feature_names), corr_matrix.shape[0])
    names = feature_names[:n_features]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix[:n_features, :n_features],
        xticklabels=names,
        yticklabels=names,
        cmap='coolwarm',
        center=0,
        ax=ax
    )
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cluster_sizes(
    labels: np.ndarray,
    title: str = "Cluster Size Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    绘制聚类大小分布图

    Args:
        labels: 聚类标签
        title: 图表标题
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt

    unique_labels, counts = np.unique(labels, return_counts=True)

    # 排除噪声
    mask = unique_labels != -1
    unique_labels = unique_labels[mask]
    counts = counts[mask]

    # 按大小排序
    sorted_indices = np.argsort(counts)[::-1]
    unique_labels = unique_labels[sorted_indices]
    counts = counts[sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(unique_labels)), counts)
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Sample Count')
    ax.set_title(title)
    ax.set_xticks(range(len(unique_labels)))
    ax.set_xticklabels([f'C{l}' for l in unique_labels], rotation=45)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(count), ha='center', va='bottom')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
