#!/usr/bin/env python3
"""
对话质量评估可视化脚本
生成各模型在不同维度上的表现图表

使用方法:
    python visualize_results.py --input evaluation_summary.json --output ./plots
    python visualize_results.py  # 使用默认路径
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 颜色配置
COLORS = {
    'glm4': '#4CAF50',
    'gpt-4o-voice-mode': '#2196F3',
    'llamaomni2': '#FF9800',
    'original': '#9C27B0',
    'qwen2.5': '#F44336',
    'rl_real_all': '#00BCD4'
}

CATEGORY_COLORS = {
    'age': '#3498db',
    'emotion': '#e74c3c',
    'gender': '#2ecc71',
    'sarcasm': '#f39c12'
}


def load_summary(file_path: str) -> Dict[str, Any]:
    """加载评估汇总数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_model_ranking(summary: Dict, output_dir: Path):
    """
    图1: 模型总体排名柱状图
    """
    model_stats = summary.get('model_statistics', {})

    if not model_stats:
        print("No model statistics found")
        return

    models = list(model_stats.keys())
    avg_scores = [model_stats[m]['avg_score'] for m in models]

    # 排序
    sorted_data = sorted(zip(models, avg_scores), key=lambda x: x[1], reverse=True)
    models, avg_scores = zip(*sorted_data)

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [COLORS.get(m, '#666666') for m in models]
    bars = ax.barh(models, avg_scores, color=colors, edgecolor='white', linewidth=1.5)

    # 添加数值标签
    for bar, score in zip(bars, avg_scores):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Average Score', fontsize=12)
    ax.set_title('Model Ranking by Average Score', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(avg_scores) * 1.15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'model_ranking.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_performance(summary: Dict, output_dir: Path):
    """
    图2: 各分类下模型表现分组柱状图
    """
    model_stats = summary.get('model_statistics', {})
    categories = ['age', 'emotion', 'gender', 'sarcasm']

    models = list(model_stats.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(categories))
    width = 0.12
    multiplier = 0

    for model in models:
        category_avg = model_stats[model].get('category_avg', {})
        scores = [category_avg.get(cat, 0) for cat in categories]

        offset = width * multiplier
        color = COLORS.get(model, '#666666')
        rects = ax.bar(x + offset, scores, width, label=model, color=color, alpha=0.85)
        multiplier += 1

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Model Performance by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([cat.upper() for cat in categories], fontsize=11)
    ax.legend(loc='upper right', ncols=2, fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'category_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_distribution(summary: Dict, output_dir: Path):
    """
    图3: 各模型分数分布堆叠柱状图
    """
    model_stats = summary.get('model_statistics', {})

    models = list(model_stats.keys())
    scores = ['1', '2', '3', '4', '5']

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(models))
    width = 0.6

    bottom = np.zeros(len(models))
    colors_dist = ['#d32f2f', '#f57c00', '#fbc02d', '#4caf50', '#1976d2']

    for i, score in enumerate(scores):
        counts = [model_stats[m]['score_distribution'].get(score, 0) for m in models]
        ax.bar(x, counts, width, label=f'Score {score}', bottom=bottom, color=colors_dist[i], alpha=0.85)
        bottom += counts

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Score Distribution by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.legend(title='Score', loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'score_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_radar_chart(summary: Dict, output_dir: Path):
    """
    图4: 模型能力雷达图
    """
    model_stats = summary.get('model_statistics', {})
    categories = ['age', 'emotion', 'gender', 'sarcasm']

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    for model, stats in model_stats.items():
        category_avg = stats.get('category_avg', {})
        values = [category_avg.get(cat, 0) for cat in categories]
        values += values[:1]  # 闭合

        color = COLORS.get(model, '#666666')
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([cat.upper() for cat in categories], fontsize=12)
    ax.set_ylim(0, 5)
    ax.set_title('Model Capability Radar', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'radar_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_heatmap(summary: Dict, output_dir: Path):
    """
    图5: 模型-分类得分热力图
    """
    model_stats = summary.get('model_statistics', {})
    categories = ['age', 'emotion', 'gender', 'sarcasm']

    models = list(model_stats.keys())

    # 构建矩阵
    data = np.zeros((len(models), len(categories)))
    for i, model in enumerate(models):
        category_avg = model_stats[model].get('category_avg', {})
        for j, cat in enumerate(categories):
            data[i, j] = category_avg.get(cat, 0)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels([cat.upper() for cat in categories], fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    # 添加数值标签
    for i in range(len(models)):
        for j in range(len(categories)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=11, fontweight='bold')

    ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score', fontsize=12)

    plt.tight_layout()
    output_path = output_dir / 'heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_category_boxplot(summary: Dict, output_dir: Path):
    """
    图6: 各分类得分箱线图
    """
    category_stats = summary.get('category_statistics', {})

    if not category_stats:
        return

    categories = list(category_stats.keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    # 构建数据
    data = []
    labels = []
    colors = []

    for cat in categories:
        stats = category_stats[cat]
        dist = stats.get('score_distribution', {})

        # 从分布重建分数列表
        scores = []
        for score, count in dist.items():
            scores.extend([int(score)] * count)

        if scores:
            data.append(scores)
            labels.append(cat.upper())
            colors.append(CATEGORY_COLORS.get(cat, '#666666'))

    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Distribution by Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 6)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'category_boxplot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_html_report(summary: Dict, output_dir: Path):
    """
    生成HTML报告
    """
    model_stats = summary.get('model_statistics', {})
    category_stats = summary.get('category_statistics', {})
    meta = summary.get('meta', {})

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .meta {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #4CAF50; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .chart img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .highlight {{ color: #4CAF50; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🎵 Audio Dialogue Quality Evaluation Report</h1>

    <div class="meta">
        <h2>📊 Overview</h2>
        <p><strong>Timestamp:</strong> {meta.get('timestamp', 'N/A')}</p>
        <p><strong>Total Evaluations:</strong> {meta.get('total_evaluations', 0)}</p>
        <p><strong>Total Models:</strong> {meta.get('total_models', 0)}</p>
        <p><strong>Overall Average Score:</strong> <span class="highlight">{meta.get('overall_avg_score', 0):.3f}</span></p>
    </div>

    <h2>📈 Model Ranking</h2>
    <div class="chart">
        <img src="model_ranking.png" alt="Model Ranking">
    </div>

    <h2>📊 Category Performance</h2>
    <div class="chart">
        <img src="category_performance.png" alt="Category Performance">
    </div>

    <h2>🎯 Model Capability Radar</h2>
    <div class="chart">
        <img src="radar_chart.png" alt="Radar Chart">
    </div>

    <h2>🌡️ Performance Heatmap</h2>
    <div class="chart">
        <img src="heatmap.png" alt="Heatmap">
    </div>

    <h2>📉 Score Distribution</h2>
    <div class="chart">
        <img src="score_distribution.png" alt="Score Distribution">
    </div>

    <h2>📋 Detailed Statistics</h2>
    <div class="stats">
"""

    # 模型统计卡片
    for model, stats in sorted(model_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True):
        html += f"""
        <div class="card">
            <h3>{model}</h3>
            <p><strong>Average Score:</strong> <span class="highlight">{stats['avg_score']:.3f}</span></p>
            <p><strong>Total Evaluations:</strong> {stats['total_count']}</p>
            <table>
                <tr><th>Category</th><th>Avg Score</th></tr>
"""
        for cat, score in stats.get('category_avg', {}).items():
            html += f"                <tr><td>{cat.upper()}</td><td>{score:.3f}</td></tr>\n"

        html += """            </table>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    output_path = output_dir / 'report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation Visualization")
    parser.add_argument(
        '--input', '-i',
        default='/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/Results/qwen_eval_results/evaluation_summary.json',
        help='Input evaluation summary JSON'
    )
    parser.add_argument(
        '--output', '-o',
        default='/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/Results/qwen_eval_results/plots',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Evaluation Visualization")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_dir}")
    print()

    # 加载数据
    summary = load_summary(input_path)

    # 生成图表
    print("Generating plots...")
    plot_model_ranking(summary, output_dir)
    plot_category_performance(summary, output_dir)
    plot_score_distribution(summary, output_dir)
    plot_radar_chart(summary, output_dir)
    plot_heatmap(summary, output_dir)
    plot_category_boxplot(summary, output_dir)

    # 生成HTML报告
    print("\nGenerating HTML report...")
    generate_html_report(summary, output_dir)

    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"Open report: {output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
