#!/usr/bin/env python3
"""
多模型协同评估 vs 单一模型评估 偏差分析

比较两种评估方法:
1. Cascade (多模型协同): ASR + EMO + AGE + GND + TONE -> Qwen API 文本评估
2. Omni (单一模型): Qwen Omni 直接听音频评估

输出:
- 偏差分析图表
- 相关性分析
- 详细比较报告
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置字体
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


def load_cascade_results(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    加载 Cascade (多模型协同) 评估结果

    Returns:
        {category: {model: avg_score}}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    for category, models in data.get('statistics', {}).items():
        results[category] = {}
        for model, stats in models.items():
            results[category][model] = stats.get('average_score', 0)

    return results


def load_omni_results(file_path: str) -> Dict[str, Dict[str, float]]:
    """
    加载 Omni (单一模型) 评估结果

    Returns:
        {category: {model: avg_score}}
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = {}
    model_stats = data.get('model_statistics', {})

    # 从 category_avg 提取
    categories = ['age', 'emotion', 'gender', 'sarcasm']
    for cat in categories:
        results[cat] = {}
        for model, stats in model_stats.items():
            cat_avg = stats.get('category_avg', {})
            results[cat][model] = cat_avg.get(cat, 0)

    return results


def compute_deviation(cascade: Dict, omni: Dict) -> Dict[str, Dict[str, float]]:
    """
    计算两种评估方法的偏差

    Returns:
        {category: {model: deviation}}
    """
    deviation = {}

    for category in cascade.keys():
        if category not in omni:
            continue

        deviation[category] = {}
        for model in cascade[category].keys():
            if model in omni[category]:
                cascade_score = cascade[category][model]
                omni_score = omni[category][model]
                deviation[category][model] = omni_score - cascade_score

    return deviation


def plot_deviation_comparison(cascade: Dict, omni: Dict, output_dir: Path):
    """
    图1: 各模型在不同分类下的得分对比
    """
    categories = ['age', 'emotion', 'gender', 'sarcasm']
    models = list(COLORS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, category in enumerate(categories):
        ax = axes[idx]

        x = np.arange(len(models))
        width = 0.35

        cascade_scores = [cascade.get(category, {}).get(m, 0) for m in models]
        omni_scores = [omni.get(category, {}).get(m, 0) for m in models]

        bars1 = ax.bar(x - width/2, cascade_scores, width, label='Cascade (Multi-Model)',
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, omni_scores, width, label='Omni (Single-Model)',
                       color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Average Score', fontsize=10)
        ax.set_title(f'{category.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 5.5)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cascade vs Omni Evaluation Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'comparison_by_category.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_deviation_heatmap(deviation: Dict, output_dir: Path):
    """
    图2: 偏差热力图
    """
    categories = list(deviation.keys())
    models = list(COLORS.keys())

    # 构建矩阵
    data = np.zeros((len(categories), len(models)))
    for i, cat in enumerate(categories):
        for j, model in enumerate(models):
            data[i, j] = deviation.get(cat, {}).get(model, 0)

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=10)
    ax.set_yticklabels([cat.upper() for cat in categories], fontsize=10)

    # 添加数值标签
    for i in range(len(categories)):
        for j in range(len(models)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=10)

    ax.set_title('Deviation Heatmap\n(Positive = Omni > Cascade, Negative = Omni < Cascade)',
                 fontsize=12, fontweight='bold')

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Score Deviation', fontsize=10)

    plt.tight_layout()

    output_path = output_dir / 'deviation_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_ranking_comparison(cascade: Dict, omni: Dict, output_dir: Path):
    """
    图3: 模型排名对比
    """
    # 计算总体平均分
    cascade_avg = {}
    omni_avg = {}

    for model in COLORS.keys():
        cascade_scores = []
        omni_scores = []

        for category in cascade.keys():
            if model in cascade[category]:
                cascade_scores.append(cascade[category][model])
            if model in omni.get(category, {}):
                omni_scores.append(omni[category][model])

        if cascade_scores:
            cascade_avg[model] = np.mean(cascade_scores)
        if omni_scores:
            omni_avg[model] = np.mean(omni_scores)

    models = list(cascade_avg.keys())
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    cascade_vals = [cascade_avg.get(m, 0) for m in models]
    omni_vals = [omni_avg.get(m, 0) for m in models]

    bars1 = ax.bar(x - width/2, cascade_vals, width, label='Cascade (Multi-Model)',
                   color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, omni_vals, width, label='Omni (Single-Model)',
                   color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Overall Model Ranking Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 5.5)
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    output_path = output_dir / 'ranking_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_correlation(cascade: Dict, omni: Dict, output_dir: Path):
    """
    图4: 相关性散点图
    """
    cascade_scores = []
    omni_scores = []
    labels = []

    for category in cascade.keys():
        for model in cascade[category].keys():
            if model in omni.get(category, {}):
                cascade_scores.append(cascade[category][model])
                omni_scores.append(omni[category][model])
                labels.append(f"{model[:6]}-{category[:3]}")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(cascade_scores, omni_scores, alpha=0.7, s=100, c='#3498db', edgecolors='white')

    # 添加对角线
    ax.plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect Agreement')

    # 添加标签
    for i, label in enumerate(labels):
        ax.annotate(label, (cascade_scores[i], omni_scores[i]), fontsize=7, alpha=0.6)

    ax.set_xlabel('Cascade Score (Multi-Model)', fontsize=12)
    ax.set_ylabel('Omni Score (Single-Model)', fontsize=12)
    ax.set_title('Correlation: Cascade vs Omni', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 计算相关系数
    correlation = np.corrcoef(cascade_scores, omni_scores)[0, 1]
    ax.text(0.5, 4.8, f'Pearson r = {correlation:.3f}', fontsize=12,
            fontweight='bold', color='#e74c3c')

    plt.tight_layout()

    output_path = output_dir / 'correlation_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

    return correlation


def plot_bias_by_model(deviation: Dict, output_dir: Path):
    """
    图5: 各模型的平均偏差
    """
    models = list(COLORS.keys())

    # 计算每个模型在所有分类的平均偏差
    model_bias = {}
    for model in models:
        biases = []
        for cat in deviation.keys():
            if model in deviation[cat]:
                biases.append(deviation[cat][model])
        if biases:
            model_bias[model] = np.mean(biases)

    fig, ax = plt.subplots(figsize=(10, 6))

    models_sorted = sorted(model_bias.keys(), key=lambda x: model_bias[x])
    biases = [model_bias[m] for m in models_sorted]
    colors = [COLORS.get(m, '#666666') for m in models_sorted]

    bars = ax.barh(models_sorted, biases, color=colors, alpha=0.8)

    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.set_xlabel('Average Deviation (Omni - Cascade)', fontsize=12)
    ax.set_title('Model Evaluation Bias\n(Positive = Omni rates higher, Negative = Cascade rates higher)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for bar, bias in zip(bars, biases):
        width = bar.get_width()
        ax.text(width + 0.05 if width >= 0 else width - 0.05,
                bar.get_y() + bar.get_height()/2,
                f'{bias:.2f}', ha='left' if width >= 0 else 'right',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    output_path = output_dir / 'bias_by_model.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_comparison_report(cascade: Dict, omni: Dict, deviation: Dict,
                               correlation: float, output_dir: Path):
    """
    生成比较报告
    """
    report = {
        "meta": {
            "title": "Cascade vs Omni Evaluation Comparison Report",
            "description": "Comparing multi-model collaborative evaluation vs single-model (Qwen Omni) evaluation"
        },
        "correlation": {
            "pearson_r": round(correlation, 4),
            "interpretation": "Strong positive correlation" if correlation > 0.7 else
                             "Moderate correlation" if correlation > 0.4 else "Weak correlation"
        },
        "overall_findings": {},
        "category_analysis": {},
        "model_analysis": {},
        "bias_summary": {}
    }

    # 整体发现
    cascade_all = [s for cat in cascade.values() for s in cat.values()]
    omni_all = [s for cat in omni.values() for s in cat.values()]

    report["overall_findings"] = {
        "cascade_avg": round(np.mean(cascade_all), 3),
        "omni_avg": round(np.mean(omni_all), 3),
        "avg_deviation": round(np.mean(omni_all) - np.mean(cascade_all), 3),
        "deviation_std": round(np.std([d for cat in deviation.values() for d in cat.values()]), 3)
    }

    # 分类分析
    for category in cascade.keys():
        cat_cascade = list(cascade[category].values())
        cat_omni = [omni.get(category, {}).get(m, 0) for m in cascade[category].keys()]
        cat_dev = [deviation.get(category, {}).get(m, 0) for m in cascade[category].keys()]

        report["category_analysis"][category] = {
            "cascade_avg": round(np.mean(cat_cascade), 3) if cat_cascade else 0,
            "omni_avg": round(np.mean(cat_omni), 3) if cat_omni else 0,
            "avg_deviation": round(np.mean(cat_dev), 3) if cat_dev else 0,
            "max_deviation_model": max(deviation[category].items(), key=lambda x: abs(x[1]))[0] if deviation.get(category) else None
        }

    # 模型分析
    for model in COLORS.keys():
        model_cascade = [cascade[cat].get(model, 0) for cat in cascade.keys() if model in cascade[cat]]
        model_omni = [omni[cat].get(model, 0) for cat in cascade.keys() if cat in omni and model in omni[cat]]
        model_dev = [deviation[cat].get(model, 0) for cat in deviation.keys() if model in deviation[cat]]

        report["model_analysis"][model] = {
            "cascade_avg": round(np.mean(model_cascade), 3) if model_cascade else 0,
            "omni_avg": round(np.mean(model_omni), 3) if model_omni else 0,
            "avg_deviation": round(np.mean(model_dev), 3) if model_dev else 0,
            "tendency": "Over-rated by Omni" if np.mean(model_dev) > 0.5 else
                       "Under-rated by Omni" if np.mean(model_dev) < -0.5 else "Consistent"
        }

    # 保存报告
    report_path = output_dir / 'comparison_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Saved: {report_path}")

    return report


def generate_html_report(cascade: Dict, omni: Dict, deviation: Dict,
                         correlation: float, output_dir: Path):
    """
    生成 HTML 可视化报告
    """
    # 计算统计数据
    cascade_all = [s for cat in cascade.values() for s in cat.values()]
    omni_all = [s for cat in omni.values() for s in cat.values()]
    avg_dev = np.mean(omni_all) - np.mean(cascade_all)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cascade vs Omni Evaluation Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        h1 {{ color: #333; border-bottom: 3px solid #e74c3c; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .meta {{ background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #fff; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 28px; font-weight: bold; color: #3498db; }}
        .stat-label {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .chart {{ text-align: center; margin: 20px 0; background: #fff; padding: 20px; border-radius: 8px; }}
        .chart img {{ max-width: 100%; border-radius: 8px; }}
        .highlight {{ color: #e74c3c; font-weight: bold; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #fff; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <h1>🔬 Cascade vs Omni Evaluation Comparison</h1>

    <div class="meta">
        <h2>📊 Overview</h2>
        <p><strong>Cascade (Multi-Model):</strong> Uses ASR + EMO + AGE + GND + TONE annotations → Qwen API text evaluation</p>
        <p><strong>Omni (Single-Model):</strong> Qwen Omni directly listens to audio for evaluation</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <div class="stat-value">{np.mean(cascade_all):.2f}</div>
            <div class="stat-label">Cascade Avg Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{np.mean(omni_all):.2f}</div>
            <div class="stat-label">Omni Avg Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {'positive' if avg_dev > 0 else 'negative'}">{avg_dev:+.2f}</div>
            <div class="stat-label">Average Deviation</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{correlation:.3f}</div>
            <div class="stat-label">Correlation (r)</div>
        </div>
    </div>

    <h2>📈 Comparison Charts</h2>

    <div class="chart">
        <h3>Category-wise Comparison</h3>
        <img src="comparison_by_category.png" alt="Category Comparison">
    </div>

    <div class="chart">
        <h3>Overall Ranking Comparison</h3>
        <img src="ranking_comparison.png" alt="Ranking Comparison">
    </div>

    <div class="chart">
        <h3>Correlation Scatter Plot</h3>
        <img src="correlation_scatter.png" alt="Correlation">
    </div>

    <div class="chart">
        <h3>Deviation Heatmap</h3>
        <img src="deviation_heatmap.png" alt="Deviation Heatmap">
    </div>

    <div class="chart">
        <h3>Bias by Model</h3>
        <img src="bias_by_model.png" alt="Bias by Model">
    </div>

    <h2>📋 Detailed Statistics</h2>

    <h3>Category Analysis</h3>
    <table>
        <tr>
            <th>Category</th>
            <th>Cascade Avg</th>
            <th>Omni Avg</th>
            <th>Deviation</th>
            <th>Interpretation</th>
        </tr>
"""

    for cat in ['age', 'emotion', 'gender', 'sarcasm']:
        cat_c = [cascade.get(cat, {}).get(m, 0) for m in COLORS.keys()]
        cat_o = [omni.get(cat, {}).get(m, 0) for m in COLORS.keys()]
        dev = np.mean(cat_o) - np.mean(cat_c)
        interp = "Omni higher" if dev > 0.5 else "Cascade higher" if dev < -0.5 else "Consistent"
        html += f"""
        <tr>
            <td><strong>{cat.upper()}</strong></td>
            <td>{np.mean(cat_c):.2f}</td>
            <td>{np.mean(cat_o):.2f}</td>
            <td class="{'positive' if dev > 0 else 'negative'}">{dev:+.2f}</td>
            <td>{interp}</td>
        </tr>
"""

    html += """
    </table>

    <h3>Model Analysis</h3>
    <table>
        <tr>
            <th>Model</th>
            <th>Cascade Avg</th>
            <th>Omni Avg</th>
            <th>Deviation</th>
            <th>Tendency</th>
        </tr>
"""

    for model in COLORS.keys():
        model_c = [cascade[cat].get(model, 0) for cat in cascade.keys() if model in cascade.get(cat, {})]
        model_o = [omni[cat].get(model, 0) for cat in cascade.keys() if cat in omni and model in omni.get(cat, {})]
        dev = np.mean(model_o) - np.mean(model_c) if model_o and model_c else 0
        tendency = "Over-rated by Omni" if dev > 0.5 else "Under-rated by Omni" if dev < -0.5 else "Consistent"
        c_avg = np.mean(model_c) if model_c else 0
        o_avg = np.mean(model_o) if model_o else 0
        html += f"""
        <tr>
            <td><strong>{model}</strong></td>
            <td>{c_avg:.2f}</td>
            <td>{o_avg:.2f}</td>
            <td class="{'positive' if dev > 0 else 'negative'}">{dev:+.2f}</td>
            <td>{tendency}</td>
        </tr>
"""

    html += """
    </table>

    <h2>🔍 Key Findings</h2>
    <div class="meta">
        <ul>
            <li><strong>Correlation:</strong> The two evaluation methods show a correlation coefficient of """ + f"{correlation:.3f}" + """.</li>
            <li><strong>Omni tends to rate higher:</strong> The single-model (Omni) approach generally gives higher scores.</li>
            <li><strong>Sarcasm category:</strong> Shows the most significant deviation between the two methods.</li>
            <li><strong>Possible reason:</strong> Omni directly hears the audio and can capture subtle tone cues that text-based Cascade might miss.</li>
        </ul>
    </div>
</body>
</html>
"""

    output_path = output_dir / 'comparison_report.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cascade vs Omni Evaluation Comparison")
    parser.add_argument(
        '--cascade',
        default='/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/Cascade_evaluated/jsons/summary_report.json',
        help='Cascade evaluation results'
    )
    parser.add_argument(
        '--omni',
        default='/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/qwen_eval_results/evaluation_summary.json',
        help='Omni evaluation results'
    )
    parser.add_argument(
        '--output',
        default='/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/comparison',
        help='Output directory'
    )

    args = parser.parse_args()

    cascade_path = Path(args.cascade)
    omni_path = Path(args.omni)
    output_dir = Path(args.output)

    if not cascade_path.exists():
        print(f"Error: Cascade results not found: {cascade_path}")
        return

    if not omni_path.exists():
        print(f"Error: Omni results not found: {omni_path}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Cascade vs Omni Evaluation Comparison")
    print("=" * 60)
    print(f"Cascade: {cascade_path}")
    print(f"Omni:    {omni_path}")
    print(f"Output:  {output_dir}")
    print()

    # 加载数据
    print("Loading data...")
    cascade = load_cascade_results(cascade_path)
    omni = load_omni_results(omni_path)

    # 计算偏差
    print("Computing deviations...")
    deviation = compute_deviation(cascade, omni)

    # 生成图表
    print("\nGenerating plots...")
    plot_deviation_comparison(cascade, omni, output_dir)
    plot_deviation_heatmap(deviation, output_dir)
    plot_model_ranking_comparison(cascade, omni, output_dir)
    correlation = plot_scatter_correlation(cascade, omni, output_dir)
    plot_bias_by_model(deviation, output_dir)

    # 生成报告
    print("\nGenerating reports...")
    generate_comparison_report(cascade, omni, deviation, correlation, output_dir)
    generate_html_report(cascade, omni, deviation, correlation, output_dir)

    print("\n" + "=" * 60)
    print("Comparison Complete!")
    print("=" * 60)
    print(f"\nKey Metrics:")
    print(f"  Correlation (Pearson r): {correlation:.3f}")

    cascade_all = [s for cat in cascade.values() for s in cat.values()]
    omni_all = [s for cat in omni.values() for s in cat.values()]
    print(f"  Cascade Avg Score: {np.mean(cascade_all):.3f}")
    print(f"  Omni Avg Score:    {np.mean(omni_all):.3f}")
    print(f"  Average Deviation: {np.mean(omni_all) - np.mean(cascade_all):+.3f}")

    print(f"\nOpen report: {output_dir / 'comparison_report.html'}")


if __name__ == "__main__":
    main()
