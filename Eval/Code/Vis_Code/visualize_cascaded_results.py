#!/usr/bin/env python3
"""
Evaluation Results Visualization Script
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use default font
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_summary():
    """Load summary report"""
    summary_file = Path(__file__).parent / "summary_report.json"
    with open(summary_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_visualizations(summary):
    """Create all visualization charts"""
    categories = ['age', 'sarcasm', 'emotion', 'gender']
    category_names = {'age': 'Age', 'sarcasm': 'Sarcasm',
                      'emotion': 'Emotion', 'gender': 'Gender'}
    models = ['glm4', 'gpt-4o-voice-mode', 'llamaomni2', 'original', 'qwen2.5', 'rl_real_all']
    model_colors = {
        'glm4': '#1f77b4',
        'gpt-4o-voice-mode': '#ff7f0e',
        'llamaomni2': '#2ca02c',
        'original': '#d62728',
        'qwen2.5': '#9467bd',
        'rl_real_all': '#8c564b'
    }

    # Figure 1: Model scores by category
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Model Performance by Category (Average Score)', fontsize=16, fontweight='bold')

    for idx, category in enumerate(categories):
        ax = axes1[idx // 2, idx % 2]
        scores = [summary['statistics'][category].get(model, {}).get('average_score', 0) for model in models]
        bars = ax.bar(models, scores, color=[model_colors[m] for m in models], edgecolor='black', linewidth=0.5)
        ax.set_title(category_names[category], fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Score', fontsize=10)
        ax.set_ylim(0, 5)
        ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Average Line')
        ax.tick_params(axis='x', rotation=30)

        for bar, score in zip(bars, scores):
            ax.annotate(f'{score:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    fig1.savefig('evaluation_by_category.png', dpi=150, bbox_inches='tight')
    print("Saved: evaluation_by_category.png")

    # Figure 2: Overall ranking
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    overall_scores = {}
    for model in models:
        total_score = 0
        count = 0
        for category in categories:
            if model in summary['statistics'][category]:
                total_score += summary['statistics'][category][model]['average_score']
                count += 1
        overall_scores[model] = total_score / count if count > 0 else 0

    sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
    model_names = [m[0] for m in sorted_models]
    scores = [m[1] for m in sorted_models]

    bars = ax2.barh(model_names, scores, color=[model_colors[m] for m in model_names],
                    edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Overall Average Score', fontsize=12)
    ax2.set_title('Overall Model Ranking', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 5)
    ax2.axvline(x=3, color='gray', linestyle='--', alpha=0.5)

    for bar, score in zip(bars, scores):
        ax2.annotate(f'{score:.2f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    ha='left', va='center', fontsize=10, fontweight='bold', xytext=(5, 0),
                    textcoords='offset points')

    plt.tight_layout()
    fig2.savefig('evaluation_overall_ranking.png', dpi=150, bbox_inches='tight')
    print("Saved: evaluation_overall_ranking.png")

    # Figure 3: Score distribution stacked bar chart
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Score Distribution (1-5) by Model', fontsize=16, fontweight='bold')

    score_colors = ['#d62728', '#ff7f0e', '#ffd700', '#2ca02c', '#1f77b4']

    for idx, category in enumerate(categories):
        ax = axes3[idx // 2, idx % 2]
        x = np.arange(len(models))
        width = 0.6

        bottom = np.zeros(len(models))
        for score in range(1, 6):
            counts = [summary['statistics'][category].get(model, {}).get('score_distribution', {}).get(str(score), 0)
                     for model in models]
            ax.bar(x, counts, width, bottom=bottom, label=f'Score {score}', color=score_colors[score-1], edgecolor='white', linewidth=0.5)
            bottom += counts

        ax.set_title(category_names[category], fontsize=12, fontweight='bold')
        ax.set_ylabel('Evaluation Count', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.legend(title='Score', loc='upper right', ncol=5, fontsize=8)

    plt.tight_layout()
    fig3.savefig('evaluation_score_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: evaluation_score_distribution.png")

    # Figure 4: Heatmap
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    heatmap_data = np.zeros((len(models), len(categories)))
    for i, model in enumerate(models):
        for j, category in enumerate(categories):
            heatmap_data[i, j] = summary['statistics'][category].get(model, {}).get('average_score', 0)

    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=1, vmax=5)

    ax4.set_xticks(np.arange(len(categories)))
    ax4.set_yticks(np.arange(len(models)))
    ax4.set_xticklabels([category_names[c] for c in categories], fontsize=11)
    ax4.set_yticklabels(models, fontsize=11)

    for i in range(len(models)):
        for j in range(len(categories)):
            text = ax4.text(j, i, f'{heatmap_data[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=11, fontweight='bold')

    ax4.set_title('Model Performance Heatmap (Higher is Better)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Average Score', fontsize=11)

    plt.tight_layout()
    fig4.savefig('evaluation_heatmap.png', dpi=150, bbox_inches='tight')
    print("Saved: evaluation_heatmap.png")

    # Figure 5: Radar chart
    fig5 = plt.figure(figsize=(10, 10))
    ax5 = fig5.add_subplot(111, projection='polar')

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    for model in models:
        values = [summary['statistics'][category].get(model, {}).get('average_score', 0) for category in categories]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth=2, label=model, color=model_colors[model])
        ax5.fill(angles, values, alpha=0.1, color=model_colors[model])

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels([category_names[c] for c in categories], fontsize=11)
    ax5.set_ylim(0, 5)
    ax5.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.tight_layout()
    fig5.savefig('evaluation_radar.png', dpi=150, bbox_inches='tight')
    print("Saved: evaluation_radar.png")

    plt.close('all')

    print("\nVisualization complete! 5 charts generated.")

if __name__ == "__main__":
    summary = load_summary()
    create_visualizations(summary)
