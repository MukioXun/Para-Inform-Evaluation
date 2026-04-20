"""
Cascade 自动评估结果可视化分析
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
CASCADE_DIR = Path("/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/Results/Cascade_evaluated")
JSON_DIR = CASCADE_DIR / "jsons"
OUTPUT_DIR = CASCADE_DIR / "visualization"
OUTPUT_DIR.mkdir(exist_ok=True)

# 模型名称
MODEL_DISPLAY = ['GLM4', 'GPT-4o', 'LlamaOmni2', 'Original', 'Qwen2.5', 'RL-Real-All']
MODEL_KEYS = ['glm4', 'gpt-4o-voice-mode', 'llamaomni2', 'original', 'qwen2.5', 'rl_real_all']

# 积极情感
POSITIVE_EMOTIONS = ['happy']

def load_all_data():
    """加载所有评估数据"""
    data = defaultdict(list)

    for json_file in JSON_DIR.glob("evaluated_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            item = json.load(f)
            category = item.get('category')
            if category:
                data[category].append(item)

    return data

def analyze_data(data):
    """分析数据，按类别和子类别统计"""

    # Age 分析
    age_results = {'adult': [], 'littlekid': []}
    for item in data['age']:
        label = item['label']
        if label in age_results:
            age_results[label].append(item)

    # Emotion 分析
    emotion_results = {'positive': [], 'non_positive': []}
    for item in data['emotion']:
        label = item['label']
        if label in POSITIVE_EMOTIONS:
            emotion_results['positive'].append(item)
        else:
            emotion_results['non_positive'].append(item)

    # Sarcasm 分析
    sarcasm_results = {'sarcastic': [], 'sincere': []}
    for item in data['sarcasm']:
        label = item['label']
        if label in sarcasm_results:
            sarcasm_results[label].append(item)

    # Gender 分析
    gender_results = {'male': [], 'female': []}
    for item in data['gender']:
        label = item['label']
        if label in gender_results:
            gender_results[label].append(item)

    return {
        'age': age_results,
        'emotion': emotion_results,
        'sarcasm': sarcasm_results,
        'gender': gender_results
    }

def calc_model_scores(items):
    """计算每个模型的平均分"""
    scores = {}
    for model_key in MODEL_KEYS:
        model_scores = []
        for item in items:
            for eval_result in item['evaluation_results']:
                if eval_result['model'] == model_key:
                    model_scores.append(eval_result['score'])
        if model_scores:
            scores[model_key] = {
                'mean': np.mean(model_scores),
                'std': np.std(model_scores),
                'scores': model_scores
            }
    return scores

def create_heatmap(score_matrix, row_labels, col_labels, title, save_path):
    """创建热力图"""
    fig, ax = plt.subplots(figsize=(14, max(8, len(row_labels) * 0.8)))

    sns.heatmap(score_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlGn',
                vmin=1, vmax=5,
                xticklabels=col_labels,
                yticklabels=row_labels,
                linewidths=1.5,
                linecolor='white',
                annot_kws={'fontsize': 12, 'fontweight': 'bold'},
                cbar_kws={'label': 'Cascade Score (1-5)', 'shrink': 0.8})

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Scenario', fontsize=13)

    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def create_bar_comparison(ideal_scores, challenge_scores, title, save_path):
    """创建对比柱状图"""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(MODEL_DISPLAY))
    width = 0.35

    bars1 = ax.bar(x - width/2, ideal_scores, width, label='Ideal Scenarios',
                   color='#22c55e', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, challenge_scores, width, label='Challenging Scenarios',
                   color='#ef4444', edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Average Cascade Score (1-5)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_DISPLAY, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.7)
    ax.legend(fontsize=11)

    for bar, val in zip(bars1, ideal_scores):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
    for bar, val in zip(bars2, challenge_scores):
        ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 4), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    print("Loading Cascade evaluation data...")
    data = load_all_data()

    total = sum(len(items) for items in data.values())
    print(f"Total samples: {total}")
    for cat, items in data.items():
        print(f"  - {cat}: {len(items)}")

    print("\nAnalyzing data...")
    analyzed = analyze_data(data)

    # 构建完整评分矩阵
    scenarios = []
    score_matrix = []

    # Age
    for label in ['adult', 'littlekid']:
        items = analyzed['age'][label]
        if items:
            scores = calc_model_scores(items)
            row = [scores[k]['mean'] for k in MODEL_KEYS]
            score_matrix.append(row)
            scenarios.append(f"Age: {label.upper()}")

    # Emotion
    for label in ['positive', 'non_positive']:
        items = analyzed['emotion'][label]
        if items:
            scores = calc_model_scores(items)
            row = [scores[k]['mean'] for k in MODEL_KEYS]
            score_matrix.append(row)
            scenarios.append(f"Emotion: {label.replace('_', ' ').title()}")

    # Sarcasm
    for label in ['sincere', 'sarcastic']:
        items = analyzed['sarcasm'][label]
        if items:
            scores = calc_model_scores(items)
            row = [scores[k]['mean'] for k in MODEL_KEYS]
            score_matrix.append(row)
            scenarios.append(f"Speech: {label.upper()}")

    # Gender
    for label in ['male', 'female']:
        items = analyzed['gender'][label]
        if items:
            scores = calc_model_scores(items)
            row = [scores[k]['mean'] for k in MODEL_KEYS]
            score_matrix.append(row)
            scenarios.append(f"Gender: {label.upper()}")

    score_matrix = np.array(score_matrix)

    # 1. 主热力图
    print("\nGenerating visualizations...")
    create_heatmap(score_matrix, scenarios, MODEL_DISPLAY,
                  'Cascade Evaluation: Model Scores by Scenario',
                  OUTPUT_DIR / 'cascade_scenario_heatmap.png')

    # 2. 理想 vs 挑战场景
    ideal_indices = [0, 2, 4]  # Adult, Positive, Sincere
    challenge_indices = [1, 3, 5]  # LittleKid, Non-Positive, Sarcastic

    ideal_avg = score_matrix[ideal_indices].mean(axis=0)
    challenge_avg = score_matrix[challenge_indices].mean(axis=0)

    create_bar_comparison(ideal_avg, challenge_avg,
                         'Cascade Evaluation: Ideal vs Challenging Scenarios',
                         OUTPUT_DIR / 'cascade_ideal_challenge.png')

    # 3. 各类别详细分析
    for category in ['age', 'emotion', 'sarcasm', 'gender']:
        cat_scores = []
        cat_labels = []

        for label, items in analyzed[category].items():
            if items:
                scores = calc_model_scores(items)
                row = [scores[k]['mean'] for k in MODEL_KEYS]
                cat_scores.append(row)
                cat_labels.append(f"{label.replace('_', ' ').title()}")

        if cat_scores:
            cat_matrix = np.array(cat_scores)
            create_heatmap(cat_matrix, cat_labels, MODEL_DISPLAY,
                          f'Cascade Evaluation: {category.upper()} Category',
                          OUTPUT_DIR / f'cascade_{category}_heatmap.png')

    # 4. 性能差距分析
    gap = ideal_avg - challenge_avg

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#22c55e' if g < 0.5 else '#f59e0b' if g < 1.5 else '#ef4444' for g in gap]

    bars = ax.bar(MODEL_DISPLAY, gap, color=colors, edgecolor='black', linewidth=1.5)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Model', fontsize=13)
    ax.set_ylabel('Performance Gap (Ideal - Challenging)', fontsize=13)
    ax.set_title('Cascade Evaluation: Model Robustness (Smaller Gap = More Robust)', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.5, 2)

    plt.xticks(rotation=45, ha='right', fontsize=12)

    for bar, g in zip(bars, gap):
        status = 'ROBUST' if g < 0.5 else 'Moderate' if g < 1.5 else 'NOT ROBUST'
        color = '#059669' if g < 0.5 else '#d97706' if g < 1.5 else '#dc2626'
        ax.annotate(f'{g:.2f}\n{status}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10,
                   fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cascade_performance_gap.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    # 打印统计
    print("\n" + "=" * 70)
    print("CASCADE EVALUATION RESULTS")
    print("=" * 70)

    print("\n[IDEAL SCENARIOS] Adult + Positive + Sincere:")
    for i, m in enumerate(MODEL_DISPLAY):
        print(f"   {m}: {ideal_avg[i]:.2f}")

    print("\n[CHALLENGING SCENARIOS] Child + Non-Positive + Sarcastic:")
    for i, m in enumerate(MODEL_DISPLAY):
        print(f"   {m}: {challenge_avg[i]:.2f}")

    print("\n[ROBUSTNESS] Performance Gap:")
    for i, m in enumerate(MODEL_DISPLAY):
        status = "ROBUST" if gap[i] < 0.5 else "NOT ROBUST" if gap[i] > 1.5 else "MODERATE"
        print(f"   {m}: {gap[i]:.2f} - {status}")

    print("\n" + "=" * 70)
    print(f"Visualizations saved to: {OUTPUT_DIR}")

    return score_matrix, scenarios, ideal_avg, challenge_avg, gap

if __name__ == '__main__':
    main()
