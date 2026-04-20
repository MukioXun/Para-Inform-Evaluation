"""
人工标注结果可视化分析
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
ANNOTATION_DIR = Path("/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/Results/human_annotations")
OUTPUT_DIR = Path("/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/Results/visualization")
OUTPUT_DIR.mkdir(exist_ok=True)

# 积极情感定义
POSITIVE_EMOTIONS = ['happy']

# 模型名称映射
MODEL_NAMES = {
    'model_glm4_score': 'GLM4',
    'model_gpt-4o-voice-mode_score': 'GPT-4o',
    'model_llamaomni2_score': 'LlamaOmni2',
    'model_original_score': 'Original',
    'model_qwen2_5_score': 'Qwen2.5',
    'model_rl_real_all_score': 'RL-Real-All'
}

def load_all_annotations():
    """加载所有标注数据"""
    data = defaultdict(list)

    for category in ['age', 'emotion', 'gender', 'sarcasm']:
        cat_dir = ANNOTATION_DIR / category
        if not cat_dir.exists():
            continue

        for json_file in cat_dir.glob("*.json"):
            with open(json_file, 'r', encoding='utf-8') as f:
                item = json.load(f)
                data[category].append(item)

    return data

def analyze_age_category(data):
    """分析age类别，单独统计child(littlekid)结果"""
    age_data = data['age']

    # 按ground_truth分组
    results = {'adult': [], 'littlekid': []}

    for item in age_data:
        gt = item['ground_truth']
        if gt in results:
            results[gt].append(item)

    return results

def analyze_emotion_category(data):
    """分析emotion类别，统计非积极情感结果"""
    emotion_data = data['emotion']

    positive = []
    non_positive = []

    for item in emotion_data:
        gt = item['ground_truth']
        if gt in POSITIVE_EMOTIONS:
            positive.append(item)
        else:
            non_positive.append(item)

    return {'positive': positive, 'non_positive': non_positive}

def analyze_sarcasm_category(data):
    """分析sarcasm类别，统计讽刺样本"""
    sarcasm_data = data['sarcasm']

    sarcastic = []
    sincere = []

    for item in sarcasm_data:
        gt = item['ground_truth']
        if gt == 'sarcastic':
            sarcastic.append(item)
        else:
            sincere.append(item)

    return {'sarcastic': sarcastic, 'sincere': sincere}

def calculate_scores(items, score_keys):
    """计算评分统计"""
    stats = defaultdict(lambda: {'scores': [], 'mean': 0, 'std': 0})

    for key in score_keys:
        for item in items:
            score = item['annotations'].get(key)
            if score is not None:
                stats[key]['scores'].append(score)

        if stats[key]['scores']:
            stats[key]['mean'] = np.mean(stats[key]['scores'])
            stats[key]['std'] = np.std(stats[key]['scores'])

    return dict(stats)

def plot_model_comparison(stats, title, save_path, model_names=None):
    """绘制模型对比图"""
    if not stats:
        return

    # 过滤只保留模型分数
    model_keys = [k for k in stats.keys() if k.startswith('model_')]

    if not model_keys:
        return

    names = []
    means = []
    stds = []

    for key in model_keys:
        display_name = model_names.get(key, key.replace('model_', '').replace('_score', '')) if model_names else key
        names.append(display_name)
        means.append(stats[key]['mean'])
        stds.append(stats[key]['std'])

    # 排序
    sorted_indices = np.argsort(means)[::-1]
    names = [names[i] for i in sorted_indices]
    means = [means[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(names))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='Neutral (3)')
    ax.legend()

    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax.annotate(f'{mean:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return dict(zip(names, means))

def plot_score_distribution(stats, title, save_path):
    """绘制分数分布图"""
    model_keys = [k for k in stats.keys() if k.startswith('model_')]

    if not model_keys:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data_to_plot = []
    labels = []

    for key in model_keys:
        scores = stats[key]['scores']
        if scores:
            data_to_plot.append(scores)
            labels.append(MODEL_NAMES.get(key, key))

    bp = ax.boxplot(data_to_plot, patch_artist=True, labels=labels)

    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 6)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_user_attribute_scores(stats, title, save_path):
    """绘制用户属性识别评分"""
    attr_keys = ['user_emotion_score', 'user_age_score', 'user_gender_score']
    attr_names = ['Emotion', 'Age', 'Gender']

    means = []
    stds = []

    for key in attr_keys:
        if key in stats and stats[key]['scores']:
            means.append(stats[key]['mean'])
            stds.append(stats[key]['std'])
        else:
            means.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(attr_names))
    colors = ['#3b82f6', '#10b981', '#f59e0b']

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Attribute', fontsize=12)
    ax.set_ylabel('Score (1-5)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(attr_names)
    ax.set_ylim(0, 5.5)
    ax.axhline(y=3, color='gray', linestyle='--', alpha=0.5)

    for bar, mean in zip(bars, means):
        ax.annotate(f'{mean:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_summary_report(data):
    """生成汇总报告"""
    report = []
    report.append("=" * 70)
    report.append("人工标注结果分析报告")
    report.append("=" * 70)

    # 总体统计
    total = sum(len(items) for items in data.values())
    report.append(f"\n📊 总体统计: 共 {total} 条标注数据")
    for cat, items in data.items():
        report.append(f"   - {cat}: {len(items)} 条")

    # Age 分析
    report.append("\n" + "=" * 70)
    report.append("📅 AGE 类别分析")
    report.append("=" * 70)

    age_results = analyze_age_category(data)

    for gt in ['adult', 'littlekid']:
        items = age_results[gt]
        if items:
            stats = calculate_scores(items, ['user_emotion_score', 'user_age_score', 'user_gender_score'] + list(MODEL_NAMES.keys()))

            report.append(f"\n  【{gt.upper()}】 ({len(items)} 条)")
            report.append(f"    用户属性识别:")
            report.append(f"      - 情感识别: {stats['user_emotion_score']['mean']:.2f} ± {stats['user_emotion_score']['std']:.2f}")
            report.append(f"      - 年龄识别: {stats['user_age_score']['mean']:.2f} ± {stats['user_age_score']['std']:.2f}")
            report.append(f"      - 性别识别: {stats['user_gender_score']['mean']:.2f} ± {stats['user_gender_score']['std']:.2f}")

            report.append(f"    模型响应评分:")
            for key, name in MODEL_NAMES.items():
                if key in stats and stats[key]['mean'] > 0:
                    report.append(f"      - {name}: {stats[key]['mean']:.2f} ± {stats[key]['std']:.2f}")

    # Emotion 分析
    report.append("\n" + "=" * 70)
    report.append("😊 EMOTION 类别分析")
    report.append("=" * 70)

    emotion_results = analyze_emotion_category(data)

    for label, items in [('积极情感 (happy)', emotion_results['positive']),
                         ('非积极情感', emotion_results['non_positive'])]:
        if items:
            stats = calculate_scores(items, list(MODEL_NAMES.keys()))

            report.append(f"\n  【{label}】 ({len(items)} 条)")
            for key, name in MODEL_NAMES.items():
                if key in stats and stats[key]['mean'] > 0:
                    report.append(f"    - {name}: {stats[key]['mean']:.2f} ± {stats[key]['std']:.2f}")

    # Sarcasm 分析
    report.append("\n" + "=" * 70)
    report.append("🎭 SARCASM 类别分析")
    report.append("=" * 70)

    sarcasm_results = analyze_sarcasm_category(data)

    for label, items in [('讽刺 (sarcastic)', sarcasm_results['sarcastic']),
                         ('真诚 (sincere)', sarcasm_results['sincere'])]:
        if items:
            stats = calculate_scores(items, list(MODEL_NAMES.keys()))

            report.append(f"\n  【{label}】 ({len(items)} 条)")
            for key, name in MODEL_NAMES.items():
                if key in stats and stats[key]['mean'] > 0:
                    report.append(f"    - {name}: {stats[key]['mean']:.2f} ± {stats[key]['std']:.2f}")

    # Gender 分析
    report.append("\n" + "=" * 70)
    report.append("⚧ GENDER 类别分析")
    report.append("=" * 70)

    gender_data = data.get('gender', [])
    if gender_data:
        stats = calculate_scores(gender_data, list(MODEL_NAMES.keys()))
        report.append(f"\n  共 {len(gender_data)} 条标注")
        for key, name in MODEL_NAMES.items():
            if key in stats and stats[key]['mean'] > 0:
                report.append(f"    - {name}: {stats[key]['mean']:.2f} ± {stats[key]['std']:.2f}")

    report.append("\n" + "=" * 70)

    return "\n".join(report)

def main():
    print("加载标注数据...")
    data = load_all_annotations()

    print("生成可视化图表...")

    # 1. Age 分析 - Adult vs Littlekid
    age_results = analyze_age_category(data)

    for gt in ['adult', 'littlekid']:
        items = age_results[gt]
        if items:
            stats = calculate_scores(items, list(MODEL_NAMES.keys()))
            plot_model_comparison(stats, f'Model Response Scores - {gt.upper()}',
                                 OUTPUT_DIR / f'age_{gt}_model_scores.png', MODEL_NAMES)

            stats_all = calculate_scores(items, ['user_emotion_score', 'user_age_score', 'user_gender_score'] + list(MODEL_NAMES.keys()))
            plot_user_attribute_scores(stats_all, f'User Attribute Recognition - {gt.upper()}',
                                       OUTPUT_DIR / f'age_{gt}_user_attrs.png')

    # 2. Emotion 分析 - 积极vs非积极
    emotion_results = analyze_emotion_category(data)

    for label, items in [('positive', emotion_results['positive']),
                         ('non_positive', emotion_results['non_positive'])]:
        if items:
            stats = calculate_scores(items, list(MODEL_NAMES.keys()))
            plot_model_comparison(stats, f'Model Response Scores - {label.replace("_", " ").title()} Emotion',
                                 OUTPUT_DIR / f'emotion_{label}_model_scores.png', MODEL_NAMES)

    # 3. Sarcasm 分析
    sarcasm_results = analyze_sarcasm_category(data)

    for label, items in [('sarcastic', sarcasm_results['sarcastic']),
                         ('sincere', sarcasm_results['sincere'])]:
        if items:
            stats = calculate_scores(items, list(MODEL_NAMES.keys()))
            plot_model_comparison(stats, f'Model Response Scores - {label.title()}',
                                 OUTPUT_DIR / f'sarcasm_{label}_model_scores.png', MODEL_NAMES)

    # 4. 总体对比
    all_model_stats = calculate_scores(
        [item for items in data.values() for item in items],
        list(MODEL_NAMES.keys())
    )
    plot_model_comparison(all_model_stats, 'Overall Model Response Scores',
                         OUTPUT_DIR / 'overall_model_scores.png', MODEL_NAMES)

    # 生成报告
    print("\n生成分析报告...")
    report = generate_summary_report(data)

    # 保存报告
    report_path = OUTPUT_DIR / 'annotation_summary.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n✅ 可视化结果已保存到: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
