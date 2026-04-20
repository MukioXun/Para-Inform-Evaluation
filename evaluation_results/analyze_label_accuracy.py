#!/usr/bin/env python3
"""
分析用户语音的标签与标注器标注信息的成功率
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# 定义标签映射关系
AGE_MAPPING = {
    'adult': ['middle_aged', 'young_adult', 'adult'],
    'littlekid': ['child', 'kid', 'little_kid', 'littlekid']
}

EMOTION_MAPPING = {
    'happy': ['happy'],
    'sad': ['sad'],
    'angry': ['angry'],
    'fearful': ['fearful'],
    'disgust': ['disgusted', 'disgust'],
    'surprised': ['surprised']
}

GENDER_MAPPING = {
    'male': ['male'],
    'female': ['female']
}

SARCASM_MAPPING = {
    'sarcastic': ['sarcastic'],
    'sincere': ['sincere', 'neutral']
}

def check_match(predicted, ground_truth, category):
    """检查预测值是否与真实标签匹配"""
    if predicted is None or ground_truth is None:
        return False

    predicted_lower = predicted.lower()
    ground_truth_lower = ground_truth.lower()

    if category == 'age':
        mapping = AGE_MAPPING
    elif category == 'emotion':
        mapping = EMOTION_MAPPING
    elif category == 'gender':
        mapping = GENDER_MAPPING
    elif category == 'sarcasm':
        mapping = SARCASM_MAPPING
    else:
        return predicted_lower == ground_truth_lower

    # 获取该标签对应的所有可能预测值
    valid_predictions = mapping.get(ground_truth_lower, [ground_truth_lower])

    return predicted_lower in valid_predictions

def analyze_category(category_dir, category_name):
    """分析单个类别的成功率"""
    results = {
        'total': 0,
        'correct': 0,
        'details': [],
        'by_label': defaultdict(lambda: {'total': 0, 'correct': 0}),
        'confidence_stats': []
    }

    json_files = list(category_dir.glob('*.json'))

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        ground_truth = data['label']
        category = data['category']

        for pair in data['pairs']:
            input_annotation = pair['input']['annotation']

            # 根据类别获取预测值
            if category == 'age':
                predicted = input_annotation.get('AGE', {}).get('age_group')
                confidence = input_annotation.get('AGE', {}).get('confidence')
            elif category == 'emotion':
                predicted = input_annotation.get('EMO', {}).get('emotion')
                confidence = input_annotation.get('EMO', {}).get('confidence')
            elif category == 'gender':
                predicted = input_annotation.get('GND', {}).get('gender')
                confidence = input_annotation.get('GND', {}).get('confidence')
            elif category == 'sarcasm':
                # 查看sarcasm是否有专门的字段，如果没有则跳过
                predicted = input_annotation.get('SARCASM', {}).get('label') if 'SARCASM' in input_annotation else None
                confidence = input_annotation.get('SARCASM', {}).get('confidence') if 'SARCASM' in input_annotation else None
                if predicted is None:
                    continue
            else:
                continue

            is_match = check_match(predicted, ground_truth, category)

            results['total'] += 1
            results['by_label'][ground_truth]['total'] += 1

            if is_match:
                results['correct'] += 1
                results['by_label'][ground_truth]['correct'] += 1

            results['details'].append({
                'file': json_file.name,
                'ground_truth': ground_truth,
                'predicted': predicted,
                'confidence': confidence,
                'match': is_match
            })

            if confidence is not None:
                results['confidence_stats'].append({
                    'confidence': confidence,
                    'match': is_match
                })

    return results

def main():
    base_dir = Path('/home/u2023112559/qix/Project/Final_Project/Audio_Captior/evaluation_results')

    categories = ['age', 'emotion', 'gender', 'sarcasm']

    print("=" * 70)
    print("用户语音标签与标注器标注信息成功率分析")
    print("=" * 70)

    all_results = {}

    for category in categories:
        category_dir = base_dir / category
        if not category_dir.exists():
            print(f"\n类别 '{category}' 目录不存在，跳过")
            continue

        print(f"\n{'='*70}")
        print(f"类别: {category.upper()}")
        print(f"{'='*70}")

        results = analyze_category(category_dir, category)
        all_results[category] = results

        if results['total'] == 0:
            print("  无有效数据")
            continue

        # 总体成功率
        overall_rate = results['correct'] / results['total'] * 100
        print(f"\n总体统计:")
        print(f"  总样本数: {results['total']}")
        print(f"  正确预测: {results['correct']}")
        print(f"  成功率: {overall_rate:.2f}%")

        # 按标签分组统计
        print(f"\n按标签分组统计:")
        for label, stats in results['by_label'].items():
            rate = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {label}: {stats['correct']}/{stats['total']} ({rate:.2f}%)")

        # 置信度分析
        if results['confidence_stats']:
            correct_conf = [s['confidence'] for s in results['confidence_stats'] if s['match']]
            wrong_conf = [s['confidence'] for s in results['confidence_stats'] if not s['match']]

            print(f"\n置信度分析:")
            if correct_conf:
                print(f"  正确预测平均置信度: {sum(correct_conf)/len(correct_conf):.4f}")
            if wrong_conf:
                print(f"  错误预测平均置信度: {sum(wrong_conf)/len(wrong_conf):.4f}")

    # 汇总报告
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    print(f"\n{'类别':<15} {'总数':<10} {'正确':<10} {'成功率':<10}")
    print("-" * 45)

    for category, results in all_results.items():
        if results['total'] > 0:
            rate = results['correct'] / results['total'] * 100
            print(f"{category:<15} {results['total']:<10} {results['correct']:<10} {rate:.2f}%")

    # 保存详细结果到JSON
    output_file = base_dir / 'label_accuracy_analysis.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n详细结果已保存至: {output_file}")

if __name__ == '__main__':
    main()
