#!/usr/bin/env python3
"""
对话质量评估脚本 - 使用Qwen API
"""

import json
import os
import re
from pathlib import Path
from dashscope import Generation
from datetime import datetime

# 评估prompt模板
EVALUATION_PROMPT = """You are a dialogue evaluation expert.
You are given:
- User speech information: transcription: "{user_transcription}", emotion: "{user_emotion}", sarcasm: "{user_sarcasm}", age: "{user_age}", gender: "{user_gender}"
- Agent response: transcription: "{agent_transcription}", tone: "{agent_tone}"

Your task:
Evaluate how well the agent's response fits the user in a natural conversation, considering both:
1. Content (what is said)
2. Speaking style (emotion, tone, speed, etc.)

Focus on whether the agent:
- Understands the user's emotion, sarcasm, age, and gender
- Responds with appropriate content and tone
- Maintains a natural and comfortable dialogue flow

---
### Scoring (1–5):
**5 (Perfect)**
Correctly understands user signals (emotion/sarcasm/identity) and responds with appropriate tone and high empathy.
**4 (Good)**
Content is appropriate, but tone/style does not fully enhance the interaction.
**3 (Average)**
Partially correct understanding, but response feels generic or slightly awkward.
**2 (Poor)**
Misunderstands or poorly handles user emotion/style; noticeable mismatch.
**1 (Bad)**
Completely mismatched (e.g., wrong emotion, wrong tone, sarcasm misunderstood).
---
### Key checks:
- Emotion match (happy/sad/angry etc.)
- Sarcasm understanding
- Age appropriateness (e.g., child vs adult)
- Gender appropriateness (no awkward mismatch)
- Tone consistency
---
### Output format:
The reason is <very brief reason>; The score is <1-5>."""


def extract_user_info(pair, category, label):
    """从pair中提取用户信息"""
    input_annotation = pair["input"]["annotation"]

    user_transcription = input_annotation.get("ASR", "").strip()
    user_emotion = input_annotation.get("EMO", {}).get("emotion", "unknown")
    user_age = input_annotation.get("AGE", {}).get("age_group", "unknown")
    user_gender = input_annotation.get("GND", {}).get("gender", "unknown")
    user_tone = input_annotation.get("TONE", {}).get("description", "")

    # 如果emotion是unknown，从category获取
    if user_emotion == "unknown" and category == "emotion":
        user_emotion = label

    # sarcasm处理
    user_sarcasm = "unknown"
    if category == "sarcasm":
        user_sarcasm = label

    return {
        "transcription": user_transcription,
        "emotion": user_emotion,
        "sarcasm": user_sarcasm,
        "age": user_age,
        "gender": user_gender,
        "tone": user_tone
    }


def extract_agent_info(pair):
    """从pair中提取agent信息"""
    output_annotation = pair["output"]["annotation"]

    return {
        "transcription": output_annotation.get("ASR", "").strip(),
        "tone": output_annotation.get("TONE", {}).get("description", ""),
        "model": pair["output"].get("model", "unknown")
    }


def call_qwen_api(prompt):
    """调用Qwen API"""
    try:
        response = Generation.call(
            model='qwen-turbo',
            prompt=prompt,
            max_tokens=200,
            temperature=0.3
        )

        if response.status_code == 200:
            return response.output.text
        else:
            return f"API Error: {response.code} - {response.message}"
    except Exception as e:
        return f"Exception: {str(e)}"


def parse_evaluation_result(result_text):
    """解析评估结果，提取分数和原因"""
    # 尝试匹配 "The reason is ...; The score is X"
    reason_match = re.search(r'The reason is\s*(.+?);\s*The score is\s*(\d)', result_text, re.IGNORECASE | re.DOTALL)

    if reason_match:
        reason = reason_match.group(1).strip()
        score = int(reason_match.group(2))
        return reason, score

    # 尝试其他格式 - 更灵活的匹配
    score_match = re.search(r'score[:\s]*(\d)', result_text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))
        # 尝试提取原因
        reason_match = re.search(r'reason[:\s]*(.+?)(?:score|;|\Z)', result_text, re.IGNORECASE | re.DOTALL)
        reason = reason_match.group(1).strip() if reason_match else "N/A"
        return reason, score

    # 尝试从文本中直接提取数字作为分数（最后一个1-5的数字）
    score_matches = re.findall(r'\b([1-5])\b', result_text)
    if score_matches:
        score = int(score_matches[-1])  # 取最后一个作为分数
        # 提取原因（分数之前的内容）
        reason = result_text.split(str(score))[0].strip()
        if reason:
            return reason[:200], score

    # 最后尝试：如果文本中有1-5的数字
    for digit in ['5', '4', '3', '2', '1']:
        if digit in result_text:
            return result_text[:200], int(digit)

    return result_text[:100], None


def evaluate_pair(pair, category, label):
    """评估单个对话对"""
    user_info = extract_user_info(pair, category, label)
    agent_info = extract_agent_info(pair)

    prompt = EVALUATION_PROMPT.format(
        user_transcription=user_info["transcription"],
        user_emotion=user_info["emotion"],
        user_sarcasm=user_info["sarcasm"],
        user_age=user_info["age"],
        user_gender=user_info["gender"],
        agent_transcription=agent_info["transcription"],
        agent_tone=agent_info["tone"]
    )

    result_text = call_qwen_api(prompt)
    reason, score = parse_evaluation_result(result_text)

    return {
        "model": agent_info["model"],
        "user_info": user_info,
        "agent_info": agent_info,
        "raw_response": result_text,
        "reason": reason,
        "score": score
    }


def process_json_file(file_path):
    """处理单个JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    category = data["category"]
    label = data["label"]
    results = []

    print(f"\n处理文件: {file_path.name}")
    print(f"类别: {category}, 标签: {label}")
    print(f"共 {len(data['pairs'])} 个对话对")

    for i, pair in enumerate(data["pairs"]):
        print(f"  评估对话对 {i+1}/{len(data['pairs'])} (model: {pair['output']['model']})...", end=" ")

        result = evaluate_pair(pair, category, label)
        results.append(result)

        if result["score"]:
            print(f"分数: {result['score']}")
        else:
            print(f"解析失败: {result['raw_response'][:50]}...")

    return {
        "file": file_path.name,
        "category": category,
        "label": label,
        "evaluation_results": results
    }


def main():
    """主函数"""
    base_dir = Path("/home/u2023112559/qix/Project/Final_Project/Audio_Captior/evaluation_results")
    output_dir = base_dir / "evaluated"
    output_dir.mkdir(exist_ok=True)

    # 收集所有JSON文件
    all_json_files = list(base_dir.glob("*/*.json"))
    print(f"找到 {len(all_json_files)} 个JSON文件")

    all_results = []
    summary_stats = {}

    for file_path in all_json_files:
        result = process_json_file(file_path)
        all_results.append(result)

        # 保存单个文件结果
        output_file = output_dir / f"evaluated_{file_path.name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"  已保存: {output_file.name}")

        # 统计
        category = result["category"]
        if category not in summary_stats:
            summary_stats[category] = {}

        for eval_result in result["evaluation_results"]:
            model = eval_result["model"]
            score = eval_result["score"]

            if model not in summary_stats[category]:
                summary_stats[category][model] = {"scores": [], "count": 0}

            if score is not None:
                summary_stats[category][model]["scores"].append(score)
                summary_stats[category][model]["count"] += 1

    # 生成汇总报告
    summary_report = {
        "evaluation_time": datetime.now().isoformat(),
        "total_files": len(all_json_files),
        "statistics": {}
    }

    print("\n" + "="*60)
    print("评估汇总")
    print("="*60)

    for category, models in summary_stats.items():
        print(f"\n{category.upper()}:")
        summary_report["statistics"][category] = {}

        for model, stats in models.items():
            if stats["scores"]:
                avg_score = sum(stats["scores"]) / len(stats["scores"])
                summary_report["statistics"][category][model] = {
                    "average_score": round(avg_score, 2),
                    "total_evaluations": stats["count"],
                    "score_distribution": {
                        str(i): stats["scores"].count(i) for i in range(1, 6)
                    }
                }
                print(f"  {model}: 平均分 {avg_score:.2f} (共 {stats['count']} 次评估)")

    # 保存汇总报告
    summary_file = output_dir / "summary_report.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    print(f"\n汇总报告已保存: {summary_file}")

    return all_results, summary_report


if __name__ == "__main__":
    main()
