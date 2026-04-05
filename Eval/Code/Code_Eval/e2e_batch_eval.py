#!/usr/bin/env python3
"""
Qwen Omni 音频对话质量评估脚本
批量评估 evaluation_results 中的音频配对，输出 JSON 格式结果

功能：
1. 遍历 evaluation_results 目录
2. 合并 user + model 音频，调用 Qwen Omni API 评估
3. 输出 JSON 格式结果
4. 统计各模型在各维度的表现

使用方法：
    python batch_eval.py --input ./evaluation_results --audio ./audio --output ./qwen_eval_results
"""
import os
import sys
import json
import base64
import tempfile
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI
from pydub import AudioSegment

# Prompt 定义
PROMPT_API = """
You are a dialogue evaluation expert.
You are given:
- User speech and Model response are merged into one audio, and the reference audio is the user speech. The merged audio contains the user speech followed by the model response, with a short silence in between.
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
The reason is <very brief reason>; The score is <1-5>.
"""


# ================= 数据结构 =================

@dataclass
class EvalResult:
    """单次评估结果"""
    category: str
    label: str
    dir_name: str
    model_name: str
    audio_file: str
    score: int
    reason: str
    raw_response: str


@dataclass
class ModelStats:
    """模型统计"""
    model_name: str
    total_count: int
    score_sum: int
    scores: List[int]

    # 分类统计
    category_scores: Dict[str, List[int]]

    def add_score(self, score: int, category: str):
        self.total_count += 1
        self.score_sum += score
        self.scores.append(score)
        if category not in self.category_scores:
            self.category_scores[category] = []
        self.category_scores[category].append(score)

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.total_count if self.total_count > 0 else 0.0


# ================= 核心函数 =================

class QwenEvaluator:
    """Qwen Omni 评估器"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def get_audio_base64(self, file_path: str) -> str:
        """将音频文件转换为 Base64"""
        with open(file_path, "rb") as f:
            return f"data:audio/wav;base64,{base64.b64encode(f.read()).decode('utf-8')}"

    def merge_audio(self, audio1_path: str, audio2_path: str, silence_ms: int = 500) -> str:
        """合并两段音频，添加静音分隔"""
        aud1 = AudioSegment.from_wav(audio1_path)
        aud2 = AudioSegment.from_wav(audio2_path)
        silence = AudioSegment.silent(duration=silence_ms)
        combined = aud1 + silence + aud2

        temp_path = tempfile.mktemp(suffix=".wav")
        combined.export(temp_path, format="wav")
        return temp_path

    def evaluate(self, user_audio: str, model_audio: str) -> Tuple[int, str, str]:
        """
        评估单个音频配对

        Returns:
            (score, reason, raw_response)
        """
        temp_path = None
        try:
            # 合并音频
            temp_path = self.merge_audio(user_audio, model_audio)
            audio_data = self.get_audio_base64(temp_path)

            # 调用 API
            completion = self.client.chat.completions.create(
                model="qwen3-omni-flash-2025-12-01",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": audio_data, "format": "wav"}},
                        {"type": "text", "text": PROMPT_API}
                    ]
                }],
                modalities=["text"],
                stream=False,
            )

            raw_response = completion.choices[0].message.content
            reason, score = self._parse_response(raw_response)
            return score, reason, raw_response

        except Exception as e:
            return 0, f"Error: {str(e)}", ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def _parse_response(self, result_text: str) -> Tuple[str, int]:
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


def scan_evaluation_results(input_dir: Path, audio_dir: Path) -> List[Dict[str, Any]]:
    """
    扫描评测结果，提取需要评估的音频配对

    Returns:
        [{
            "result_file": Path,
            "category": str,
            "label": str,
            "dir_name": str,
            "user_audio": str,
            "model_audio": str,
            "model_name": str
        }]
    """
    tasks = []

    for category_dir in input_dir.iterdir():
        if not category_dir.is_dir():
            continue

        for result_file in category_dir.glob("*.json"):
            if result_file.name == "summary.json":
                continue

            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[Warning] Failed to read {result_file}: {e}")
                continue

            category = data.get("category", category_dir.name)
            label = data.get("label", "")
            dir_name = data.get("dir_name", result_file.stem)

            # 音频目录
            audio_subdir = audio_dir / category / dir_name

            # 处理每个配对
            pairs = data.get("pairs", [])
            for pair in pairs:
                input_info = pair.get("input", {})
                output_info = pair.get("output", {})

                user_file = input_info.get("file", "")
                model_file = output_info.get("file", "")
                model_name = output_info.get("model", "unknown")

                if user_file and model_file:
                    user_path = audio_subdir / user_file
                    model_path = audio_subdir / model_file

                    if user_path.exists() and model_path.exists():
                        tasks.append({
                            "result_file": result_file,
                            "category": category,
                            "label": label,
                            "dir_name": dir_name,
                            "user_audio": str(user_path),
                            "model_audio": str(model_path),
                            "model_name": model_name
                        })

    return tasks


def process_task(task: Dict, evaluator: QwenEvaluator) -> EvalResult:
    """处理单个评估任务"""
    score, reason, raw_response = evaluator.evaluate(
        task["user_audio"],
        task["model_audio"]
    )

    return EvalResult(
        category=task["category"],
        label=task["label"],
        dir_name=task["dir_name"],
        model_name=task["model_name"],
        audio_file=task["model_audio"],
        score=score,
        reason=reason,
        raw_response=raw_response
    )


# ================= 主流程 =================

def run_batch_evaluation(
    input_dir: str,
    audio_dir: str,
    output_dir: str,
    limit: int = 0
):
    """运行批量评估"""
    input_path = Path(input_dir)
    audio_path = Path(audio_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen Omni Batch Evaluation")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Audio:  {audio_path}")
    print(f"Output: {output_path}")
    print()

    # 1. 扫描任务
    print("[Step 1] Scanning evaluation results...")
    tasks = scan_evaluation_results(input_path, audio_path)
    print(f"Found {len(tasks)} audio pairs to evaluate")

    if limit > 0:
        tasks = tasks[:limit]
        print(f"Limited to {len(tasks)} tasks")

    # 2. 初始化评估器
    print("\n[Step 2] Initializing Qwen Omni evaluator...")
    evaluator = QwenEvaluator()

    # 3. 批量评估
    print("\n[Step 3] Running evaluation...")
    results: List[EvalResult] = []
    model_stats: Dict[str, ModelStats] = {}

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] {task['model_name']} @ {task['dir_name']}...", end=" ")

        result = process_task(task, evaluator)
        results.append(result)

        # 更新统计
        model_name = result.model_name
        if model_name not in model_stats:
            model_stats[model_name] = ModelStats(
                model_name=model_name,
                total_count=0,
                score_sum=0,
                scores=[],
                category_scores={}
            )
        model_stats[model_name].add_score(result.score, result.category)

        print(f"score={result.score}")

    # 4. 保存详细结果
    print("\n[Step 4] Saving results...")

    # 详细结果
    results_file = output_path / "detailed_results.json"
    results_data = [asdict(r) for r in results]
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {results_file}")

    # 5. 统计汇总
    print("\n[Step 5] Computing statistics...")

    # 模型统计
    model_summary = {}
    for model_name, stats in model_stats.items():
        model_summary[model_name] = {
            "total_count": stats.total_count,
            "avg_score": round(stats.avg_score, 3),
            "score_distribution": {
                "1": stats.scores.count(1),
                "2": stats.scores.count(2),
                "3": stats.scores.count(3),
                "4": stats.scores.count(4),
                "5": stats.scores.count(5),
            },
            "category_avg": {
                cat: round(sum(scores) / len(scores), 3) if scores else 0
                for cat, scores in stats.category_scores.items()
            }
        }

    # 整体统计
    total_evaluations = len(results)
    all_scores = [r.score for r in results if r.score > 0]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0

    summary = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": total_evaluations,
            "total_models": len(model_stats),
            "overall_avg_score": round(overall_avg, 3)
        },
        "model_statistics": model_summary,
        "category_statistics": compute_category_stats(results)
    }

    summary_file = output_path / "evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {summary_file}")

    # 6. 打印统计
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nOverall Statistics:")
    print(f"  Total evaluations: {total_evaluations}")
    print(f"  Overall avg score: {overall_avg:.3f}")

    print(f"\nModel Rankings (by avg score):")
    sorted_models = sorted(model_summary.items(), key=lambda x: x[1]['avg_score'], reverse=True)
    for i, (model, stats) in enumerate(sorted_models, 1):
        print(f"  {i}. {model}: {stats['avg_score']:.3f} ({stats['total_count']} samples)")

    print(f"\nCategory Statistics:")
    cat_stats = summary["category_statistics"]
    for cat, stats in cat_stats.items():
        print(f"  {cat}: avg={stats['avg_score']:.3f}, count={stats['count']}")


def compute_category_stats(results: List[EvalResult]) -> Dict[str, Dict]:
    """计算各分类的统计"""
    cat_data: Dict[str, List[int]] = {}

    for r in results:
        if r.score > 0:
            if r.category not in cat_data:
                cat_data[r.category] = []
            cat_data[r.category].append(r.score)

    return {
        cat: {
            "count": len(scores),
            "avg_score": round(sum(scores) / len(scores), 3) if scores else 0,
            "score_distribution": {
                "1": scores.count(1),
                "2": scores.count(2),
                "3": scores.count(3),
                "4": scores.count(4),
                "5": scores.count(5),
            }
        }
        for cat, scores in cat_data.items()
    }


# ================= 入口 =================

def main():
    parser = argparse.ArgumentParser(description="Qwen Omni Batch Evaluation")

    parser.add_argument(
        "--input", "-i",
        default="/home/u2023112559/qix/Project/Final_Project/Audio_Captior/evaluation_results",
        help="Input evaluation results directory"
    )
    parser.add_argument(
        "--audio", "-a",
        default="/home/u2023112559/qix/Project/Final_Project/Audio_Captior/audio",
        help="Audio files directory"
    )
    parser.add_argument(
        "--output", "-o",
        default="/home/u2023112559/qix/Project/Final_Project/Audio_Captior/Eval/qwen_eval_results",
        help="Output directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of tasks (0=unlimited)"
    )

    args = parser.parse_args()

    run_batch_evaluation(
        input_dir=args.input,
        audio_dir=args.audio,
        output_dir=args.output,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
