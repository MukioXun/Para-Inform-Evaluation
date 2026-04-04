#!/usr/bin/env python3
"""
筛选两种评测方式分差>=3的样本
Filter samples where |Omni_score - Cascade_score| >= 3
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# Paths
BASE_DIR = Path(__file__).parent
CASCADE_DIR = BASE_DIR / "Cascade_evaluated" / "jsons"
OMNI_FILE = BASE_DIR / "qwen_eval_results" / "detailed_results.json"
OUTPUT_DIR = BASE_DIR / "comparison"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_cascade_results():
    """Load all Cascade evaluation results"""
    results = {}  # {dir_name: {model_name: {score, reason, category, label, user_input, agent_output}}}

    for json_file in CASCADE_DIR.glob("evaluated_*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dir_name = data["file"].replace(".json", "")
        category = data["category"]
        label = data["label"]

        for eval_result in data["evaluation_results"]:
            model = eval_result["model"]
            score = eval_result["score"]
            reason = eval_result.get("reason", "")

            # Extract user input and agent output
            user_input = eval_result.get("user_info", {}).get("transcription", "N/A")
            agent_output = eval_result.get("agent_info", {}).get("transcription", "N/A")

            if dir_name not in results:
                results[dir_name] = {}

            results[dir_name][model] = {
                "score": score,
                "reason": reason,
                "category": category,
                "label": label,
                "user_input": user_input,
                "agent_output": agent_output
            }

    return results


def load_omni_results():
    """Load Omni evaluation results"""
    results = {}  # {dir_name: {model_name: {score, reason, category, label}}}

    with open(OMNI_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        dir_name = item["dir_name"]
        model = item["model_name"]
        score = item["score"]
        reason = item.get("reason", "")
        category = item["category"]
        label = item["label"]

        if dir_name not in results:
            results[dir_name] = {}

        results[dir_name][model] = {
            "score": score,
            "reason": reason,
            "category": category,
            "label": label
        }

    return results


def find_large_deviations(cascade_results, omni_results, threshold=3):
    """Find samples where |Omni - Cascade| >= threshold"""

    deviations_by_category = defaultdict(list)

    for dir_name in cascade_results:
        if dir_name not in omni_results:
            continue

        for model in cascade_results[dir_name]:
            if model not in omni_results[dir_name]:
                continue

            cascade = cascade_results[dir_name][model]
            omni = omni_results[dir_name][model]

            cascade_score = cascade["score"]
            omni_score = omni["score"]
            deviation = omni_score - cascade_score

            if abs(deviation) >= threshold:
                category = cascade["category"]
                label = cascade["label"]

                deviations_by_category[category].append({
                    "dir_name": dir_name,
                    "model": model,
                    "label": label,
                    "cascade_score": cascade_score,
                    "omni_score": omni_score,
                    "deviation": deviation,
                    "cascade_reason": cascade["reason"],
                    "omni_reason": omni["reason"],
                    "user_input": cascade.get("user_input", "N/A"),
                    "agent_output": cascade.get("agent_output", "N/A")
                })

    return deviations_by_category


def generate_report(deviations_by_category):
    """Generate JSON and Markdown reports"""

    # Sort by absolute deviation within each category
    for cat in deviations_by_category:
        deviations_by_category[cat].sort(key=lambda x: abs(x["deviation"]), reverse=True)

    # Summary statistics
    total_count = sum(len(v) for v in deviations_by_category.values())
    summary = {
        "meta": {
            "title": "Large Deviation Samples (|Omni - Cascade| >= 3)",
            "threshold": 3,
            "total_samples": total_count
        },
        "statistics": {},
        "samples_by_category": {}
    }

    for cat, samples in deviations_by_category.items():
        positive_dev = [s for s in samples if s["deviation"] > 0]  # Omni > Cascade
        negative_dev = [s for s in samples if s["deviation"] < 0]  # Cascade > Omni

        summary["statistics"][cat] = {
            "count": len(samples),
            "omni_higher": len(positive_dev),
            "cascade_higher": len(negative_dev),
            "avg_deviation": sum(s["deviation"] for s in samples) / len(samples) if samples else 0,
            "max_deviation": max(s["deviation"] for s in samples) if samples else 0,
            "min_deviation": min(s["deviation"] for s in samples) if samples else 0
        }

        summary["samples_by_category"][cat] = samples

    # Save JSON report
    json_path = OUTPUT_DIR / "large_deviation_samples.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"JSON report saved: {json_path}")

    # Generate Markdown report
    md_lines = []
    md_lines.append("# Large Deviation Samples Report")
    md_lines.append("")
    md_lines.append(f"**Threshold**: |Omni - Cascade| >= 3")
    md_lines.append(f"**Total samples found**: {total_count}")
    md_lines.append("")

    # Statistics table
    md_lines.append("## Summary Statistics")
    md_lines.append("")
    md_lines.append("| Category | Count | Omni Higher | Cascade Higher | Avg Deviation |")
    md_lines.append("|----------|-------|-------------|----------------|---------------|")
    for cat in ["age", "emotion", "gender", "sarcasm"]:
        if cat in summary["statistics"]:
            s = summary["statistics"][cat]
            md_lines.append(f"| {cat} | {s['count']} | {s['omni_higher']} | {s['cascade_higher']} | {s['avg_deviation']:+.2f} |")
    md_lines.append("")

    # Detailed samples by category
    for cat in ["age", "emotion", "gender", "sarcasm"]:
        if cat not in deviations_by_category:
            continue

        samples = deviations_by_category[cat]
        md_lines.append(f"## {cat.upper()} ({len(samples)} samples)")
        md_lines.append("")

        for i, s in enumerate(samples, 1):
            md_lines.append(f"### {i}. {s['dir_name']} ({s['model']})")
            md_lines.append("")
            md_lines.append(f"- **Label**: {s['label']}")
            md_lines.append(f"- **Cascade Score**: {s['cascade_score']}")
            md_lines.append(f"- **Omni Score**: {s['omni_score']}")
            md_lines.append(f"- **Deviation**: {s['deviation']:+d} (Omni {'>' if s['deviation'] > 0 else '<'} Cascade)")
            md_lines.append("")
            md_lines.append(f"**User Input**: {s['user_input']}")
            md_lines.append("")
            md_lines.append(f"**Agent Output**: {s['agent_output']}")
            md_lines.append("")
            md_lines.append(f"**Cascade Reason**: {s['cascade_reason'][:200]}..." if len(s['cascade_reason']) > 200 else f"**Cascade Reason**: {s['cascade_reason']}")
            md_lines.append("")
            md_lines.append(f"**Omni Reason**: {s['omni_reason'][:200]}..." if len(s['omni_reason']) > 200 else f"**Omni Reason**: {s['omni_reason']}")
            md_lines.append("")
            md_lines.append("---")
            md_lines.append("")

    md_path = OUTPUT_DIR / "large_deviation_samples.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines))
    print(f"Markdown report saved: {md_path}")

    return summary


def print_summary(deviations_by_category):
    """Print summary to console"""
    print("\n" + "="*60)
    print("LARGE DEVIATION SAMPLES (|Omni - Cascade| >= 3)")
    print("="*60)

    total = 0
    for cat in ["age", "emotion", "gender", "sarcasm"]:
        if cat in deviations_by_category:
            count = len(deviations_by_category[cat])
            total += count
            positive = sum(1 for s in deviations_by_category[cat] if s["deviation"] > 0)
            negative = count - positive
            avg_dev = sum(s["deviation"] for s in deviations_by_category[cat]) / count if count > 0 else 0
            print(f"\n[{cat.upper()}] {count} samples (Omni higher: {positive}, Cascade higher: {negative}, avg: {avg_dev:+.2f})")

            # Show top 3
            for s in deviations_by_category[cat][:3]:
                direction = "↑" if s["deviation"] > 0 else "↓"
                print(f"  {direction} {s['dir_name']} ({s['model']}): Cascade={s['cascade_score']}, Omni={s['omni_score']}, Diff={s['deviation']:+d}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total} samples with deviation >= 3")
    print("="*60)


def main():
    print("Loading Cascade evaluation results...")
    cascade_results = load_cascade_results()
    print(f"  Loaded {len(cascade_results)} directories")

    print("Loading Omni evaluation results...")
    omni_results = load_omni_results()
    print(f"  Loaded {len(omni_results)} directories")

    print("\nFinding large deviations (>= 3 points)...")
    deviations = find_large_deviations(cascade_results, omni_results, threshold=3)

    print_summary(deviations)
    summary = generate_report(deviations)

    return summary


if __name__ == "__main__":
    main()
