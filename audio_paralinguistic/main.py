#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
语音副语言感知评估系统 - 主入口

用法示例:
    # 单音频标注
    python main.py annotate --input audio.wav --output result.json

    # 批量处理
    python main.py batch --input_dir ./audio/ --output ./results/annotations.jsonl

    # 聚类分析
    python main.py cluster --input ./results/annotations.jsonl --output ./output/

    # 完整流程
    python main.py pipeline --input_dir ./audio/ --output_dir ./output/
"""
import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig, FeatureConfig
from core.pipeline import Pipeline
from fusion.feature_merger import FeatureMerger
from clustering.cluster_engine import ClusterEngine
from clustering.dimension_reduction import DimensionReducer
from clustering.cluster_evaluator import ClusterEvaluator
from evaluation.schema_inducer import SchemaInducer
from evaluation.report_generator import ReportGenerator
from api.qwen_client import QwenClient


def create_parser():
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        description="语音副语言感知评估系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # ===== annotate 模式 =====
    annotate_parser = subparsers.add_parser("annotate", help="单音频标注")
    annotate_parser.add_argument("--input", required=True, help="输入音频文件")
    annotate_parser.add_argument("--output", required=True, help="输出结果文件")
    annotate_parser.add_argument("--annotators", nargs='+',
                                 default=["asr", "emotion", "speaker"],
                                 help="使用的标注器")

    # ===== batch 模式 =====
    batch_parser = subparsers.add_parser("batch", help="批量标注")
    batch_parser.add_argument("--input_dir", required=True, help="输入音频目录")
    batch_parser.add_argument("--output", required=True, help="输出JSONL文件")
    batch_parser.add_argument("--num_workers", type=int, default=4, help="并行线程数")
    batch_parser.add_argument("--annotators", nargs='+',
                             default=["asr", "emotion", "speaker"],
                             help="使用的标注器")

    # ===== cluster 模式 =====
    cluster_parser = subparsers.add_parser("cluster", help="聚类分析")
    cluster_parser.add_argument("--input", required=True, help="标注结果JSONL文件")
    cluster_parser.add_argument("--output_dir", required=True, help="输出目录")
    cluster_parser.add_argument("--method", default="hdbscan",
                               choices=["kmeans", "dbscan", "hdbscan", "agglomerative"],
                               help="聚类方法")
    cluster_parser.add_argument("--n_clusters", type=int, default=None,
                               help="聚类数量（kmeans/agglomerative需要）")
    cluster_parser.add_argument("--dim_reduction", default="umap",
                               choices=["pca", "tsne", "umap"],
                               help="降维方法")
    cluster_parser.add_argument("--use_qwen", action="store_true",
                               help="使用千帆大模型辅助分析")

    # ===== pipeline 模式 =====
    pipeline_parser = subparsers.add_parser("pipeline", help="完整流程")
    pipeline_parser.add_argument("--input_dir", required=True, help="输入音频目录")
    pipeline_parser.add_argument("--output_dir", required=True, help="输出目录")
    pipeline_parser.add_argument("--num_workers", type=int, default=4)
    pipeline_parser.add_argument("--cluster_method", default="hdbscan")
    pipeline_parser.add_argument("--use_qwen", action="store_true")

    return parser


def run_annotate(args):
    """运行单音频标注"""
    print(f"[Annotate] Processing: {args.input}")

    pipeline = Pipeline(num_workers=1)

    # TODO: 注册标注器
    # from annotators.semantic.asr_annotator import ASRAnnotator
    # pipeline.register_annotator("asr", ASRAnnotator())

    result = pipeline.process_single(args.input, args.output)
    print(f"[Done] Result saved to: {args.output}")


def run_batch(args):
    """运行批量标注"""
    print(f"[Batch] Processing directory: {args.input_dir}")

    pipeline = Pipeline(num_workers=args.num_workers)

    # TODO: 注册标注器

    results = pipeline.process_batch(args.input_dir, args.output)
    print(f"[Done] Processed {len(results)} files")


def run_cluster(args):
    """运行聚类分析"""
    print(f"[Cluster] Analyzing: {args.input}")

    from utils.json_utils import load_jsonl

    # 加载标注结果
    annotations = load_jsonl(args.input)
    print(f"Loaded {len(annotations)} annotations")

    # 特征融合
    merger = FeatureMerger()
    merged_features = merger.merge_batch(annotations)
    feature_matrix = FeatureMerger.to_feature_matrix(merged_features)

    print(f"Feature matrix shape: {feature_matrix.shape}")

    # 标准化
    from fusion.normalization import Normalizer
    normalizer = Normalizer(method="z-score")
    feature_matrix = normalizer.fit_transform(feature_matrix)

    # 降维
    reducer = DimensionReducer(method=args.dim_reduction, n_components=2)
    embedding = reducer.fit_transform(feature_matrix)

    # 聚类
    cluster_params = {}
    if args.method in ["kmeans", "agglomerative"] and args.n_clusters:
        cluster_params["n_clusters"] = args.n_clusters

    engine = ClusterEngine(method=args.method, **cluster_params)
    labels = engine.fit_predict(feature_matrix)
    result = engine.get_result()

    print(f"Found {result.n_clusters} clusters")

    # 评估
    evaluator = ClusterEvaluator()
    eval_result = evaluator.evaluate(feature_matrix, labels)
    print(f"Silhouette score: {eval_result.silhouette_score:.4f}")

    # 标注体系推导
    feature_names = list(merged_features[0].feature_dict.keys()) if merged_features else []
    sample_ids = [mf.audio_id for mf in merged_features]

    inducer = SchemaInducer(use_llm=args.use_qwen)
    if args.use_qwen:
        inducer.llm_client = QwenClient()

    schema = inducer.induce(feature_matrix, labels, feature_names, sample_ids)

    # 生成报告
    os.makedirs(args.output_dir, exist_ok=True)
    report_gen = ReportGenerator(output_dir=args.output_dir)
    report_path = report_gen.generate(schema, result, eval_result)

    # 保存可视化
    viz_path = os.path.join(args.output_dir, "clustering_visualization.png")
    report_gen.generate_visualization_report(embedding, labels, viz_path)

    # 保存标注体系
    schema_path = os.path.join(args.output_dir, "annotation_schema.json")
    inducer.save(schema, schema_path)

    print(f"[Done] Results saved to: {args.output_dir}")


def run_pipeline(args):
    """运行完整流程"""
    print(f"[Pipeline] Starting full pipeline")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: 批量标注
    annotations_path = os.path.join(args.output_dir, "annotations.jsonl")
    print("\n[Step 1] Running batch annotation...")

    batch_args = argparse.Namespace(
        input_dir=args.input_dir,
        output=annotations_path,
        num_workers=args.num_workers,
        annotators=["asr", "emotion", "speaker"]
    )
    # run_batch(batch_args)  # TODO: 启用后取消注释
    print(f"  Annotations saved to: {annotations_path}")

    # Step 2: 聚类分析
    print("\n[Step 2] Running cluster analysis...")
    cluster_output = os.path.join(args.output_dir, "cluster_analysis")

    cluster_args = argparse.Namespace(
        input=annotations_path,
        output_dir=cluster_output,
        method=args.cluster_method,
        use_qwen=args.use_qwen
    )
    # run_cluster(cluster_args)  # TODO: 启用后取消注释
    print(f"  Cluster results saved to: {cluster_output}")

    print(f"\n[Done] Pipeline completed. Results in: {args.output_dir}")


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    print("=" * 60)
    print("语音副语言感知评估系统")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    if args.mode == "annotate":
        run_annotate(args)
    elif args.mode == "batch":
        run_batch(args)
    elif args.mode == "cluster":
        run_cluster(args)
    elif args.mode == "pipeline":
        run_pipeline(args)


if __name__ == "__main__":
    main()
