"""
人工标注平台 - Flask 应用
用于标注音频数据的属性识别结果
"""
import os
import json
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path

# ================= 配置 =================
BASE_DIR = Path("/home/u2023112559/qix/Project/Final_Project/Audio_Captior")
AUDIO_DIR = BASE_DIR / "audio"
EVALUATION_DIR = BASE_DIR / "evaluation_results"
ANNOTATION_DIR = BASE_DIR / "human_annotations"  # 人工标注结果保存目录

# 确保标注目录存在
ANNOTATION_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# ================= 路由 =================

@app.route('/')
def index():
    """主页 - 选择类别"""
    categories = [d for d in os.listdir(AUDIO_DIR)
                  if (AUDIO_DIR / d).is_dir()]
    return render_template('index.html', categories=sorted(categories))


@app.route('/category/<category>')
def category_view(category):
    """类别页面 - 显示该类别下所有音频目录"""
    cat_dir = AUDIO_DIR / category
    if not cat_dir.exists():
        return "Category not found", 404

    # 获取所有子目录
    dirs = [d for d in os.listdir(cat_dir)
            if (cat_dir / d).is_dir()]

    # 检查标注状态
    annotation_status = {}
    for d in dirs:
        annotation_file = ANNOTATION_DIR / category / f"{d}.json"
        annotation_status[d] = annotation_file.exists()

    return render_template('category.html',
                          category=category,
                          dirs=sorted(dirs),
                          annotation_status=annotation_status)


@app.route('/annotate/<category>/<dir_name>')
def annotate_view(category, dir_name):
    """标注页面 - 显示音频和标注表单"""
    # 读取评估结果
    eval_file = EVALUATION_DIR / category / f"{dir_name}.json"
    if not eval_file.exists():
        return "Evaluation file not found", 404

    with open(eval_file, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    # 读取已有的标注（如果存在）
    annotation_file = ANNOTATION_DIR / category / f"{dir_name}.json"
    existing_annotation = None
    if annotation_file.exists():
        with open(annotation_file, 'r', encoding='utf-8') as f:
            existing_annotation = json.load(f)

    return render_template('annotate.html',
                          category=category,
                          dir_name=dir_name,
                          eval_data=eval_data,
                          existing_annotation=existing_annotation)


@app.route('/audio/<category>/<dir_name>/<filename>')
def serve_audio(category, dir_name, filename):
    """提供音频文件"""
    audio_path = AUDIO_DIR / category / dir_name / filename
    if not audio_path.exists():
        return "Audio not found", 404
    return send_from_directory(AUDIO_DIR / category / dir_name, filename)


@app.route('/api/submit', methods=['POST'])
def submit_annotation():
    """提交标注结果"""
    data = request.json
    category = data.get('category')
    dir_name = data.get('dir_name')
    annotations = data.get('annotations')

    if not all([category, dir_name, annotations]):
        return jsonify({"error": "Missing required fields"}), 400

    # 保存标注结果
    save_dir = ANNOTATION_DIR / category
    save_dir.mkdir(exist_ok=True)

    save_file = save_dir / f"{dir_name}.json"

    result = {
        "category": category,
        "dir_name": dir_name,
        "ground_truth": data.get('ground_truth'),
        "annotations": annotations,
        "timestamp": data.get('timestamp')
    }

    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return jsonify({"success": True, "saved_to": str(save_file)})


@app.route('/api/stats')
def get_stats():
    """获取标注统计"""
    stats = {}

    for category in os.listdir(AUDIO_DIR):
        if not (AUDIO_DIR / category).is_dir():
            continue

        total = len([d for d in os.listdir(AUDIO_DIR / category)
                    if (AUDIO_DIR / category / d).is_dir()])

        annotated = 0
        ann_dir = ANNOTATION_DIR / category
        if ann_dir.exists():
            annotated = len([f for f in os.listdir(ann_dir)
                           if f.endswith('.json')])

        stats[category] = {
            "total": total,
            "annotated": annotated,
            "remaining": total - annotated
        }

    return jsonify(stats)


# ================= 模板 =================

# 创建模板目录
os.makedirs('/home/u2023112559/qix/Project/Final_Project/Audio_Captior/annotation_platform/templates', exist_ok=True)

# 创建静态文件目录
os.makedirs('/home/u2023112559/qix/Project/Final_Project/Audio_Captior/annotation_platform/static', exist_ok=True)

if __name__ == '__main__':
    print("=" * 60)
    print("Audio Annotation Platform")
    print("=" * 60)
    print(f"Audio Directory: {AUDIO_DIR}")
    print(f"Evaluation Results: {EVALUATION_DIR}")
    print(f"Annotations Save: {ANNOTATION_DIR}")
    print("=" * 60)
    print("Starting server at http://localhost:4602")
    print("=" * 60)
    app.run(host='0.0.0.0', port=4602, debug=True)
