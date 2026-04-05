# 音频人工标注平台

用于人工标注音频属性识别结果的 Web 平台。

## 功能特性

- 🎧 **音频播放**: 在线播放用户音频和模型响应音频
- 📝 **转录文本**: 显示 ASR 转录结果
- 🤖 **模型预测**: 展示情感、年龄、性别等属性识别结果
- ✏️ **人工标注**: 验证模型预测正确性，评估响应质量
- 📊 **进度追踪**: 实时显示标注进度

## 快速开始

### 1. 启动服务器

```bash
cd /home/u2023112559/qix/Project/Final_Project/Audio_Captior/annotation_platform
./start.sh
```

或直接运行:

```bash
python app.py
```

### 2. 访问平台

打开浏览器访问: **http://localhost:4601**

### 3. 标注流程

1. **选择类别**: 在首页点击要标注的类别 (age/emotion/gender/sarcasm)
2. **选择数据**: 点击待标注的数据项
3. **播放音频**: 收听用户音频和各模型的响应音频
4. **查看预测**: 查看 ASR 转录和属性识别结果
5. **填写标注**: 验证模型预测是否正确
6. **保存提交**: 点击"保存标注"按钮

## 数据统计

| 类别 | 数据量 |
|------|--------|
| age | 10 |
| emotion | 20 |
| gender | 10 |
| sarcasm | 8 |
| **总计** | **48** |

## 目录结构

```
annotation_platform/
├── app.py              # Flask 应用主程序
├── start.sh            # 启动脚本
├── templates/
│   ├── index.html      # 首页
│   ├── category.html   # 类别页面
│   └── annotate.html   # 标注页面
└── static/             # 静态资源
```

## 标注结果

人工标注结果保存在:
```
/home/u2023112559/qix/Project/Final_Project/Audio_Captior/human_annotations/
```

每个标注文件包含:
```json
{
  "category": "age",
  "dir_name": "04-18-05-07_0_adult",
  "ground_truth": "adult",
  "annotations": {
    "user_emotion_score": 5,
    "user_age_score": 4,
    "user_gender_score": 5,
    "best_model": "gpt-4o-voice-mode",
    "response_emotion_fit_score": 4,
    "response_quality_score": 4,
    "model_glm4_score": 3,
    "model_gpt-4o-voice-mode_score": 5,
    "model_llamaomni2_score": 3,
    "model_original_score": 4,
    "model_qwen2_5_score": 3,
    "model_rl_real_all_score": 4,
    "notes": ""
  },
  "timestamp": "2026-04-05T16:30:00.000Z"
}
```

## 标注指南

### 评分说明 (1-5分制)

| 分数 | 含义 | 说明 |
|------|------|------|
| 1分 | 很差/完全错误 | 完全不符合实际情况 |
| 2分 | 较差 | 大部分不符合实际情况 |
| 3分 | 一般/部分正确 | 有一定合理性但存在明显问题 |
| 4分 | 较好 | 基本正确，有小瑕疵 |
| 5分 | 很好/完全正确 | 完全符合预期 |

### 用户属性识别评分

- **情感识别准确度** (1-5分): 模型对用户情感的识别是否准确
- **年龄识别准确度** (1-5分): 模型对用户年龄段的识别是否准确
- **性别识别准确度** (1-5分): 模型对用户性别的识别是否准确

### 响应质量评分

- **最佳响应模型**: 选择响应最自然、最贴合用户意图的模型
- **响应情感适配度** (1-5分): 响应情感是否与用户情感状态匹配
- **整体响应质量** (1-5分): 综合考虑响应的自然度、相关性和有用性

### 各模型单独评分

为每个模型的响应单独打分 (1-5分)，便于后续对比分析

## 注意事项

1. 建议使用耳机进行音频标注，以获得更好的音质
2. 如遇网络代理问题，可在终端设置: `export no_proxy="localhost,127.0.0.1"`
3. 服务器默认运行在开发模式，适合小规模标注使用

## API 接口

### 获取统计信息
```
GET /api/stats
```

返回:
```json
{
  "age": { "total": 10, "annotated": 5, "remaining": 5 },
  ...
}
```

### 提交标注
```
POST /api/submit
Content-Type: application/json

{
  "category": "age",
  "dir_name": "xxx",
  "ground_truth": "adult",
  "annotations": { ... }
}
```
