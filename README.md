# Wav2Vec2 + ResNet 语音情感识别

> **免责声明**: 本项目仅作为课程作业提交，并非本人原创研发的项目。项目中的模型架构、训练策略及代码实现均基于课程要求完成，仅供学习交流使用。

基于 Wav2Vec2 预训练模型和 1D ResNet 的语音情感识别系统。

## 特性

- **Wav2Vec2 预训练模型**: 使用 Facebook 预训练的 wav2vec2-base 作为特征提取器
- **1D ResNet**: 4个残差块用于语音特征处理
- **微调策略**: 解冻最后4层编码器进行微调
- **数据增强**: 噪声注入、时间偏移、音量扰动
- **早停机制**: 防止过拟合
- **可视化工具**: 混淆矩阵、训练曲线、分类报告

## 项目结构

```
Speech emotion recognition/
├── train_resnet_wav2vec_v2.py     # 训练脚本
├── generate_resnet_charts_simple.py # 图表生成脚本
├── requirements.txt               # 依赖包
├── configs/                       # 配置文件
├── dataset/                       # 数据集列表
│   ├── train_list.txt
│   └── test_list.txt
├── database/                      # 原始数据(RAVDESS)
├── output_resnet_wav2vec/         # 训练输出
│   ├── best_model/               # 最佳模型权重
│   ├── images/                  # 评估图表
│   ├── results.json              # 评估结果
│   └── training_history.json     # 训练历史
└── models/                       # 模型定义
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train_resnet_wav2vec_v2.py --epoch 35 --batch_size 16 --lr 0.00015
```

### 3. 生成评估图表

```bash
python generate_resnet_charts_simple.py
```

## 模型架构

```
Wav2Vec2 (facebook/wav2vec2-base)
    ↓
特征维度: 768
    ↓
1D ResNet blocks:
  - ResBlock1D(768, 512)
  - ResBlock1D(512, 512)
  - ResBlock1D(512, 1024) [stride=2]
  - ResBlock1D(1024, 1024)
    ↓
AdaptiveAvgPool1d
    ↓
Classifier:
  - Linear(1024, 512) + LayerNorm + ReLU + Dropout
  - Linear(512, 8)    # 8类情感分类
```

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| epochs | 35 | 训练轮数 |
| batch_size | 16 | 批大小 |
| learning_rate | 1.5e-4 | 分类头学习率 |
| wav2vec_lr | 1.5e-5 | Wav2Vec2学习率 (0.1x) |
| hidden_size | 512 | 隐藏层维度 |
| dropout | 0.25 | Dropout比例 |
| label_smoothing | 0.1 | 标签平滑 |

## 数据增强

- **噪声注入**: 40%概率，强度0.005
- **时间偏移**: 40%概率，范围±10%
- **音量扰动**: 50%概率，范围0.8-1.2

## 情感类别

8类情感分类:
1. neutral (中性)
2. calm (平静)
3. happy (高兴)
4. sad (悲伤)
5. angry (愤怒)
6. fearful (恐惧)
7. disgust (厌恶)
8. surprised (惊讶)

## 数据集

使用 RAVDESS 数据集 (Ryerson Audio-Visual Database of Emotional Speech and Song):
- 训练集: 1152 样本 (80%)
- 测试集: 288 样本 (20%)

## 训练结果

| 指标 | 数值 |
|------|------|
| **准确率** | **89.24%** |
| F1 Score | 0.8926 |
| Precision | 0.8974 |
| Recall | 0.8924 |
| 最佳Epoch | 28 |
| 训练时间 | ~41分钟 |

### 混淆矩阵

各类别预测效果良好，neutral、calm、happy、sad、angry、fearful、disgust、surprised 均有较高的识别率。

## 生成图表

训练完成后可在 `output_resnet_wav2vec/images/` 目录下查看：

- `confusion_matrix.png` - 混淆矩阵
- `classification_report.png` - 分类报告
- `emotion_distribution.png` - 情感分布
- `training_curves.png` - 训练曲线
- `validation_accuracy.png` - 验证准确率
