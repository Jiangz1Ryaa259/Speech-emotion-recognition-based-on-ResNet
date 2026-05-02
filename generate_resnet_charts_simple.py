#!/usr/bin/env python
"""
使用保存的结果生成Wav2Vec2 + ResNet模型的评估图表
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def main():
    print("=" * 70)
    print("生成 Wav2Vec2 + ResNet 评估图表")
    print("=" * 70)

    # 加载结果
    with open('output_resnet_wav2vec/results.json', 'r') as f:
        results = json.load(f)

    # 加载训练历史
    with open('output_resnet_wav2vec/training_history.json', 'r') as f:
        history = json.load(f)

    acc = results['accuracy']
    f1 = results['f1']
    prec = results['precision']
    rec = results['recall']
    cm = np.array(results['confusion_matrix'])

    labels_name = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

    print(f"\n结果:")
    print(f"Accuracy:  {acc:.4f} ({100.*acc:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

    # 创建输出目录
    output_dir = 'output_resnet_wav2vec/images'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_name, yticklabels=labels_name)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix\nWav2Vec2 + ResNet Speech Emotion Recognition')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n已保存: {output_dir}/confusion_matrix.png")

    # 2. 分类报告图
    # 重新计算每个类别的指标
    report_dict = {}
    for i, label in enumerate(labels_name):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = cm[i, :].sum()
        report_dict[label] = {'precision': precision, 'recall': recall, 'f1-score': f1_score, 'support': support}

    metrics_names = ['precision', 'recall', 'f1-score']
    x = np.arange(len(labels_name))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, m in enumerate(metrics_names):
        values = [report_dict[label][m] for label in labels_name]
        bars = ax.bar(x + i * width, values, width, label=m, color=colors[i])
        # 添加数值标签
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Emotion Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classification Report - Per Class Metrics\n(Wav2Vec2 + ResNet)', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels_name, rotation=45, ha='right', fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/classification_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/classification_report.png")

    # 3. 情感分布
    plt.figure(figsize=(10, 6))
    unique, counts = np.unique(np.argmax(cm, axis=1), return_counts=True)
    # 使用实际标签计数
    totals = cm.sum(axis=1)
    bars = plt.bar(labels_name, totals, color='steelblue', edgecolor='navy', alpha=0.8)
    plt.xlabel('Emotion Category', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Test Set Emotion Distribution\n(Wav2Vec2 + ResNet)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    for bar, count in zip(bars, totals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/emotion_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/emotion_distribution.png")

    # 4. 训练曲线
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss Curve', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Accuracy曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation Acc')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    plt.suptitle('Wav2Vec2 + ResNet Training Progress', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/training_curves.png")

    # 5. 验证准确率变化曲线
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [a * 100 for a in history['val_acc']], 'r-o', linewidth=2, markersize=4, label='Validation Accuracy')
    best_epoch = np.argmax(history['val_acc']) + 1
    best_acc = max(history['val_acc']) * 100
    plt.axhline(y=best_acc, color='green', linestyle='--', alpha=0.7, label=f'Best: {best_acc:.2f}%')
    plt.scatter([best_epoch], [best_acc], color='green', s=150, zorder=5, marker='*', label=f'Best Epoch: {best_epoch}')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Validation Accuracy Over Epochs\n(Wav2Vec2 + ResNet)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(range(0, len(epochs)+1, 5))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/validation_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_dir}/validation_accuracy.png")

    print(f"\n所有图表已保存到: {output_dir}/")
    print(f"\n" + "="*50)
    print("结果摘要 (Wav2Vec2 + ResNet):")
    print("="*50)
    print(f"最终准确率: {100.*acc:.2f}%")
    print(f"F1分数: {f1:.4f}")
    print(f"最佳验证准确率: {100.*results['best_accuracy']:.2f}%")
    print(f"训练时间: {results['total_time_seconds']/60:.1f} 分钟")
    print(f"最佳模型在Epoch: {results.get('best_epoch', 'N/A')}")


if __name__ == '__main__':
    main()