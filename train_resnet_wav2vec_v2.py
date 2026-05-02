#!/usr/bin/env python
"""
Wav2Vec2 + ResNet微调语音情感识别 - 优化版v2
平衡速度和效果，在合理时间内达到更好效果
"""

import os
import sys
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from datetime import timedelta


class ResBlock1D(nn.Module):
    """1D残差块用于语音特征"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class Wav2Vec2ResNetSERv2(nn.Module):
    """优化的Wav2Vec2 + ResNet语音情感识别模型v2"""
    def __init__(self, num_class=8, hidden_size=512, num_res_blocks=4, dropout=0.25):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')

        # 解冻最后4层Wav2Vec2编码器
        for name, param in self.wav2vec2.named_parameters():
            layer_num = None
            if 'encoder.layers.' in name:
                try:
                    layer_num = int(name.split('encoder.layers.')[1].split('.')[0])
                except:
                    pass
            if layer_num is not None and layer_num >= 8:  # 解冻 8, 9, 10, 11
                param.requires_grad = True
            else:
                param.requires_grad = False

        feature_dim = 768

        # 1D ResNet特征提取 + 残差连接
        self.resnet = nn.Sequential(
            ResBlock1D(feature_dim, hidden_size, stride=1),
            nn.Dropout(dropout),
            ResBlock1D(hidden_size, hidden_size, stride=1),
            nn.Dropout(dropout),
            ResBlock1D(hidden_size, hidden_size * 2, stride=2),  # 下采样
            nn.Dropout(dropout),
            ResBlock1D(hidden_size * 2, hidden_size * 2, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_class)
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        x = outputs.last_hidden_state
        x = x.transpose(1, 2)
        x = self.resnet(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


class AudioDataset(Dataset):
    """音频数据集 - 增强版数据增强"""
    def __init__(self, data_list_path, mode='train', max_duration=5, sample_rate=16000):
        with open(data_list_path, 'r', encoding='utf-8') as f:
            self.lines = f.readlines()
        self.mode = mode
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')

    def __len__(self):
        return len(self.lines)

    def _augment(self, samples):
        """数据增强"""
        if np.random.random() < 0.4:
            noise = np.random.randn(len(samples)).astype(np.float32) * 0.005
            samples = samples + noise

        if np.random.random() < 0.4:
            shift = np.random.randint(-int(self.sample_rate * 0.1), int(self.sample_rate * 0.1))
            samples = np.roll(samples, shift)

        if np.random.random() < 0.5:
            gain = np.random.uniform(0.8, 1.2)
            samples = samples * gain

        return samples

    def __getitem__(self, idx):
        path, label = self.lines[idx].replace('\n', '').split('\t')

        try:
            samples, sr = librosa.load(path, sr=self.sample_rate)
            samples = samples.astype(np.float32)
        except:
            return self.__getitem__((idx + 1) % len(self.lines))

        if len(samples) < self.sample_rate * 0.1:
            return self.__getitem__((idx + 1) % len(self.lines))

        if self.mode == 'train':
            samples = self._augment(samples)

        max_samples = int(self.sample_rate * self.max_duration)
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        inputs = self.feature_extractor(samples, sampling_rate=self.sample_rate, return_tensors='pt')
        input_values = inputs.input_values

        return input_values.squeeze(0), int(label)


def collate_fn(batch):
    features, labels = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    padded = []
    for f in features:
        pad_len = max_len - f.shape[0]
        if pad_len > 0:
            f = np.pad(f, (0, pad_len))
        padded.append(torch.FloatTensor(f))
    return torch.stack(padded), torch.LongTensor(labels)


def train_resnet_wav2vec_v2(max_epoch=35, batch_size=16, lr=1.5e-4):
    """训练函数v2"""
    print("=" * 70)
    print("语音情感识别 - Wav2Vec2 + ResNet 优化版v2 (GPU加速)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\n加载数据集...")
    train_dataset = AudioDataset('dataset/train_list.txt', mode='train')
    test_dataset = AudioDataset('dataset/test_list.txt', mode='eval')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

    print("\n初始化优化模型v2...")
    model = Wav2Vec2ResNetSERv2(
        num_class=8,
        hidden_size=512,
        num_res_blocks=4,
        dropout=0.25
    ).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,}")

    # 标签平滑
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 分层学习率
    optimizer = torch.optim.AdamW([
        {'params': model.wav2vec2.parameters(), 'lr': lr * 0.1},
        {'params': model.resnet.parameters(), 'lr': lr * 0.5},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    print(f"\n开始训练 ({max_epoch} epochs)...")
    best_acc = 0
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    start_time = time.time()

    for epoch in range(1, max_epoch + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{max_epoch}')
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.1f}%'})

        scheduler.step()

        # 验证
        model.eval()
        preds, labels_all = [], []
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc='Evaluating', leave=False):
                features = features.to(device)
                outputs = model(features)
                pred = outputs.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                labels_all.extend(labels.numpy())

        val_acc = accuracy_score(labels_all, preds)
        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total

        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        elapsed = time.time() - start_time

        print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Train Acc={train_acc*100:.2f}%, Val Acc={val_acc*100:.2f}% [{str(timedelta(seconds=int(elapsed)))}]')

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            os.makedirs('output_resnet_wav2vec/best_model', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
            }, 'output_resnet_wav2vec/best_model/model.pth')
            print(f'  -> 最佳模型已保存! Acc={val_acc*100:.2f}%')

    total_time = time.time() - start_time

    with open('output_resnet_wav2vec/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\n训练完成! 最终评估...")
    checkpoint = torch.load('output_resnet_wav2vec/best_model/model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    preds, labels_all = [], []
    with torch.no_grad():
        for features, labels in tqdm(test_loader, desc='Final Evaluation'):
            features = features.to(device)
            outputs = model(features)
            pred = outputs.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            labels_all.extend(labels.numpy())

    acc = accuracy_score(labels_all, preds)
    f1 = f1_score(labels_all, preds, average='weighted')
    prec = precision_score(labels_all, preds, average='weighted')
    rec = recall_score(labels_all, preds, average='weighted')
    cm = confusion_matrix(labels_all, preds)

    results = {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': cm.tolist(),
        'best_accuracy': best_acc,
        'best_epoch': best_epoch,
        'total_time_seconds': total_time
    }
    with open('output_resnet_wav2vec/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("最终结果 (Wav2Vec2 + ResNet 优化版v2):")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f} ({100.*acc:.2f}%)")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"最佳模型: Epoch {best_epoch}, Acc={best_acc*100:.2f}%")
    print(f"总时间: {total_time/60:.1f} 分钟")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1.5e-4)
    args = parser.parse_args()

    train_resnet_wav2vec_v2(max_epoch=args.epoch, batch_size=args.batch_size, lr=args.lr)