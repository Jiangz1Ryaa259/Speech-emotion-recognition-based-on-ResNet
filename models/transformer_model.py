"""
基于Transformer的语音情感识别模型
使用预训练的音频模型进行特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerSER(nn.Module):
    """基于Transformer的语音情感识别模型"""
    def __init__(self, input_size=768, num_class=4, hidden_size=256,
                 nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.feature_dim = input_size

        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, hidden_size))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 类令牌
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_class)
        )

    def forward(self, x):
        # x: (batch, seq_len, feature_dim) 或 (batch, feature_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, feature_dim)

        batch_size = x.shape[0]

        # 投影
        x = self.projection(x)  # (batch, seq_len, hidden_size)

        # 添加类令牌
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, hidden_size)

        # 添加位置编码
        x = x + self.pos_embedding[:, :x.shape[1], :]

        # Transformer编码
        x = self.transformer(x)

        # 取类令牌的输出
        x = x[:, 0]  # (batch, hidden_size)

        # 分类
        x = self.head(x)
        return x


class ConformerSER(nn.Module):
    """基于Conformer的语音情感识别模型"""
    def __init__(self, input_size=768, num_class=4, hidden_size=256, num_layers=4):
        super().__init__()
        self.feature_dim = input_size

        # 投影层
        self.input_projection = nn.Linear(input_size, hidden_size)

        # 卷积模块
        self.conv_module = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=1),
            nn.GLU(dim=-1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.Sigmoid()
        )

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        # 多头注意力
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, dropout=0.1, batch_first=True)

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            self._make_transformer_layer(hidden_size) for _ in range(num_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_class)
        )

    def _make_transformer_layer(self, hidden_size):
        return nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 输入投影
        x = self.input_projection(x)  # (batch, seq, hidden)

        for layer in self.transformer_layers:
            # 前馈
            x = x + 0.5 * self.ffn(x)

            # 自注意力
            attn_out, _ = self.attn(x, x, x)
            x = x + attn_out

            # 卷积模块
            conv_in = x.transpose(1, 2)  # (batch, hidden, seq)
            conv_out = self.conv_module(conv_in)
            x = x + conv_out.transpose(1, 2)

        # 全局池化
        x = x.mean(dim=1)  # (batch, hidden)

        # 分类
        return self.classifier(x)