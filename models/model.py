"""
CNN + Transformer encoder model for Morse audio → text (CTC).
Replacing BiLSTM with Transformer: fully parallel across time axis,
much faster on GPU for long sequences (1000+ frames).

Input:  [B, n_mels, T]  log-mel spectrogram
Output: [T', B, vocab_size]  log-probabilities for CTC
"""
import math
import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.morse_map import VOCAB_SIZE


class MorseCNN(nn.Module):
    """Extract local frequency-time features. Also downsamples time by 2x."""
    def __init__(self, n_mels: int = 64, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 — halve freq only, keep full time resolution
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # freq ÷2, time unchanged
            nn.Dropout2d(0.1),

            # Block 2 — halve both (only 2x time reduction total)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # freq ÷2, time ÷2
            nn.Dropout2d(0.1),

            # Block 3 — halve freq only
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),  # freq ÷2, time unchanged
            nn.Dropout2d(0.1),
        )
        # After pooling: freq = n_mels//8, time = T//2
        cnn_out = 128 * (n_mels // 8)
        self.proj = nn.Linear(cnn_out, d_model)

    def forward(self, x):
        # x: [B, n_mels, T]
        x = x.unsqueeze(1)              # [B, 1, n_mels, T]
        x = self.net(x)                 # [B, 128, n_mels//8, T//2]
        B, C, F, T = x.shape
        x = x.permute(3, 0, 1, 2)      # [T//2, B, C, F]
        x = x.reshape(T, B, C * F)     # [T//2, B, features]
        x = self.proj(x)               # [T//2, B, d_model]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))  # [max_len, 1, d_model]

    def forward(self, x):
        # x: [T, B, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MorseModel(nn.Module):
    def __init__(
        self,
        n_mels:     int = 64,
        d_model:    int = 256,
        nhead:      int = 8,
        num_layers: int = 4,
        dim_ff:     int = 1024,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.cnn    = MorseCNN(n_mels, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=False,
            norm_first=True,        # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.fc = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, x):
        # x: [B, n_mels, T]
        x = self.cnn(x)           # [T//2, B, d_model]
        x = self.pos_enc(x)
        x = self.transformer(x)   # [T//2, B, d_model]
        x = self.fc(x)            # [T//2, B, vocab_size]
        return torch.log_softmax(x, dim=-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = MorseModel()
    print(f"Parameters: {model.count_params():,}")
    dummy = torch.randn(4, 64, 1500)
    out   = model(dummy)
    print(f"Input: {dummy.shape} → Output: {out.shape}")
