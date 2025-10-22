import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os


# 数据加载与划分
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "waimai_10k.csv")
df = pd.read_csv(csv_path)
# df = pd.read_csv("waimai_10k.csv")
df = df.dropna()
texts, labels = df["review"].tolist(), df["label"].tolist()

# 分层抽样
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(texts, labels):
    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")


# 数据集定义
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained("D:/Study/bert-base-chinese")

MAX_LEN = 64

def encode_texts(texts):
    enc = tokenizer(
        texts, padding='max_length', truncation=True,
        max_length=MAX_LEN, return_tensors='pt'
    )
    return enc["input_ids"], enc["attention_mask"]

train_ids, train_masks = encode_texts(train_texts)
val_ids, val_masks = encode_texts(val_texts)
train_labels = torch.tensor(train_labels)
val_labels = torch.tensor(val_labels)

class WaimaiDataset(Dataset):
    def __init__(self, ids, masks, labels):
        self.ids, self.masks, self.labels = ids, masks, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {
            "ids": self.ids[i],
            "mask": self.masks[i],
            "labels": self.labels[i]
        }

train_loader = DataLoader(WaimaiDataset(train_ids, train_masks, train_labels), batch_size=32, shuffle=True)
val_loader = DataLoader(WaimaiDataset(val_ids, val_masks, val_labels), batch_size=32)


# Transformer 模型实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe = pe.unsqueeze(0)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, enable=True):
        super().__init__()
        self.enable = enable
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dk = d_model // num_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        if not self.enable:
            return x

        B, L, D = x.size()
        qkv = self.qkv(x)  # [B, L, 3*D]
        qkv = qkv.view(B, L, 3, self.num_heads, self.dk)  # [B, L, 3, heads, dk]
        q = qkv[:, :, 0].permute(0, 2, 1, 3)  # [B, heads, L, dk]
        k = qkv[:, :, 1].permute(0, 2, 1, 3)  # [B, heads, L, dk]
        v = qkv[:, :, 2].permute(0, 2, 1, 3)  # [B, heads, L, dk]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # [B, heads, L, L]
        if mask is not None:
            mask = mask.to(torch.bool).to(x.device)
            if mask.dim() == 2:  # [B, L] -> [B, 1, 1, L]
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:  # [B, L, L] -> [B, 1, L, L]
                mask = mask.unsqueeze(1)
            else:
                pass
            scores = scores.masked_fill(mask == 0, float('-1e9'))
        attn = F.softmax(scores, dim=-1)  # [B, heads, L, L]
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, enable=True):
        super().__init__()
        self.enable = enable
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x) if self.enable else x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, ffn_enable=True, sa_enable=True, res_ln=True):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, heads, sa_enable)
        self.ffn = FeedForward(d_model, 4*d_model, ffn_enable)
        self.res_ln = res_ln
        if res_ln:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        if self.res_ln:
            x = self.norm1(x + attn_out)
        else:
            x = attn_out
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out) if self.res_ln else ffn_out

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4,
                 ffn=True, sa=True, res_ln=True, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, ffn_enable=ffn, sa_enable=sa, res_ln=res_ln)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, ids, mask):
        x = self.embed(ids)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc(x[:, 0, :])  # 取[CLS]位置


# 训练逻辑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using gpu")
else:
    print("Using cpu")

model = TransformerClassifier(
    vocab_size=len(tokenizer),
    d_model=128,
    num_layers=2,
    num_heads=4,
    ffn=True,        #  Position-wise FFN
    sa=True,         # Multi-head Self-Attention
    res_ln=True,     # 残差连接 + LayerNorm
    num_classes=2
).to(device)
# model = TransformerClassifier(
#     vocab_size=len(tokenizer),
#     d_model=128,
#     num_layers=2,
#     num_heads=4,
#     ffn=False,       # 关闭 FFN
#     sa=True,
#     res_ln=True,
#     num_classes=2
# ).to(device)

optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=5)
criterion = nn.CrossEntropyLoss()

train_losses, val_accs = [], []

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # 验证
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(ids, mask)
            preds += outputs.argmax(1).cpu().tolist()
            trues += labels.cpu().tolist()
    acc = accuracy_score(trues, preds)
    print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={acc:.4f}")
    train_losses.append(total_loss/len(train_loader))
    val_accs.append(acc)


# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)  
plt.plot(train_losses, 'b-', label="Train Loss", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training Loss without Residual & LayerNorm")
plt.grid(True, alpha=0.3)


plt.subplot(1, 2, 2) 
plt.plot(val_accs, 'r-', label="Val Accuracy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Validation Accuracy without Residual & LayerNorm")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()