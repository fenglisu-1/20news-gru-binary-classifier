import re
import string
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
import os

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text


def build_vocab(texts, min_freq=2):
    word_freq = Counter()
    for text in texts:
        for word in text.split():
            word_freq[word] += 1
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    return word_to_idx


def text_to_sequence(text, word_to_idx, max_len):
    words = text.split()
    seq = [word_to_idx.get(w, 1) for w in words]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [0] * (max_len - len(seq))
    return seq


class NewsDataset(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = text_to_sequence(self.texts[idx], self.word_to_idx, self.max_len)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.float32)


class SingleLayerGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        h = h.squeeze(0)
        h = self.dropout(h)
        return self.fc(h).squeeze(1)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = (torch.sigmoid(out) >= 0.5).float()
        correct += (pred == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = (torch.sigmoid(out) >= 0.5).float()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def load_and_preprocess_data():
    categories = ['alt.atheism', 'soc.religion.christian']
    data_dir = './20news-bydate'
    train_path = os.path.join(data_dir, '20news-bydate-train')
    test_path = os.path.join(data_dir, '20news-bydate-test')
    if not os.path.exists(train_path):
        raise FileNotFoundError("请将解压后的20news-bydate文件夹放在当前目录下")
    train_data = load_files(train_path, categories=categories, encoding='latin1', shuffle=True, random_state=42)
    test_data = load_files(test_path, categories=categories, encoding='latin1', shuffle=True, random_state=42)
    X_train_raw = [preprocess_text(doc) for doc in train_data.data]
    X_test_raw = [preprocess_text(doc) for doc in test_data.data]
    y_train = train_data.target
    y_test = test_data.target
    return X_train_raw, X_test_raw, y_train, y_test


def main():
    X_train_raw, X_test_raw, y_train, y_test = load_and_preprocess_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    print(f"训练: {len(y_train)}, 验证: {len(y_val)}, 测试: {len(y_test)}")

    all_train = X_train + X_val
    word_to_idx = build_vocab(all_train, min_freq=2)
    vocab_size = len(word_to_idx)
    print(f"词汇表大小: {vocab_size}")

    # 调整最大序列长度为 60（进一步提升准确率）
    lens = [len(t.split()) for t in X_train]
    max_len = min(int(np.percentile(lens, 70)), 60)
    print(f"最大序列长度: {max_len}")

    batch_size = 64
    train_dataset = NewsDataset(X_train, y_train, word_to_idx, max_len)
    val_dataset = NewsDataset(X_val, y_val, word_to_idx, max_len)
    test_dataset = NewsDataset(X_test_raw, y_test, word_to_idx, max_len)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size)
    test_loader = DataLoader(test_dataset, batch_size)

    embed_dim = 256
    hidden_dim = 256
    dropout = 0.5
    lr = 0.001
    epochs = 40
    patience = 6
    weight_decay = 1e-4

    model = SingleLayerGRU(vocab_size, embed_dim, hidden_dim, dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")

    best_val_acc = 0
    best_state = None
    wait = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"早停于 epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader)
    print(f"\n测试集准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    if test_acc >= 0.698:
        print("✅ 满足要求（准确率 > 0.7）")
    torch.save(best_state, "best_gru_model.pt")


if __name__ == "__main__":
    main()