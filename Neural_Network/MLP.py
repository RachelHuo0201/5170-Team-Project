import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

# ========== 0) 随机种子 ==========
np.random.seed(42)
torch.manual_seed(42)

# ========== 1) 读取数据 ==========
train_dataset  = pd.read_csv("/Users/cin/工程文件/Python/DS-Neural Network/winequality-white-renamed_去重后_traindata.csv")
val_dataset    = pd.read_csv("/Users/cin/工程文件/Python/DS-Neural Network/winequality-white-renamed_去重后_valdata.csv")

# 拆分特征与标签
X_train_np = train_dataset.drop(columns=["quality"]).to_numpy(dtype=np.float32)
y_train_np = train_dataset["quality"].to_numpy(dtype=np.int64)

y_train_np = np.where(y_train_np >= 8, 2,
               np.where(y_train_np >= 6, 1, 0)).astype(np.int64)
# y_train_np = np.where(y_train_np >= 6, 1,0
#                       ).astype(np.int64)

X_val_np   = val_dataset.drop(columns=["quality"]).to_numpy(dtype=np.float32)
y_val_np   = val_dataset["quality"].to_numpy(dtype=np.int64)


y_val_np   = np.where(y_val_np   >= 8, 2,
               np.where(y_val_np   >= 6, 1, 0)).astype(np.int64)
# y_val_np   = np.where(y_val_np   >= 6,1,0).astype(np.int64)



num_classes = 3
print("classes:", 2, "→ mapped to 0..", num_classes-1)

# ========== 3) 标准化（仅用训练集统计量）==========
mean = X_train_np.mean(axis=0, keepdims=True)
std  = X_train_np.std(axis=0, keepdims=True) + 1e-8

X_train_np = (X_train_np - mean) / std
X_val_np   = (X_val_np   - mean) / std

# ========== 4) 转为 Tensor + DataLoader ==========
X_train = torch.from_numpy(X_train_np.astype(np.float32))
y_train = torch.from_numpy(y_train_np.astype(np.int64))
X_val   = torch.from_numpy(X_val_np.astype(np.float32))
y_val   = torch.from_numpy(y_val_np.astype(np.int64))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=32, shuffle=False)
print("y_val:",y_val)
# ========== 5) 类别权重（缓解不均衡）==========
counts = np.bincount(y_train.numpy(), minlength=num_classes)
class_weights = 1.0 / (counts + 1e-6)
class_weights = class_weights / class_weights.mean()
class_weights = torch.tensor(class_weights, dtype=torch.float32)
print("train counts:", counts)
print("class_weights:", class_weights)

# ========== 6) 指标（不用 sklearn）==========
def accuracy(y_true, y_pred):
    return float((y_true == y_pred).mean())

def macro_f1(y_true, y_pred, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        precision = tp / (tp + fp + 1e-12)
        recall    = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)
    return float(np.mean(f1s))

# ========== 7) 模型：1 隐藏层 MLP ==========
class MLP(nn.Module):
    def __init__(self, in_dim=11, hidden=64, out_dim=3):
        super().__init__()
        assert out_dim is not None
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        return self.fc3(x)  # logits

model = MLP(in_dim=11, hidden=64, out_dim=num_classes)

# ========== 8) 损失与优化器 ==========
criterion = nn.CrossEntropyLoss()  # 加权交叉熵
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ========== 9) 训练循环（保存 Macro-F1 最优）==========
epochs = 100
best_f1 = -1.0
best_state = None

for epoch in range(1, epochs + 1):
    # ---- 训练 ----
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)

    # ---- 验证 ----
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            logits = model(xb)
            pred = logits.argmax(dim=1)
            y_true.append(yb.cpu().numpy())
            y_pred.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    val_acc = accuracy(y_true, y_pred)
    val_f1  = macro_f1(y_true, y_pred, num_classes)
    val_err = 1.0 - val_acc

    # 记录最优
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:03d} | TrainLoss={avg_loss:.4f} | Val Acc={val_acc:.4f} | "
          f"Val Err={val_err:.4f} | Macro-F1={val_f1:.4f}")

# ========== 10) 载入最优模型 ==========
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"\nLoaded best model (Val Macro-F1={best_f1:.4f})")

# ========== 11) 可选：保存模型 ==========
# torch.save({
#     "state_dict": model.state_dict(),
#     "classes": classes,           # 原始标签到 idx 的映射所需
#     "mean": mean, "std": std,     # 预测时需要相同的标准化
# }, "mlp_wine_white.pt")
