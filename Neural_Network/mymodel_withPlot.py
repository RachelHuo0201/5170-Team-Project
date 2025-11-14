import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc,
    average_precision_score, precision_recall_curve,
    cohen_kappa_score, matthews_corrcoef,
    balanced_accuracy_score, log_loss
)
import pandas as pd

# -------------------- 超参数 --------------------
nn_in = 13          # 输入维度（你加了两列 type_*，共 13）
nn_out = 2          # 二分类
n_hidden = 3
epochs = 100
nn_neural = 64
batch_train = 64
batch_val = 32
lr = 1e-4
seed = 141
torch.manual_seed(seed)
np.random.seed(seed)

# -------------------- 数据加载与二分类映射 --------------------
origin_white = pd.read_csv("/Users/cin/工程文件/R/5170-Team-Project/Neural_Network/winequality-white-renamed_去重后.csv")
origin_red   = pd.read_csv("/Users/cin/工程文件/R/5170-Team-Project/Neural_Network/winequality-red-renamed_去重后.csv")
origin_white['type_white'] = 1; origin_white['type_red'] = 0
origin_red['type_white']   = 0; origin_red['type_red']   = 1
df = pd.concat([origin_white, origin_red], axis=0, ignore_index=True)

X = df.drop(columns=["quality"]).to_numpy(dtype=np.float32)
y = df['quality'].to_numpy(dtype=np.int64)

# quality 映射成二分类：<=6 为 0， >6 为 1
y = np.where(y <= 6, 0, 1).astype(np.int64)

# 数据切分 & 标准化（用训练集统计量）
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=seed
)
mean_train = x_train_np.mean(axis=0, keepdims=True)
std_train  = x_train_np.std(axis=0, keepdims=True) + 1e-8
x_train_np = (x_train_np - mean_train) / std_train
x_test_np  = (x_test_np  - mean_train) / std_train

print("Train class counts:", np.bincount(y_train_np))
print("Test  class counts:", np.bincount(y_test_np))
print("X shape:", X.shape)

train_loader = DataLoader(
    TensorDataset(torch.from_numpy(x_train_np), torch.from_numpy(y_train_np)),
    batch_size=batch_train, shuffle=True
)
val_loader   = DataLoader(
    TensorDataset(torch.from_numpy(x_test_np), torch.from_numpy(y_test_np)),
    batch_size=batch_val, shuffle=False
)

# -------------------- 模型 --------------------
model = nn.Sequential(
    nn.Linear(nn_in, nn_neural),
    nn.ReLU(),
    nn.Linear(nn_neural, nn_neural),
    nn.ReLU(),
    nn.Linear(nn_neural, nn_neural),
    nn.ReLU(),
    nn.Linear(nn_neural, nn_out),
)

criterion = nn.CrossEntropyLoss()                  # 输入 logits，标签 long
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# -------------------- 指标/绘图函数 --------------------
def compute_topk_accuracy(probs_or_logits, y_true, k=3, are_logits=False):
    """
    支持 [N,C] 概率或 logits；若是二分类的一列概率也可。
    返回 (topk_acc, C, k_eff)
    """
    p = torch.as_tensor(probs_or_logits)
    y = torch.as_tensor(y_true).long()

    # 若一维或 [N,1]，视作二分类正类概率，补两列 [1-p, p]
    if p.ndim == 1 or (p.ndim == 2 and p.size(1) == 1):
        if are_logits:
            pos = torch.sigmoid(p.view(-1))
        else:
            pos = p.view(-1)
        p = torch.stack([1.0 - pos, pos], dim=1)

    if are_logits:
        p = F.softmax(p, dim=1)

    assert p.ndim == 2, f"probs should be [N, C], got {tuple(p.shape)}"
    N, C = p.shape
    k_eff = max(1, min(k, C))

    topk_idx = p.topk(k_eff, dim=1).indices
    acc = (topk_idx == y.unsqueeze(1)).any(dim=1).float().mean().item()
    return acc, C, k_eff

def plot_learning_curves(history):
    epochs = range(1, len(history['train_loss'])+1)

    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve'); plt.legend(); plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve'); plt.legend(); plt.tight_layout()
    plt.show()

    if 'val_f1' in history:
        plt.figure()
        plt.plot(epochs, history['val_f1'], label='Val Macro-F1')
        plt.xlabel('Epoch'); plt.ylabel('F1'); plt.title('F1 (Macro) Curve'); plt.legend(); plt.tight_layout()
        plt.show()

def plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix'):
    if normalize:
        cm = cm.astype(np.float64)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha='right')
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else str(cm[i, j])
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout(); plt.show()

def plot_multiclass_roc_pr(y_true_np, prob_np, class_names):
    # 统一为 numpy
    y_true_np = np.array(y_true_np).reshape(-1)
    prob_np   = np.array(prob_np)
    n_classes = prob_np.shape[1]

    # ROC (OvR)
    plt.figure()
    for c in range(n_classes):
        y_true_bin = (y_true_np == c).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, prob_np[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {class_names[c]} (AUC={roc_auc:.3f})')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC (OvR)')
    plt.legend(); plt.tight_layout(); plt.show()

    # PR (OvR)
    plt.figure()
    for c in range(n_classes):
        y_true_bin = (y_true_np == c).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_bin, prob_np[:, c])
        ap = average_precision_score(y_true_bin, prob_np[:, c])
        plt.plot(recall, precision, label=f'Class {class_names[c]} (AP={ap:.3f})')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR (OvR)')
    plt.legend(); plt.tight_layout(); plt.show()

# -------------------- 训练循环 --------------------
history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'val_f1':[]}

for ep in range(epochs):
    # Train
    model.train()
    train_running_loss, train_correct, n_train = 0.0, 0, 0
    for x, yb in train_loader:
        optimizer.zero_grad()
        logits = model(x)                      # [N, C]
        loss = criterion(logits, yb)           # CrossEntropyLoss 接受 logits + labels
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item() * x.size(0)
        train_correct += (logits.argmax(1) == yb).sum().item()
        n_train += x.size(0)

    train_loss = train_running_loss / max(1, n_train)
    train_acc  = train_correct / max(1, n_train)

    # Val
    model.eval()
    val_running_loss, val_correct, n_val = 0.0, 0, 0
    all_labels, all_preds = [], []
    prob_list = []
    with torch.no_grad():
        for x, yb in val_loader:
            logits = model(x)                  # logits
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)

            val_running_loss += loss.item() * x.size(0)
            val_correct += (probs.argmax(1) == yb).sum().item()
            n_val += x.size(0)

            all_labels.extend(yb.cpu().numpy())
            all_preds.extend(probs.argmax(1).cpu().numpy())
            prob_list.append(probs.cpu())

    val_loss = val_running_loss / max(1, n_val)
    val_acc  = val_correct / max(1, n_val)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = torch.cat(prob_list, dim=0).numpy()   # [N, C]

    f1_macro = f1_score(all_labels, all_preds, average='macro')

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(f1_macro)

    print(f"Epoch {ep:03d} | TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} "
          f"| TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | Macro-F1={f1_macro:.4f}")

# -------------------- 训练结束：最终评估与可视化 --------------------
class_names = [str(i) for i in range(nn_out)]

print("\n=== Classification Report (Val) ===")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
plot_confusion_matrix(cm, class_names, normalize=False, title='Confusion Matrix (Counts)')
plot_confusion_matrix(cm, class_names, normalize=True,  title='Confusion Matrix (Normalized)')

# 其它整体指标
kappa = cohen_kappa_score(all_labels, all_preds)
mcc   = matthews_corrcoef(all_labels, all_preds)
bacc  = balanced_accuracy_score(all_labels, all_preds)
topk_acc, C, k_eff = compute_topk_accuracy(all_probs, all_labels, k=3, are_logits=False)  # 二分类时会自动用 k=2

print(f"Cohen Kappa: {kappa:.4f}")
print(f"Matthews Corrcoef: {mcc:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Top-{k_eff} (C={C}) Accuracy: {topk_acc:.4f}")

# Log loss（需要概率）
try:
    print(f"Log Loss: {log_loss(all_labels, all_probs):.4f}")
except Exception as e:
    print("Log loss not computed:", e)

# ROC-AUC / PR-AUC（OvR，多分类通用；二分类同样适用）
try:
    roc_auc_macro = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    print(f"ROC AUC (macro, OvR): {roc_auc_macro:.4f}")
except Exception as e:
    print("ROC-AUC not computed:", e)

plot_learning_curves(history)
plot_multiclass_roc_pr(all_labels, all_probs, class_names)
