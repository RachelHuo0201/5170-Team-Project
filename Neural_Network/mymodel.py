import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd
nn_in=12
nn_out=2
n_hidden=3
epoch=100
nn_neural=64

origin_dataset=pd.read_csv("/Users/cin/工程文件/R/5170-Team-Project/Neural_Network/winequality-white-renamed_去重后.csv")
origin_dataset_red=pd.read_csv("/Users/cin/工程文件/R/5170-Team-Project/Neural_Network/winequality-red-renamed_去重后.csv")
# origin_dataset['type_white']=1
# origin_dataset['type_red']=0
# origin_dataset_red['type_white']=0
# origin_dataset_red['type_red']=1
# origin_dataset = pd.concat([origin_dataset, origin_dataset_red], axis=0)
# origin_x_np=origin_dataset.drop(columns=["quality"]).to_numpy(dtype=np.float32)
# origin_y_np=origin_dataset['quality'].to_numpy(dtype=np.int64)

origin_dataset['type']=1
origin_dataset_red['type']=0
origin_dataset = pd.concat([origin_dataset, origin_dataset_red], axis=0)

origin_x_np=origin_dataset.drop(columns=["type"]).to_numpy(dtype=np.float32)
origin_y_np=origin_dataset['type'].to_numpy(dtype=np.int64)

# for i in range(0,origin_x_np.shape[0]):
#     if origin_y_np[i] <=6:
#         origin_y_np[i] = 0
#     else:
#         origin_y_np[i] = 1

count=set(origin_y_np)
for i in count:
    print(i," amount: ",np.sum(origin_y_np == i))
print(origin_x_np.shape)
# kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# for train_idx, val_idx in kf.split(origin_x_np, origin_y_np):
#     X_train, X_val = origin_x_np[train_idx], origin_x_np[val_idx]
#     y_train, y_val = origin_y_np[train_idx], origin_y_np[val_idx]
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(

    origin_x_np,
    origin_y_np,
    test_size=0.2,
    stratify=origin_y_np,
    random_state=141,
)
mean_train = x_train_np.mean(axis=0, keepdims=True)
std_train  = x_train_np.std(axis=0, keepdims=True)
x_train_np = (x_train_np - mean_train) / std_train
x_test_np  = (x_test_np  - mean_train) / std_train



count=set(y_test_np)
print("quality in test set")
for i in count:
    print(i," amount: ",np.sum(y_test_np == i))
train_loader = DataLoader(TensorDataset( torch.from_numpy(x_train_np), torch.from_numpy(y_train_np)), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(x_test_np),   torch.from_numpy(y_test_np)),   batch_size=32, shuffle=False)
model=nn.Sequential(
    nn.Linear(nn_in, nn_neural),
    nn.ReLU(),
    nn.Linear(nn_neural, nn_neural),
    nn.ReLU(),
    nn.Linear(nn_neural, nn_neural),
    nn.ReLU(),
    nn.Linear(nn_neural, nn_out),
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    n=0
    for x,y in train_loader:
        optimizer.zero_grad()
        y_pred=model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*x.size(0)

        correct += (y_pred.argmax(1) == y).sum().item()
         # 打印损失值
        n += x.size(0)
    running_loss = running_loss / n
    train_acc =correct / n
    print('epoch: ', epoch, 'loss: ', running_loss, 'acc: ', train_acc)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    n = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in val_loader:
            logits = model(x)  # 不要再套 sigmoid
            pred_labels = logits.argmax(dim=1)

            all_preds.extend(pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            correct += (pred_labels == y).sum().item()
            n += x.size(0)
    val_acc = correct / n
    print('train acc: ', val_acc)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

print("final Train")
model.eval()
running_loss = 0.0
correct = 0
total = 0
n = 0
all_preds = []
all_labels = []
with torch.no_grad():
    for x, y in val_loader:
        y_pred = F.sigmoid(model(x))
        pred_labels = y_pred.argmax(dim=1)

        all_preds.extend(pred_labels.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        y_pred_onehot = F.one_hot(pred_labels, num_classes=nn_out)
        correct += (pred_labels == y).sum().item()
            # 打印损失值
        n += x.size(0)
    val_acc = correct / n
print( 'acc: ', val_acc)
f1_macro = f1_score(all_labels, all_preds, average='macro')
f1_weighted = f1_score(all_labels, all_preds, average='weighted')

print(f'Validation accuracy: {val_acc:.4f}')
print(f'F1 (macro): {f1_macro:.4f}')
print(f'F1 (weighted): {f1_weighted:.4f}')