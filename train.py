import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 读取数据
data = pd.read_csv('./train.csv')

# 编码分类特征
label_encoder = LabelEncoder()
data['id'] = label_encoder.fit_transform(data['id'])
data['type'] = label_encoder.fit_transform(data['type'])

# 检查数据
print(data.head())

# 按 'type' 分组
grouped = data.groupby('type')

# 创建最终的DataFrame
result = []

for type_value, group in grouped:
    ids = group['id'].unique().tolist()
    targets = []
    for id_value in ids:
        target_sequence = group[group['id'] == id_value].sort_values('dt', ascending=False)['target'].tolist()
        targets.append(target_sequence)
    result.append([type_value, ids, targets])

# 转换为 DataFrame
result_df = pd.DataFrame(result, columns=['type', 'id_list', 'target_sequences'])

# 创建seq2seq训练数据
def create_sequences(target_sequences, seq_len, pred_len):
    X, y = [], []
    for seq in target_sequences:
        if len(seq) < seq_len + pred_len:
            continue
        for i in range(len(seq) - seq_len - pred_len + 1):
            X.append(seq[i:i+seq_len])
            y.append(seq[i+seq_len:i+seq_len+pred_len])
    return X, y

seq_len = 200
pred_len = 10
X, y = [], []
for idx, row in result_df.iterrows():
    targets = row['target_sequences']
    X_type, y_type = create_sequences(targets, seq_len, pred_len)
    X.extend(X_type)
    y.extend(y_type)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 创建数据加载器
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class Seq2SeqModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(Seq2SeqModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

input_dim = 1
hidden_dim = 64
num_layers = 1
output_dim = pred_len
model = Seq2SeqModel(input_dim, hidden_dim, num_layers, output_dim).cuda()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型并保存最佳模型
num_epochs = 100
best_loss = float('inf')
best_model_path = 'best_seq2seq_model.pth'

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch'):
        X_batch = X_batch.unsqueeze(-1)  # 添加特征维度
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        optimizer.zero_grad()
        outputs = model(X_batch.cuda())
        loss = criterion(outputs, y_batch.cuda())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Best Loss: {best_loss:.4f}')

# 加载最佳模型
model.load_state_dict(torch.load(best_model_path))

# 使用最佳模型进行预测并输出结果
def predict(model, input_seq):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).cuda()
        output_seq = model(input_seq).squeeze().numpy()
    return output_seq

for idx, row in result_df.iterrows():
    id_list = row['id_list']
    targets = row['target_sequences']
    for i, target_seq in enumerate(targets):
        if len(target_seq) >= seq_len:
            input_seq = target_seq[-seq_len:]
            pred_seq = predict(model, input_seq)
            print(f'ID: {id_list[i]}, Type: {row["type"]}, Predicted Target: {pred_seq}')

# 将预测结果转换为 DataFrame 并保存
prediction_df = pd.DataFrame(results)
prediction_df.to_csv('./predicted_targets.csv', index=False)
print(prediction_df.head())