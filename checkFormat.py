import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

# 数据加载和预处理
data = pd.read_csv('/Users/coconut/data/electricity/train.csv')

# 编码分类特征
label_encoder = LabelEncoder()
data['id'] = label_encoder.fit_transform(data['id'])
data['type'] = label_encoder.fit_transform(data['type'])

# 特征和标签
features = ['id', 'dt', 'type']
target = 'target'

X = data[features]
y = data[target]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# LSTM-CNN 模型
class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(LSTM_CNN_Model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_size, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 计算全连接层的输入维度
        self.fc1_input_dim = 16 * (input_dim // 2)
        self.fc1 = nn.Linear(self.fc1_input_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 假设 X_train 的特征数是3
input_dim = X_train.shape[1]
hidden_size = 64  # 假设隐藏状态的维度为 64
model = LSTM_CNN_Model(input_dim, hidden_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    test_loss = criterion(y_pred, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')
