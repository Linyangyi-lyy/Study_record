import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# 参数设置
seq_len = 20
epochs = 35
test_size = 300

input_size=1
hidden_size=64
num_layers=2
output_size=1

# 读取数据
df = pd.read_csv('SH600519.csv')
df['data'] = pd.to_datetime(df['date']) # 转换日期格式
df = df.sort_values('date').reset_index(drop=True) # 建立新索引
total_samples = len(df)

# 数据处理
dataset = df['open'].values # 取开盘价作为数据集,转为numpy数组
dataset_2d = dataset.reshape(-1, 1) # 转换为二维数组,用于归一化
dataset_2d_train = dataset_2d[:total_samples - test_size]
dataset_2d_test = dataset_2d[total_samples - test_size:]
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset_2d)
# print(dataset_scaled.ndim)

# 划分测试集和训练集
train_dataset_scaled = scaler.fit_transform(dataset_2d_train)
test_dataset_scaled = scaler.fit_transform(dataset_2d_test)

# 定义数据集构造函数
def make_dataset(dataset, seq_len):
    x = [] # 输入序列
    y = [] # 预测目标
    for i in range(len(dataset) - seq_len):
        x.append(dataset[i:i + seq_len, 0])
        y.append(dataset[i + seq_len, 0])
    return np.array(x), np.array(y)

# 准备序列样本
train_x, train_y = make_dataset(train_dataset_scaled, seq_len)
test_x, test_y = make_dataset(test_dataset_scaled, seq_len)
# print(train_x.shape, test_x.shape)

# 创建张量
train_x = torch.tensor(train_x, dtype=torch.float32).unsqueeze(-1)
train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)
test_x = torch.tensor(test_x, dtype=torch.float32).unsqueeze(-1)
test_y = torch.tensor(test_y, dtype=torch.float32).unsqueeze(-1)

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)

# 定义LSTM模型
class LSTM_Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTM_Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步
        return out
    
# 训练模型
model = LSTM_Network(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
loss_list = []

# 保存每个epoch的loss
train_loss_list = []
test_loss_list = []

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for x_batch, y_batch in train_loader:
        output = model(x_batch)
        loss = criterion(output, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    # 计算test loss
    model.eval()
    with torch.no_grad():
        test_pred = model(test_x)
        test_loss = criterion(test_pred, test_y).item()
    test_loss_list.append(test_loss)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Test Loss: {test_loss:.6f}")

# 绘制Loss曲线
plt.figure(figsize=(8,5))
plt.plot(train_loss_list, label='Train Loss')
plt.plot(test_loss_list, label='Test Loss')
plt.title('Training vs Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# 预测
model.eval()
with torch.no_grad():
    pred = model(test_x)

# 反归一化
pred_price = scaler.inverse_transform(pred.numpy())
real_price = scaler.inverse_transform(test_y.numpy())

# 绘图
plt.figure(figsize=(10,5))
plt.plot(real_price, label='Real Open Price')
plt.plot(pred_price, label='Predicted Open Price')
plt.legend()
plt.title("LSTM Stock Price Prediction")
plt.show()

