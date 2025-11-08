# 读取数据 - 处理数据 - 导入模型- 得到结果
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# 参数设置
seq_len = 50
epochs = 50
input_size=1
hidden_size=64
num_layers=2
output_size=1

# 读取数据
df = pd.read_csv('SH600519.csv')
df['data'] = pd.to_datetime(df['date']) # 转换日期格式
df = df.sort_values('date').reset_index(drop=True) # 建立新索引

# 数据处理
dataset = df['open'].values # 取开盘价作为数据集,转为numpy数组
dataset_2d = dataset.reshape(-1, 1) # 转换为二维数组,用于归一化
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset_2d)
# print(dataset_scaled.ndim)

# 划分测试集和训练集
train_dataset_scaled = dataset_scaled[:2126]
test_dataset_scaled = dataset_scaled[2126-seq_len:]

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

train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

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

for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        output = model(x_batch)
        loss = criterion(output, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

model.eval()
with torch.no_grad():
    pred = model(test_x).numpy()

# 反归一化预测结果
pred_real = scaler.inverse_transform(pred)
y_real = scaler.inverse_transform(test_y.numpy())

# 可视化
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.plot(y_real, label='Real Open Price')
plt.plot(pred_real, label='Predicted Open Price')
plt.legend()
plt.title("LSTM Stock Price Prediction (Open)")
plt.show()

