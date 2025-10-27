

# Pytorch - 神经网络 学习

## <font color=708090>一、Tensor（张量） </font> 

**<font size=4>概念：是一个三维及以上的数组 </font>**

**<font size=4>创建： `torch.tensor()` </font> ** 

```python
# 张量的创建
import torch
import numpy as np 

data = [[1,2], [3,4]]

# 直接创建
tensor_x = torch.tensor(data)

# 通过NumPy数组创建
np_data = np.array(data)
tensor_np = torch.from_numpy(np_data)

# 从张量本身创建
tensor_ones = torch.ones_like(tensor_x)  # 创建与x_data形状相同的全1张量
tensor_rand = torch.rand_like(tensor_x, dtype=torch.float)  # 创建与x_data形状相同的随机张量
```



**<font size=4>基本参数 </font>**

* `shape`：形状
* `dtype`：数据类型
* `device`：存储设备





## <font color=708090>二、处理数据集 </font>

<font size=4>**解决**</font>：数据集按**批次（batch）、打乱（shuffle）数据、加速加载（多线程）**的步骤 

<font size=4>**两大工具**</font>

* `Dataset`：定义 **数据是什么**，以及 **怎么读取一条数据**
* ` DataLoader `： 负责 **按批次取数据**、**打乱数据**、**并行加载** 



### 1. `Dataset`（自定义的数据集对象）

* `__init__`：实例化 Dataset 对象时，会运行一次 __init__ 函数。
* `__len__`：返回数据集中的样本数量
* `__getitem__`： 函数从给定索引处的数据集中加载并返回一个样本`idx`

```python
# 实现一个dataset
from torch.utils.data import Dataset

class Mydataset(Dataset):
    def __init__(self):
        self.data = [1,2,3,4,5]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
dataset = Mydataset()	# 自动调用 __init__ 
print(dataset.data)		# 返回 [1,2,3,4,5]
print(len(dataset))		# 实际上调用 dataset.__len__()，返回 5
print(dataset[2])		# 实际上调用 dataset.__getitem__(2)，返回 3
```



### 2. `DataLoader`

- `dataset`：要加载的数据集
- `batch_size`：样本数量
- `shuffle`：是否打乱数据

```python
# 将训练数据打乱并分批次
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        self.data = [1, 2, 3, 4, 5, 6]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 可以多次循环，增加训练轮数
for batch in dataloader:
    print(batch)
'''
输出：
tensor([4, 2])
tensor([3, 6])
tensor([1, 5])
```





## <font color=708090>三、自动求导 & torch.autograd </font>

> "在训练神经网络时，最常用的算法是 **反向传播**。在该算法中，参数（模型权重）根据损失函数相对于给定参数的**梯度**进行调整。"
>
> **反向传播**：即分析预测错误的原因，以便更新权重（从输出层开始往回分析：输出层 - 隐藏层 - 输入层）
>
> PyTorch 有一个内置的微分引擎，称为`torch.autograd`。它支持自动计算任何计算图的梯度。



### 1. 自动求导

```python
# 让Pytorch实现自动求导
import torch

x = torch.tensor(2.0, requires_grad=True) 	# 创建一个值为2，需要梯度的张量
y = x**2 + 3*x + 1			# 构建计算图 y = x² + 3x + 1
print("y 的值:", y)  		   # 输出 y 的值: tensor(11., grad_fn=<AddBackward0>)	后半部分表示是Pytorch在追踪，这个张量是通过加法操作得到的

# 反向传播，计算 dy/dx
y.backward()	

# 求出梯度，输出7
print("x 的梯度:", x.grad) 
```



### 2. 应用 - 计算梯度

我们需要计算 **损失函数loss** 对 **参数（w和b）** 的导数 ，以此调整

```python
# 在神经网络中应用
import torch

# 假设神经网络的一个神经元
w = torch.tensor(1.0, requires_grad=True)  # 权重
b = torch.tensor(0.5, requires_grad=True)  # 偏置
x = torch.tensor(2.0)                      # 输入

# 前向传播
y = w * x + b          # y = 1.0×2.0 + 0.5 = 2.5
loss = (y - 3.0)**2    # 假设目标是3.0，损失=(2.5-3.0)²=0.25

# 反向传播
loss.backward()

print("w的梯度:", w.grad)  # 如何调整w，输出 w的梯度: tensor(-2.)
print("b的梯度:", b.grad)  # 如何调整b，输出 b的梯度: tensor(-1.)
```



### 3. 禁用跟踪（示例：梯度下降法 找出 y = x² 的最小值）

> **<font face = '微软雅黑' color= DC143C size=4>梯度下降法：x_new = x_old - η * ∇J(w) </font>**
>
> => 新参数 = 旧参数 - 步长(学习率) × 梯度（ x.grad ）

```python
# 梯度下降示例
import torch

# 1. 随机起点（假设我们在x=5的位置）
x = torch.tensor(5.0, requires_grad=True)
learning_rate = 0.1  # 学习率（步长）

print("开始梯度下降：")
for epoch in range(10):  # 迭代10次
    # 前向传播：计算函数值
    y = x**2
    
    # 反向传播：计算梯度（导数）
    y.backward()
    
    # 梯度下降：更新参数
    with torch.no_grad():  # 更新时不追踪梯度
        x -= learning_rate * x.grad  # x_new = x_old - η × gradient
    
    # 清空梯度
    x.grad.zero_()
    
    print(f"第{epoch+1}次: x = {x:.3f}, y = {y:.3f}, 梯度 = {x.grad}")
```

- **如果不禁用跟踪，在经过`x -= learning_rate * x.grad`之后再求梯度时，就会按照此式计算微分**





## <font color=708090>四、神经网络 </font>

**<font size=4>使用`torch.nn`包构建神经网络</font> **

### 1. 单层感知机(MLP) - 解决线性问题

```python
# 示例：使用神经网络学习 y = 2x 线性关系
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__() # 调用父类（nn.Module）的初始化方法
        self.fc1 = nn.Linear(1, 1)  # 输入1维，输出1维

    def forward(self, x):
        return self.fc1(x)  # 前向传播：线性变换

# 实例化网络
model = SimpleNN()
print(model)

# 创建数据集（数据准备，网络学习的目标）
x_data = torch.tensor([[1.0], [2.0], [3.0]])  # 输入
y_data = torch.tensor([[2.0], [4.0], [6.0]])  # 目标输出

# 定义损失函数与优化器（优化器作用：控制更新参数）
criterion = nn.MSELoss()  # 均方误差损失 MSE = (预测值 - 真实值)² 的平均值
optimizer = optim.SGD(model.parameters(), lr=0.01)  # SGD优化器（负责实际优化参数步骤）

# 训练模型
for epoch in range(1000):  # 训练1000次
    model.train()  # 设置模型为训练模式

    # 前向传播
    y_pred = model(x_data)

    # 计算损失
    loss = criterion(y_pred, y_data)

    # 反向传播
    optimizer.zero_grad()  # 清空上一步的梯度
    loss.backward()  # 计算梯度

    # 更新参数
    optimizer.step()  # 使用优化器更新模型参数

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}", "w:", model.fc1.weight.data, "b:", model.fc1.bias.data) 


'''
输出：
Epoch 0, Loss: 26.642709732055664 w: tensor([[-0.5248]]) b: tensor([0.9390])
Epoch 100, Loss: 0.26270681619644165 w: tensor([[1.4061]]) b: tensor([1.3500])
Epoch 200, Loss: 0.16233696043491364 w: tensor([[1.5332]]) b: tensor([1.0612])
Epoch 300, Loss: 0.1003144308924675 w: tensor([[1.6330]]) b: tensor([0.8342])
Epoch 400, Loss: 0.061988234519958496 w: tensor([[1.7115]]) b: tensor([0.6558])
Epoch 500, Loss: 0.03830496594309807 w: tensor([[1.7732]]) b: tensor([0.5155])
Epoch 600, Loss: 0.023670120164752007 w: tensor([[1.8217]]) b: tensor([0.4052])
Epoch 700, Loss: 0.014626693911850452 w: tensor([[1.8599]]) b: tensor([0.3185])
Epoch 800, Loss: 0.009038408286869526 w: tensor([[1.8898]]) b: tensor([0.2504])
Epoch 900, Loss: 0.005585184786468744 w: tensor([[1.9134]]) b: tensor([0.1968])
可以观察到，参数 w 逐渐靠近2，参数 b 逐渐靠近0
```

**整个神经网络的流程：**

* 定义模型结构（nn.Module）
* 前向传播（forward）
* 损失函数（MSELoss）
* 优化器（SGD）
* 训练循环（梯度下降） 



### 2. 多层感知机 - 解决非线性问题

**多个神经元：多个线性函数的融合**

**需要：隐藏层 + 激活函数**

- <font color=A52A2A size=4>**以下是关于神经网络自我理解的图片 （图源：李宏毅 - 机器学习）**</font>

**1. 可以将每个神经元理解为一个线性函数（所以单层感知机只能解决线性问题），多个神经元相当于多个线性函数的融合，即可表达非线性函数**

<img src="E:\研\学习记录\Study_record\图片存储\pytorch-1.jpg" alt="pytorch-1" style="zoom:30%;" />

**2. 可以将曲线化为多段直线，比如泰勒展开**

<img src="E:\研\学习记录\Study_record\图片存储\pytorch-2.jpg" alt="pytorch-2" style="zoom:30%;" />

**3. 激活函数：可以使线性变非线性 或 限制函数范围，达到更好的模拟效果(图为 Sigmoid函数)**

<img src="E:\研\学习记录\Study_record\图片存储\pytorch-3.jpg" alt="pytorch-3" style="zoom:30%;" />

```python
# 示例：使用神经网络学习 y = x² 关系
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个两层神经网络
class TwoLayerNN(nn.Module):
    def __init__(self):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(1, 10)  # 隐藏层：10个神经元
        self.relu = nn.ReLU()        # 激活函数
        self.fc2 = nn.Linear(10, 1)  # 输出层

    def forward(self, x):
        x = self.fc1(x)   # 线性变换1
        x = self.relu(x)  # 激活函数 relu: max(0, x)
        x = self.fc2(x)   # 线性变换2
        return x

# 初始化模型
model = TwoLayerNN()

# 生成数据，例如 y = x² 的曲线
x_data = torch.linspace(-3, 3, 100).unsqueeze(1) # 创建从-3到3的100个等间距数字，并变为二维张量
y_data = x_data.pow(2)

# 损失函数 & 优化器
criterion = nn.MSELoss() # MSE = (预测值 - 真实值)² 的平均值
optimizer = optim.Adam(model.parameters(), lr=0.01) # 比 SGD 更快的优化器

# 开始训练
for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

'''
输出：
Epoch 0, Loss: 14.724333763122559
Epoch 200, Loss: 0.3091835379600525
Epoch 400, Loss: 0.062044452875852585
Epoch 600, Loss: 0.022268002852797508
Epoch 800, Loss: 0.012258048169314861
Epoch 1000, Loss: 0.008690610527992249
Epoch 1200, Loss: 0.0069376761093735695
Epoch 1400, Loss: 0.006143274717032909
Epoch 1600, Loss: 0.005260563921183348
Epoch 1800, Loss: 0.004814780782908201
```



### 3. 用Python定义一个神经网络

```python
import random
import numpy as np

class Network(object):

    # 定义模型  
    # 输入：net = network.Network([784, 30, 10])
    def __init__(self, sizes):
        self.num_layers = len(sizes)    # 网络层数
        self.sizes = sizes              # 每层神经元个数
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]    # 偏置初始化
        self.weights = [np.random.randn(y, x)                          
                        for x, y in zip(sizes[:-1], sizes[1:])]     # 权重初始化
        
        # sizes[1:]：从索引1开始切片，这里指 [30, 10]；  sizes[0]指 索引为0的元素（每层神经元的个数）
        # zip(sizes[:-1], sizes[1:])：将前一层和后一层的神经元个数配对，如(784,30),(30,10)
        # np.random.randn(y, x)：生成y行x列的矩阵


    # 前向传播
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b) # 通过激活函数
        return a                        # 返回输出层的激活值


    # SGD 优化器
    # 实现了神经网络的随机梯度下降，使用小批量数据来更新网络参数
    # 输入：net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):  # epochs：训练轮数； mini_batch_size：小批量数据大小； eta：学习率
        if test_data: n_test = len(test_data) # 记录测试数据集大小
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)   # 打乱训练数据（可避免局部最优）
            mini_batches = [                
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)] # 将训练数据划分为小批量，列表里是具体数据
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta) # 处理每个小批量，更新参数
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))  # 如果有测试数据：显示准确率
            else:
                print ("Epoch {0} complete".format(j))    # 如果无测试数据：只显示epoch完成信息

        # xrange(epochs) 生成 [0, 1, 2, ..., epochs-1]
        # xrange(起始, 结束, 步长)
        # format()：字符串格式化的方法，用于将变量值插入到字符串中的特定位置


    # 小批量数据更新
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # 初始化偏置梯度
        nabla_w = [np.zeros(w.shape) for w in self.weights] # 初始化权重梯度
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y) # 计算梯度（函数见下）
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # 梯度累加
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # 梯度累加
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)] # 更新权重
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)] # 更新偏置
        
        # np.zeros(b.shape) ：创建一个与 b 形状相同的全零矩阵
          

    # 反向传播算法（x：样本输入； y：期望输出）
    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向
        activation = x
        activations = [x] # 存储每层的激活值
        zs = [] # 记录每层加权输入
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # 矩阵运算 w · x + b
            zs.append(z)
            activation = sigmoid(z)     # 通过激活函数，得到激活值
            activations.append(activation) # 记录激活值
        # 后向传递
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) # 计算输出层误差
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
        # np.dot(): 矩阵乘法


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        r"""Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

```



























