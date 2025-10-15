

# Pytorch 学习

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

print("w的梯度:", w.grad)  # 如何调整w
print("b的梯度:", b.grad)  # 如何调整b
```

























