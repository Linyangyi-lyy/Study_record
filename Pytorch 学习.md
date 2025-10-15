# Pytorch 学习

## <font color=" grey">一、Tensor（张量） </font> 

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





## <font color=" grey">二、处理数据集 </font>

<font size=4>**解决**</font>：数据集按**批次（batch）、打乱（shuffle）数据、加速加载（多线程）**的步骤 

<font size=4>**两大工具**</font>

* `Dataset`：定义 **数据是什么**，以及 **怎么读取一条数据**
* ` DataLoader `： 负责 **按批次取数据**、**打乱数据**、**并行加载** 



### 1. `Dataset`（自定义的数据集对象）

```python
# 实现一个dataset

```









































