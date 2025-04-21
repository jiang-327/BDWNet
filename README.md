# 木材缺陷检测

这个项目使用深度学习方法进行木材缺陷检测和分割。

## 项目结构

```
├── configs/               # 配置文件
│   └── config.py         # 参数配置
├── data/                 # 数据目录
│   ├── train/           # 训练集
│   ├── val/             # 验证集
│   └── test/            # 测试集
├── models/              # 模型定义
│   ├── loss.py         # 损失函数定义
│   └── network.py      # 网络结构定义
├── utils/              # 工具函数
│   ├── dataset.py      # 数据集加载和预处理
│   └── metrics.py      # 评价指标计算
├── train.py           # 训练脚本
└── test.py            # 测试脚本
```

## 运行步骤

### 1. 环境准备

确保安装了以下依赖：
```
pip install torch torchvision numpy opencv-python matplotlib albumentations scikit-learn tqdm
```

### 2. 数据准备

请将数据集按照以下结构放置：
```
data/
├── train/              # 训练集图像和掩码
├── val/                # 验证集图像和掩码
└── test/               # 测试集图像和掩码
```

每个目录下应包含：
- images/：存放图像文件
- masks/：存放对应的掩码文件

### 3. 训练模型

运行以下命令开始训练：
```
python train.py
```

### 4. 测试模型

训练完成后，运行以下命令测试模型性能：
```
python test.py
```

## 主要文件说明

- `models/network.py`：WoodDefectBD网络模型定义
- `models/loss.py`：损失函数定义
- `utils/dataset.py`：数据加载和预处理
- `utils/metrics.py`：模型评估指标
- `configs/config.py`：训练和模型参数配置
- `train.py`：训练脚本
- `test.py`：测试脚本 