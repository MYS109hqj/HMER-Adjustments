import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile


# DenseNet-B
class Bottleneck(nn.Module):
    # Bottleneck 层是 DenseNet 的一个重要部分，包含两个卷积层。
    # 第一个是 1x1 卷积，减少通道数；第二个是 3x3 卷积，用于特征学习。
    # 通过 1x1 卷积，减少通道数后，**计算量较小，同时能提取更多有用的特征**。
    # bn（Batch Normalization批量归一化）的作用是**标准化卷积层输出的特征图**，
    # 使得每个特征图的均值接近0，标准差接近1，**进一步稳定训练过程**。
    # Dropout：如果启用 Dropout，则会在每一层后应用 Dropout，防止过拟合。
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.leaky_relu(self.bn2(self.conv2(out)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# single layer
class SingleLayer(nn.Module):
    # SingleLayer 层只包含一个 3x3 的卷积层。它没有像 Bottleneck 层那样使用 1x1 卷积减少通道数，
    # 因此计算量稍大一些，但结构更简单。
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.leaky_relu(x, inplace=True))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat((x, out), 1)
        return out


# transition layer
class Transition(nn.Module):
    # Transition 层的作用是连接 DenseBlock 和 DenseBlock 之间，并减少通道数。
    # 它使用 1x1 卷积来压缩通道数，然后应用平均池化（avg_pool2d）来下采样空间尺寸。
    def __init__(self, nChannels, nOutChannels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nOutChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True)
        return out


class DenseNet(nn.Module):
    def __init__(self, params):
        super(DenseNet, self).__init__()
        growthRate = params['densenet']['growthRate'] # 定义每个 DenseBlock 中每层的通道增加量
        reduction = params['densenet']['reduction'] # 控制在 Transition 层中减少通道数的比例
        bottleneck = params['densenet']['bottleneck'] # 是否使用 Bottleneck 层（即使用 1x1 卷积来减少通道数）
        use_dropout = params['densenet']['use_dropout'] # 是否在模型中使用 Dropout

        # DenseBlock 是 DenseNet 模型中的基本模块，每个 DenseBlock 都由多个卷积层组成。
        # 在每个 DenseBlock 内，层与层之间是密切连接的，每一层都会将其输出与前面的所有层的输出进行拼接。
        # nDenseBlocks = 16 意味着在模型中总共有 16 个这样的 DenseBlock，每个 DenseBlock 包含多个卷积层。
        # 每个 DenseBlock 都会增加通道数（根据增长率 growthRate : growthRate 是 DenseNet 中每个 DenseBlock 层增加的通道数）。
        nDenseBlocks = 16
        # nChannels 表示当前层的输出通道数
        # 2 * growthRate 是模型的初始通道数，通常在 DenseNet 中，初始通道数是 2 * growthRate（可以理解为网络一开始的特征图宽度）
        nChannels = 2 * growthRate
        self.conv1 = nn.Conv2d(params['encoder']['input_channel'], nChannels, kernel_size=7, padding=3, stride=2, bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        # 这行代码更新 nChannels，它表示 DenseBlock 完成后，输出通道数的变化。
        # 因为每个 DenseBlock 中有 nDenseBlocks 层，每层都会增加 growthRate 个通道，因此输出的通道数应该增加 nDenseBlocks * growthRate。
        nChannels += nDenseBlocks * growthRate
        # nChannels 是 DenseBlock 完成后的输出通道数，而 reduction 是一个控制比例的因子，用于减少通道数。
        # reduction 通常小于 1（比如 0.5），意味着我们希望将通道数减小到原来的 reduction 倍。
        # math.floor 函数用于取整，使得最终的通道数是一个整数。
        nOutChannels = int(math.floor(nChannels * reduction))
        # 这行代码创建了一个 Transition 层，它将 nChannels 转换为 nOutChannels，并进行下采样。
        # Transition 层通过 1x1 卷积减少通道数，并通过池化操作减小空间尺寸。
        self.trans1 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        nOutChannels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, nOutChannels, use_dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, use_dropout):
        # 用于构建一个DenseBlock
        layers = []
        # 根据传入的参数，创建多个 Bottleneck 或 SingleLayer 层，并将它们连接成一个 Sequential 模块。
        # 这个模块会在模型的前向传播过程中作为一个整体使用。
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)

        return out