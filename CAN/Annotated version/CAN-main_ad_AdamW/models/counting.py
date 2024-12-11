import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAtt(nn.Module):
    # ChannelAtt（通道注意力机制）用于自适应地调整输入特征图的通道权重，
    # 通过使用全局平均池化和一个简单的全连接网络来生成每个通道的权重。
    # 这种方式可以使模型关注更重要的通道，从而提高模型的性能。
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        # 使用全局平均池化对输入特征图的每个通道进行压缩，输出为1x1大小的特征图。
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 一个包含两个全连接层的网络，第一个全连接层将通道数减少到原来的1/reduction，
        # 第二个全连接层恢复通道数，最后通过Sigmoid激活函数得到每个通道的权重。
        self.fc = nn.Sequential(
                nn.Linear(channel, channel//reduction), # 缩小通道数
                nn.ReLU(), # 激活函数
                nn.Linear(channel//reduction, channel), # 恢复通道数
                nn.Sigmoid()) # 输出权重

    def forward(self, x):
        b, c, _, _ = x.size() # 获取输入的batch size、通道数、高度和宽度
        # 对输入特征图进行全局平均池化，得到每个通道的全局描述
        y = self.avg_pool(x).view(b, c)
        # 通过全连接网络生成通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # 对输入特征图进行通道加权操作
        return x * y


class CountingDecoder(nn.Module):
    # CountingDecoder（计数解码器）用于从输入特征图中预测计数值。
    # 它通过卷积层提取特征，再通过通道注意力机制调整通道的重要性，最后通过卷积进行计数预测。
    def __init__(self, in_channel, out_channel, kernel_size):
        super(CountingDecoder, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        # 转换层，用于将输入的特征图通过卷积映射到一个新的空间（512个通道）
        self.trans_layer = nn.Sequential(
            nn.Conv2d(self.in_channel, 512, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(512)) # 使用批量归一化稳定训练过程
        # 通道注意力机制，用于调整每个通道的权重
        self.channel_att = ChannelAtt(512, 16)
        # 预测层，将通道数从512压缩到目标输出通道数
        self.pred_layer = nn.Sequential(
            nn.Conv2d(512, self.out_channel, kernel_size=1, bias=False),
            nn.Sigmoid()) # 输出预测结果，使用Sigmoid激活函数进行归一化

    def forward(self, x, mask):
        b, c, h, w = x.size()
        x = self.trans_layer(x)
        # 使用通道注意力机制调整特征图的通道权重
        x = self.channel_att(x)
        # 通过预测层生成计数结果
        x = self.pred_layer(x)
        # 如果提供了mask（用于区域限制或无效区域），则按mask进行元素级乘法
        if mask is not None:
            x = x * mask
        # 展平特征图，用于后续的计数计算
        x = x.view(b, self.out_channel, -1)
        # 对展平后的特征图在最后一个维度上求和，得到每个样本的总计数值
        x1 = torch.sum(x, dim=-1)
        # 返回两个结果：x1是总计数，x是未经过展平的特征图
        return x1, x.view(b, self.out_channel, h, w)
