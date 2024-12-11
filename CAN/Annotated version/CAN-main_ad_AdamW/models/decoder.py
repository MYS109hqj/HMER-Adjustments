import torch
import torch.nn as nn
from models.attention import Attention
import math
import numpy as np
from counting_utils import gen_counting_label


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats # num_pos_feats: 位置嵌入的特征维度数量
        self.temperature = temperature # 用于缩放 dim_t 的温度参数
        self.normalize = normalize # 布尔值，决定是否对位置嵌入进行归一化处理
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        # y_embed: 沿着 height（y轴）方向对掩码 mask 进行累加（即 cumsum），得到每个像素行的累积索引。cumsum(1) 计算的是每一行的位置索引。
        # x_embed: 类似地，x_embed 是沿着 width（x轴）方向对掩码 mask 进行累加得到的，表示每一列的位置索引。
        # print(f"mask.shape: {mask.shape}")  # Debug: 输出 mask 的形状
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        
        # Debug: 输出 y_embed 和 x_embed 的形状
        # print(f"y_embed.shape: {y_embed.shape}")
        # print(f"x_embed.shape: {x_embed.shape}")
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        # dim_t: 位置编码的缩放因子，用于生成每个位置的特征向量。dim_t 是一个长度为 num_pos_feats 的向量，表示每个特征的衰减尺度。
        # torch.arange(self.num_pos_feats) 会生成一个从0到 num_pos_feats-1 的数组。
        # 接下来的表达式计算每个位置的缩放因子，其中 self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) 是位置编码中衰减的关键公式。
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Debug: 输出 dim_t 的值
        # print(f"dim_t: {dim_t}")

        # pos_x 和 pos_y: 分别计算水平和垂直方向的位置编码。x_embed 和 y_embed 被除以 dim_t 来缩放每个位置的编码，生成位置编码向量。
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # Debug: 输出 pos_x 和 pos_y 在 sin/cos 操作前的形状
        # print(f"pos_x.shape before sin/cos: {pos_x.shape}")
        # print(f"pos_y.shape before sin/cos: {pos_y.shape}")

        # pos_x 和 pos_y（继续）: 对每个位置编码的偶数索引部分应用 sin，对奇数索引部分应用 cos，然后将其堆叠在一起，得到完整的正弦余弦编码。
        # 0::2 表示取偶数索引，1::2 表示取奇数索引。
        # sin 和 cos 使得位置编码可以在不同维度上具有周期性变化，从而帮助模型理解位置的相对关系。
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        
        # Debug: 输出 pos_x 和 pos_y 在 sin/cos 操作后的形状
        # print(f"pos_x.shape after sin/cos: {pos_x.shape}")
        # print(f"pos_y.shape after sin/cos: {pos_y.shape}")

        # 拼接 pos_y 和 pos_x，最终生成位置嵌入，形状为 [batch, num_pos_feats * 2, height, width]
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        
        # Debug: 输出最终生成的 pos 形状
        # print(f"pos.shape: {pos.shape}")
        
        return pos


class AttDecoder(nn.Module):
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim'] # 注意力机制的维度
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.word_num = params['word_num']
        self.counting_num = params['counting_decoder']['out_channel']

        """经过cnn后 长宽与原始尺寸比缩小的比例"""
        self.ratio = params['densenet']['ratio']

        # init hidden state 将CNN特征图转换为GRU的初始隐藏状态
        self.init_weight = nn.Linear(self.out_channel, self.hidden_size)
        # word embedding 将输入的标签（如符号ID）映射到嵌入向量
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        # word gru 处理嵌入向量和隐藏状态，用于逐步生成序列
        self.word_input_gru = nn.GRUCell(self.input_size, self.hidden_size)
        # attention 基于输入特征图和隐藏状态计算注意力权重，并生成上下文向量
        self.word_attention = Attention(params)
        self.encoder_feature_conv = nn.Conv2d(self.out_channel, self.attention_dim,
                                              kernel_size=params['attention']['word_conv_kernel'],
                                              padding=params['attention']['word_conv_kernel']//2)

        self.word_state_weight = nn.Linear(self.hidden_size, self.hidden_size)
        self.word_embedding_weight = nn.Linear(self.input_size, self.hidden_size)
        self.word_context_weight = nn.Linear(self.out_channel, self.hidden_size)
        self.counting_context_weight = nn.Linear(self.counting_num, self.hidden_size)
        self.word_convert = nn.Linear(self.hidden_size, self.word_num) # 将最终的状态转换为预测的符号分布

        if params['dropout']:
            self.dropout = nn.Dropout(params['dropout_ratio']) # 在训练时随机丢弃部分神经元，防止过拟合

    def forward(self, cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=True):
        batch_size, num_steps = labels.shape
        height, width = cnn_features.shape[2:]
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]

        word_alpha_sum = torch.zeros((batch_size, 1, height, width)).to(device=self.device)
        word_alphas = torch.zeros((batch_size, num_steps, height, width)).to(device=self.device)
        # 隐藏状态初始化
        # 使用 CNN 提取的特征图初始化 GRU 的隐藏状态
        # print(f"cnn_features: {cnn_features.shape}")
        hidden = self.init_hidden(cnn_features, images_mask)
        counting_context_weighted = self.counting_context_weight(counting_preds)
        # print(f"cnn_features2: {cnn_features.shape}")
        cnn_features_trans = self.encoder_feature_conv(cnn_features)
        # print(f"cnn_features3: {cnn_features.shape}")
        # 给特征图添加位置信息，增强空间感知能力
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, images_mask[:,0,:,:])
        cnn_features_trans = cnn_features_trans + pos

        if is_train:
            for i in range(num_steps):
                word_embedding = self.embedding(labels[:, i-1]) if i else self.embedding(torch.ones([batch_size]).long().to(self.device))
                # 更新隐藏状态
                # 通过 GRU单元 self.word_input_gru 将嵌入向量和前一步隐藏状态结合，更新隐藏状态 hidden。
                hidden = self.word_input_gru(word_embedding, hidden)
                # self.word_attention 计算：
                # 上下文向量 word_context_vec：聚焦于当前时间步最相关的图像区域。
                # 注意力权重 word_alpha：表示对每个图像区域的关注程度。
                # 累加的注意力权重 word_alpha_sum：用于抑制过度关注同一区域。
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)

                # 线性变换：
                # 对 隐藏状态、嵌入向量 和 上下文向量 进行线性变换，使它们映射到相同的维度，便于相加                                                     
                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)
                # 合并信息：
                # 将变换后的 隐藏状态、嵌入向量 和 上下文向量 相加
                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted
                # 生成预测概率
                word_prob = self.word_convert(word_out_state)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
        else:
            word_embedding = self.embedding(torch.ones([batch_size]).long().to(device=self.device))
            for i in range(num_steps):
                hidden = self.word_input_gru(word_embedding, hidden)
                word_context_vec, word_alpha, word_alpha_sum = self.word_attention(cnn_features, cnn_features_trans, hidden,
                                                                                   word_alpha_sum, images_mask)

                current_state = self.word_state_weight(hidden)
                word_weighted_embedding = self.word_embedding_weight(word_embedding)
                word_context_weighted = self.word_context_weight(word_context_vec)

                if self.params['dropout']:
                    word_out_state = self.dropout(current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted)
                else:
                    word_out_state = current_state + word_weighted_embedding + word_context_weighted + counting_context_weighted

                word_prob = self.word_convert(word_out_state)
                _, word = word_prob.max(1)
                word_embedding = self.embedding(word)
                word_probs[:, i] = word_prob
                word_alphas[:, i] = word_alpha
        return word_probs, word_alphas

    def init_hidden(self, features, feature_mask):
        # 通过对特征图加权平均，得到初始特征表示，并通过全连接层转换为 GRU 隐藏状态。
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        average = self.init_weight(average)
        return torch.tanh(average)
