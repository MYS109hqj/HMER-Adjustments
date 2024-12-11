import os
import cv2
import yaml
import math
import torch
import numpy as np
from difflib import SequenceMatcher


def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('尝试UTF-8编码....')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    if not params['experiment']:
        print('实验名不能为空!')
        exit(-1)
    if not params['train_image_path']:
        print('训练图片路径不能为空！')
        exit(-1)
    if not params['train_label_path']:
        print('训练label路径不能为空！')
        exit(-1)
    if not params['word_path']:
        print('word dict路径不能为空！')
        exit(-1)
    if 'train_parts' not in params:
        params['train_parts'] = 1
    if 'valid_parts' not in params:
        params['valid_parts'] = 1
    if 'valid_start' not in params:
        params['valid_start'] = 0
    if 'word_conv_kernel' not in params['attention']:
        params['attention']['word_conv_kernel'] = 1
    return params


def update_lr(optimizer, current_epoch, current_step, steps, epochs, initial_lr, warmup_epochs=5):
    """动态调整学习率，支持 warmup 和余弦退火策略。

    参数:
        - optimizer: 优化器实例。
        - current_epoch: 当前训练的 epoch。
        - current_step: 当前 batch 的步骤数。
        - steps: 每个 epoch 的 batch 数量。
        - epochs: 总训练 epoch 数。
        - initial_lr: 初始学习率。
        - warmup_epochs: warmup 阶段的 epoch 数量。
    """
    total_steps = epochs * steps
    current_total_step = current_epoch * steps + current_step

    if current_epoch < warmup_epochs:
        # 线性增加学习率，确保训练稳定开始
        new_lr = initial_lr * (current_total_step + 1) / (warmup_epochs * steps)
    else:
        # 使用余弦退火策略调整学习率
        new_lr = 0.5 * initial_lr * (1 + math.cos(math.pi * current_total_step / total_steps))

    # 更新优化器的学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr



def save_checkpoint(model, optimizer, word_score, ExpRate_score, epoch, optimizer_save=True, path='checkpoints', multi_gpu=False, local_rank=0):
    """保存模型和优化器状态。

    参数:
        - model: 需要保存的模型。
        - optimizer: 优化器实例。
        - word_score: 当前字词准确率。
        - ExpRate_score: 当前表达式识别率。
        - epoch: 当前训练的 epoch。
        - optimizer_save: 是否保存优化器状态。
        - path: 保存路径。
    """
    checkpoint_dir = os.path.join(path, model.name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = f'{checkpoint_dir}/{model.name}_WordRate-{word_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'
    
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'word_score': word_score,
        'ExpRate_score': ExpRate_score
    }

    if optimizer_save:
        state['optimizer'] = optimizer.state_dict()

    torch.save(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename



def load_checkpoint(model, optimizer, path):
    """加载模型和优化器状态。

    参数:
        - model: 需要加载的模型。
        - optimizer: 优化器实例，为 None 时不加载优化器状态。
        - path: 检查点路径。
    """
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])
    print(f'Model loaded from {path}')

    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
        print('Optimizer state loaded.')
    else:
        print('No optimizer state found in checkpoint.')

    return state.get('epoch', 0), state.get('word_score', 0), state.get('ExpRate_score', 0)



class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num


def cal_score(word_probs, word_label, mask):
    line_right = 0
    if word_probs is not None:
        _, word_pred = word_probs.max(2)
    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
              for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    
    batch_size = len(word_scores)
    for i in range(batch_size):
        if word_scores[i] == 1:
            line_right += 1

    ExpRate = line_right / batch_size
    word_scores = np.mean(word_scores)
    return word_scores, ExpRate


def draw_attention_map(image, attention):
    h, w = image.shape
    attention = cv2.resize(attention, (w, h))
    attention_heatmap = ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))*255).astype(np.uint8)
    attention_heatmap = cv2.applyColorMap(attention_heatmap, cv2.COLORMAP_JET)
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    attention_map = cv2.addWeighted(attention_heatmap, 0.4, image_new, 0.6, 0.)
    return attention_map


def draw_counting_map(image, counting_attention):
    h, w = image.shape
    counting_attention = torch.clamp(counting_attention, 0.0, 1.0).numpy()
    counting_attention = cv2.resize(counting_attention, (w, h))
    counting_attention_heatmap = (counting_attention * 255).astype(np.uint8)
    counting_attention_heatmap = cv2.applyColorMap(counting_attention_heatmap, cv2.COLORMAP_JET)
    image_new = np.stack((image, image, image), axis=-1).astype(np.uint8)
    counting_map = cv2.addWeighted(counting_attention_heatmap, 0.4, image_new, 0.6, 0.)
    return counting_map


def cal_distance(word1, word2):
    m = len(word1)
    n = len(word2)
    if m*n == 0:
        return m+n
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            a = dp[i-1][j] + 1
            b = dp[i][j-1] + 1
            c = dp[i-1][j-1]
            if word1[i-1] != word2[j-1]:
                c += 1
            dp[i][j] = min(a, b, c)
    return dp[m][n]


def compute_edit_distance(prediction, label):
    prediction = prediction.strip().split(' ')
    label = label.strip().split(' ')
    distance = cal_distance(prediction, label)
    return distance
    