import torch
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler

def collate_fn(batch_images):
    # 用于对批处理数据进行填充
    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length)).long(), torch.zeros((len(proper_items), max_length))

    for i in range(len(proper_items)):
        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1
        l = proper_items[i][1].shape[0]
        labels[i][:l] = proper_items[i][1]
        labels_masks[i][:l] = 1
    return images, image_masks, labels, labels_masks

class HMERDataset(Dataset):
    def __init__(self, params, image_path, is_train=True):
        super(HMERDataset, self).__init__()
        self.image_path = image_path
        self.is_train = is_train # 241128版本，is_train并没有起作用
        self.params = params
        self.buckets = params.get('buckets', None)

        # 获取所有 .npy 文件路径
        self.npy_files = [file for file in os.listdir(image_path) if file.endswith('.npy')]
        self.n_samples_list = self._prepare_samples()

        # 构建字符集（如果符号集固定）
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(set("0123456789abcdefghijklmnopqrstuvwxyz+-*/^=<>(){}[]")))}

    def _prepare_samples(self):
        """计算每个 .npy 文件的样本索引范围。"""
        n_samples_list = [0]
        for file in self.npy_files:
            data = np.load(os.path.join(self.image_path, file), allow_pickle=True)
            n_samples_list.append(n_samples_list[-1] + len(data))
        return n_samples_list
    
    def _find_npy_file(self, idx):
        """确定当前索引所在的 .npy 文件。"""
        for i in range(1, len(self.n_samples_list)):
            if idx < self.n_samples_list[i]:
                return self.npy_files[i - 1], idx - self.n_samples_list[i - 1]
    
    def __len__(self):
        return self.n_samples_list[-1]
        # return 20

    def _encode_label(self, label):
        """将字符串标签转换为索引列表，并附加 'eos' 结束符。"""
        return [self.char_to_idx.get(char, 0) for char in label] + [len(self.char_to_idx) + 1]  # 'eos' 编码

    def __getitem__(self, idx):
        file_name, idx_in_file = self._find_npy_file(idx)
        data = np.load(os.path.join(self.image_path, file_name), allow_pickle=True)
        sample = data[idx_in_file]

        # 读取图像并预处理
        image = torch.tensor(sample['image']).float() / 255.0
        image = image.permute(2, 0, 1)  # 调整通道维度 (C, H, W)
        
        # 处理标签
        label = sample['label']
        encoded_label = self._encode_label(label)
        label_tensor = torch.LongTensor(encoded_label)
        
        return image, label_tensor

# collate_fn 保持不变
collate_fn_dict = {
    'collate_fn': collate_fn
}

# get_crohme_dataset 需要稍作调整，移除 words 相关内容：
def get_crohme_dataset(params):
    print(f"训练数据路径 images: {params['train_image_path']}")
    print(f"验证数据路径 images: {params['eval_image_path']}")

    train_dataset = HMERDataset(params, params['train_image_path'], is_train=True)
    eval_dataset = HMERDataset(params, params['eval_image_path'], is_train=False)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn_dict['collate_fn'], pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                             num_workers=params['workers'], collate_fn=collate_fn_dict['collate_fn'], pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)} ')
    return train_loader, eval_loader


collate_fn_dict = {
    'collate_fn': collate_fn
}


