CAN模型原github仓库地址：
https://github.com/LBH1024/CAN
When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition
This is the official pytorch implementation of [CAN](https://arxiv.org/abs/2207.11463) (ECCV'2022). 
>*Bohan Li, Ye Yuan, Dingkang Liang, Xiao Liu, Zhilong Ji, Jinfeng Bai, Wenyu Liu, Xiang Bai*


对于CAN模型，由于本人需要模型直接读入经过预处理后的.npy文件，因此需要对原有dataset.py的数据输入部分进行修改。
```markdown
我的.npy文件数据格式定义：
对于.npy 后缀的文件，每个文件包含一个 list，其中的每一项是一个样本对应
的 dict，：格式如下
{
'ID': 00000,
'label': "x ^ { 2 } - 1 3 x + 3 6 < 0",
'image': np.ndarray of shape (width, height, RGB)
}
```
而公开代码中，接受的形式是images+labels，与我使用的不同。这里只解释我做出的主要修改。
我将`image_path`重定义为一系列.npy文件所处的文件夹的路径。将`label_path`的定义删除。文件路径从params中读取（也就是从config中读取）。
```python
def __init__(self, params, image_path, is_train=True):
    super(HMERDataset, self).__init__()
    self.image_path = image_path
    self.is_train = is_train
    self.params = params
    self.buckets = params.get('buckets', None)

    # 获取所有 .npy 文件路径
    self.npy_files = [file for file in os.listdir(image_path) if file.endswith('.npy')]
    self.n_samples_list = self._prepare_samples()

    ......
```
由于.npy中的'label'直接是目标标签，而非对应各标签在字典中的下标，所以可以直接进行编码。
```python
def __getitem__(self, idx):
    ......
    # 处理标签
    label = sample['label']
    encoded_label = self._encode_label(label)
    label_tensor = torch.LongTensor(encoded_label)
    ......
```
另外，由于输入数据的维度(channel)以及字符集的规模有所不同，在config中也要进行对应修改。需要修改encoder的input_channel（为3）以及counting_decoder的out_channel（为415）。另外需要在config中加上word_num为字符字典的大小(为415)。

可以直接复制三个修改后的文件，到官方CAN模型代码中进行覆盖。
