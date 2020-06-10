#_*_coding:utf-8_*_
'''
@project: poetry-generator-pytorch
@author: exudingtao
@time: 2020/3/30 3:25 下午
'''


from torch.utils.data import DataLoader
import math
import numpy as np


class PoetryDataSet:
    """
    古诗数据集生成器
    """
    def __init__(self, data, tokenizer, batch_size):
        # 数据集
        self.data = data
        self.total_size = len(self.data)
        # 分词器，用于词转编号
        self.tokenizer = tokenizer
        # 每批数据量
        self.batch_size = batch_size
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))

    def pad_line(self, line, length, padding=None):
        """
        对齐单行数据
        """
        if padding is None:
            padding = self.tokenizer.pad_id

        padding_length = length - len(line)
        if padding_length > 0:
            return line + [padding] * padding_length
        else:
            return line[:length]

    def __len__(self):
        return self.steps

    def __iter__(self):
        # 打乱数据
        np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for start in range(0, self.total_size, self.batch_size):
            end = min(start + self.batch_size, self.total_size)
            data = self.data[start:end]

            max_length = max(map(len, data))

            batch_data = []
            for str_line in data:
                # 对每一行诗词进行编码、并补齐padding
                encode_line = self.tokenizer.encode(str_line)
                pad_encode_line = self.pad_line(encode_line, max_length + 2) # 加2是因为tokenizer.encode会添加START和END
                batch_data.append(pad_encode_line)

            batch_data = np.array(batch_data)
            # yield 特征、标签
            yield batch_data[:, :-1], batch_data[:, 1:]

    def generator(self):
        while True:
            yield from self.__iter__()