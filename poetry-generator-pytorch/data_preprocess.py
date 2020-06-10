#_*_coding:utf-8_*_
'''
@project: poetry-generator-pytorch
@author: exudingtao
@time: 2020/3/30 2:57 下午
'''

import re
from collections import Counter
import numpy as np

BATCH_SIZE = 25
# 数据路径
DATA_PATH = './data/train-poetry.txt'
# 单行诗最大长度
MAX_LEN = 64
# 禁用的字符，拥有以下符号的诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 一首诗（一行）对应一个列表的元素
poetry = []
pickle_path = 'tang.npz'  # 预处理好的二进制文件


def data_pre(DATA_PATH, poetry):
    '''
    :param path: 预处理文件位置
    :return: 处理后的train
    '''
    # 按行读取数据 poetry.txt
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 遍历处理每一条数据
    for line in lines:
        # 利用正则表达式拆分标题和内容
        fields = re.split(r"[:：]", line)
        # 跳过异常数据
        if len(fields) != 2:
            continue
        # 得到诗词内容（后面不需要标题）
        content = fields[1]
        # 跳过内容过长的诗词
        if len(content) > MAX_LEN - 2:
            continue
        # 跳过存在禁用符的诗词
        if any(word in content for word in DISALLOWED_WORDS):
            continue

        poetry.append(content.replace('\n', ''))  # 最后要记得删除换行符

    print('数据预处理后，训练集共{}首诗'.format(len(poetry)))
    # for i in range(0, 5):
    #     print(poetry[i])
    return poetry


class Tokenizer:
    """
    分词器
    """
    def __init__(self, tokens):
        # 词汇表大小
        self.dict_size = len(tokens)
        # 生成映射关系
        self.token_id = {} # 映射: 词 -> 编号
        self.id_token = {} # 映射: 编号 -> 词
        for idx, word in enumerate(tokens):
            self.token_id[word] = idx
            self.id_token[idx] = word

        # 各个特殊标记的编号id，方便其他地方使用
        self.start_id = self.token_id["[START]"]
        self.end_id = self.token_id["[END]"]
        self.none_id = self.token_id["[NONE]"]
        self.pad_id = self.token_id["[PAD]"]

    def id_to_token(self, token_id):
        """
        编号 -> 词
        """
        return self.id_token.get(token_id)

    def token_to_id(self, token):
        """
        词 -> 编号
        """
        return self.token_id.get(token, self.none_id)

    def encode(self, tokens):
        """
        词列表 -> [START]编号 + 编号列表 + [END]编号
        """
        token_ids = [self.start_id, ] # 起始标记
        # 遍历，词转编号
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        token_ids.append(self.end_id) # 结束标记
        return token_ids

    def decode(self, token_ids):
        """
        编号列表 -> 词列表(去掉起始、结束标记)
        """
        # 起始、结束标记
        flag_tokens = {"[START]", "[END]"}

        tokens = []
        for idx in token_ids:
            token = self.id_to_token(idx)
            # 跳过起始、结束标记
            if token not in flag_tokens:
                tokens.append(token)
        return tokens


def pad_line(tokenizer, line, length, padding=None):
    """
    对齐单行数据
    """
    if padding is None:
        padding = tokenizer.pad_id

    padding_length = length - len(line)
    if padding_length > 0:
        return line + [padding] * padding_length
    else:
        return line[:length]


def get_data(poetry):
    '''
    :param poetry: data
    :param tokenizer:
    :return:
    '''
    # 分词
    # 最小词频
    MIN_WORD_FREQUENCY = 8
    # 统计词频，利用Counter可以直接按单个字符进行统计词频
    counter = Counter()
    for line in poetry:
        counter.update(line)
    # 过滤掉低词频的词
    tokens = [token for token, count in counter.items() if count >= MIN_WORD_FREQUENCY]
    # 看一下统计的词频
    i = 0
    for token, count in counter.items():
        if i >= 5:
            break;
        print(token, "->", count)
        i += 1
    # 为了神经网络训练 补上特殊词标记：填充字符标记、未知词标记、开始标记、结束标记
    tokens = ["[PAD]", "[NONE]", "[START]", "[END]"] + tokens
    # 映射: 词 -> 编号
    word_idx = {}
    # 映射: 编号 -> 词
    idx_word = {}
    for idx, word in enumerate(tokens):
        word_idx[word] = idx
        idx_word[idx] = word

    tokenizer = Tokenizer(tokens)
    data = poetry
    max_length = max(map(len, data))
    new_data = []
    # 诗歌长度不够opt.maxlen的在前面补空格，超过的，删除末尾的
    for str_line in data:
        # 对每一行诗词进行编码、并补齐padding
        encode_line = tokenizer.encode(str_line)
        pad_encode_line = pad_line(tokenizer, encode_line, max_length + 2)  # 加2是因为tokenizer.encode会添加START和END
        new_data.append(pad_encode_line)

    # 保存成二进制文件
    np.savez_compressed(pickle_path,
                        data=new_data,
                        word2ix=word_idx,
                        ix2word=idx_word)

    return new_data, word_idx, idx_word





