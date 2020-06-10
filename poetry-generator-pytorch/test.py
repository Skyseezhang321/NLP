#_*_coding:utf-8_*_
'''
@project: poetry-generator-pytorch
@author: exudingtao
@time: 2020/3/30 3:34 下午
'''


from data_preprocess import data_pre ,get_data
import torch
from models import PoetryModel
import warnings


# 数据路径
DATA_PATH = './data/train-poetry.txt'
# 单行诗最大长度
MAX_LEN = 64
# 禁用的字符，拥有以下符号的诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 一首诗（一行）对应一个列表的元素
poetry = []
BATCH_SIZE = 64
lr = 1e-3
model_path = 'checkpoints/tang_19.pth' # 预训练模型路径
epoch = 20
plot_every = 20  # 每20个batch 可视化一次
debug_file = '/tmp/debugp'
model_prefix = 'checkpoints/tang'  # 模型保存路径
env = 'poetry'  # visdom env
max_gen_len = 200  # 生成诗歌最长长度
prefix_words = '福如东海至，寿比南山松。'  # 不是诗歌的组成部分，用来控制生成诗歌的意境
start_words = '生日快乐'  # 诗歌开始
acrostic = True  # 是否是藏头诗
pickle_path = 'tang.npz'  # 预处理好的二进制文件


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一首完整的诗歌
    start_words：u'春江潮水连海平'
    比如start_words 为 春江潮水连海平，可以生成：
    """

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = torch.Tensor([word2ix['[START]']]).view(1, 1).long()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '[END]':
            del results[-1]
            break
    return results

def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    生成藏头诗
    start_words : u'深度学习'
    生成：
    深木通中岳，青苔半日脂。
    度山分地险，逆浪到南巴。
    学道兵犹毒，当时燕不移。
    习根通古岸，开镜出清羸。
    """
    results = []
    start_word_len = len(start_words)
    input = (torch.Tensor([word2ix['[START]']]).view(1, 1).long())
    hidden = None

    index = 0  # 用来指示已经生成了多少句藏头诗
    # 上一个词
    pre_word = '[START]'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1, 1)

    for i in range(max_gen_len):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！', '[START]'}):
            # 如果遇到句号，藏头的词送进去生成

            if index == start_word_len:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                break
            else:
                # 把藏头的词作为输入送入模型
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1, 1)
        else:
            # 否则的话，把上一次预测是词作为下一个词输入
            input = (input.data.new([word2ix[w]])).view(1, 1)
        results.append(w)
        pre_word = w
    return results


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # 数据预处理
    poetry = data_pre(DATA_PATH, poetry)
    data, word2ix, ix2word = get_data(poetry)
    model = PoetryModel(len(word2ix), 128, 256);
    map_location = lambda s, l: s
    state_dict = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state_dict)


    # python2和python3 字符串兼容

    start_words = start_words
    prefix_words =prefix_words if prefix_words else None
    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    #调用哪个函数
    gen_poetry = gen_acrostic if acrostic else generate

    result = gen_poetry(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result).replace('。', u'\n'))
