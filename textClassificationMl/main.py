#_*_coding:utf-8_*_
'''
@project: Exuding-NLP
@author: exudingtao
@time: 2020/3/1 8:55 下午
'''


import pandas as pd
import random
from sklearn.utils import shuffle
import jieba
from textClassificationMl.models.KNN import TextClassifierKNN
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    data = pd.read_csv('../data/weibo-2014-tag.csv', encoding="utf8", error_bad_lines=False, sep=None)
    data = data[['微博内容', '是否是学习者']]
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    data['是否是学习者'] = data['是否是学习者'].astype("int")
    data = shuffle(data)  # 乱序
    # 平衡样本
    isStudyData = data[data['是否是学习者'] == 0][:12000]
    notStudyData = data[data['是否是学习者'] == 1]
    data = isStudyData.append(notStudyData)
    data = shuffle(data)  # 乱序


    def split(all_list, shu=False, ratio=0.7):
        num = len(all_list)
        offset = int(num * ratio)
        if num == 0 or offset < 1:
            return [], all_list
        if shuffle:
            all_list = shuffle(all_list)  # 列表随机排序
        train = data[:offset]
        test = data[offset:-int((num - offset) * 0.5)]
        val = data[-int((num - offset) * 0.5):]
        return train, test, val


    train, test, val = split(data, shu=False, ratio=0.7)

    leaner = train[train['是否是学习者'] == 1]
    leaner = leaner['微博内容'].values.tolist()

    noleaner = train[train['是否是学习者'] == 0]
    noleaner = noleaner['微博内容'].values.tolist()

    # 去停用词
    stop_word_path = '../data/stopword.txt'
    stop_word_list = [sw.replace('\n', '') for sw in open(stop_word_path).readlines()]


    def preprocess_text(content_lines, sentences, category):
        for line in content_lines:
            try:
                segs = jieba.lcut(line)
                segs = list(filter(lambda x: len(x) > 1, segs))
                segs = list(filter(lambda x: x not in stop_word_list, segs))
                sentences.append((' '.join(segs), category))  # 分完词后 用空格串起来 且和label合并为一个元组
            except Exception as e:
                print(line)
                continue


    # 生成训练数据
    sentences = []
    preprocess_text(leaner, sentences, 'leaner')
    preprocess_text(noleaner, sentences, 'noleaner')

    random.shuffle(sentences)


    x, y = zip(*sentences)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)


    text_classifier = TextClassifierKNN()
    text_classifier.fit(x_train, y_train)
    print(text_classifier.predict('我是一个mooc的学习者'))
    print(text_classifier.score(x_test, y_test))