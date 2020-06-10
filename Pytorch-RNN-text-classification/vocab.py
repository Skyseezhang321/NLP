# _*_coding:utf-8_*_
'''
@project: OutSourceProject
@author:
@time: 2020/4/4 1:51 下午
'''


from collections import Counter
import pandas as pd
# import torchwordemb
import torch
import torchtext

import util as ut


class VocabBuilder(object):
    '''
    Read file and create word_to_index dictionary.
    This can truncate low-frequency words with min_sample option.

    The vocab item is like "word,idx,tf*idf" respectively
    Return tf*idf values

    '''

    def __init__(self, path_file=None):
        # word count
        self.word_count = VocabBuilder.count_from_file(path_file)
        self.word_to_index = {}

    @staticmethod
    def extract_tfidf(self):
        with open(self.path_file, 'r') as f_in:
            raw_text = f_in.read()
            return self.document_frequencier(raw_text).to_array()

    @staticmethod
    def count_from_file(path_file, tokenizer=ut._tokenize):
        """
        count word frequencies in a file.
        Args:
            path_file:
        Returns:
            dict: {word_n :count_n, ...}
        """
        df = pd.read_csv(path_file)#todo
        # df = pd.read_csv(path_file, delimiter='\t')
        print(df.head(3))
        # tokenize
        df['body'] = df['body'].apply(tokenizer)
        # count
        word_count = Counter([tkn for sample in df['body'].values.tolist() for tkn in sample])
        print('Original Vocab size:{}'.format(len(word_count)))
        return word_count

    def get_word_index(self, min_sample=1, padding_marker='__PADDING__', unknown_marker='__UNK__',):
        """
        create word-to-index mapping. Padding and unknown are added to last 2 indices.
        Args:
            min_sample: for Truncation
            padding_marker: padding mark
            unknown_marker: unknown-word mark
        Returns:
            dict: {word_n: index_n, ... }
        """
        # truncate low fq word
        # 过滤掉低词频的词 prune by tf*idf values

        # TODO prune the vocab by using tf*idf values

        tokens = [token for token, count in self.word_count.items()
                  if count >= min_sample]
        # 为了神经网络训练 补上特殊词标记：填充字符标记、未知词标记、开始标记、结束标记
        tokens = [padding_marker, unknown_marker] + tokens
        # 映射: 词 -> 编号
        for idx, word in enumerate(tokens):
            self.word_to_index[word] = idx
        print('Turncated vocab size:{} (removed:{})'.format(len(self.word_to_index),
                                                            len(self.word_count) - len(self.word_to_index)))
        return self.word_to_index,None


class GloveVocabBuilder(object) :

    def __init__(self, path_glove):
        self.vec = None
        self.vocab = None
        self.path_glove = path_glove

    def get_word_index(self, padding_marker='__PADDING__', unknown_marker='__UNK__',):
        # _vocab, _vec = torchwordemb.load_glove_text(self.path_glove)
        # vocab = {padding_marker:0, unknown_marker:1}
        # for tkn, indx in _vocab.items():
        #     vocab[tkn] = indx + 2
        # vec_2 = torch.zeros((2, _vec.size(1)))
        # vec_2[1].normal_()
        # self.vec = torch.cat((vec_2, _vec))
        # self.vocab = vocab
        # return self.vocab, self.vec
        pass


if __name__ == "__main__":

    v_builder = VocabBuilder(path_file='data2/train.csv')
    d = v_builder.get_word_index(min_sample=10)
    print(d['__UNK__'])
    for k, v in sorted(d.items())[:100]:
        print(k, v)

    # v_builder = GloveVocabBuilder()
    # d, vec = v_builder.get_word_index()
    # print (d['__UNK__'])
    # for k, v in sorted(d.items())[:100]:
    #     print (k,v)
    #     print(v)
