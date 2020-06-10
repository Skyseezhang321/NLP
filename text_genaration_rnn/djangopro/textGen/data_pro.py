#_*_coding:utf-8_*_
'''
@project: djangopro
@author: exudingtao
@time: 2020/4/8 7:15 下午
'''
#分词
import jieba
def preprocess_text(content_lines):
    try:
        segs = jieba.lcut(content_lines)
        #segs = list(filter(lambda x: len(x)>1,segs))
        segs = ' '.join(segs)#分完词后 用空格串起来 且和label合并为一个元组
        return segs
    except Exception as e:
        return ' '

if __name__ == '__main__':
    with open('./data/三重门.txt', 'r') as f:
        with open('./data/train.txt', 'w') as fw:
            for line in f.readlines():
                line = preprocess_text(line)
                fw.write(line)