#_*_coding:utf-8_*_
'''
@project: Exuding-NLP
@author: exudingtao
@time: 2020/3/1 8:54 下午
'''

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class TextClassifierKNN():

    def __init__(self, classifier=KNeighborsClassifier(n_neighbors=3)):
        self.classifier = classifier
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=12000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)