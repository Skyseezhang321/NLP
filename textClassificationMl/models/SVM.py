#_*_coding:utf-8_*_
'''
@project: Exuding-NLP
@author: exudingtao
@time: 2020/3/1 8:49 下午
'''


from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

class TextClassifierSVM():

    def __init__(self, classifier = SVC(kernel='linear')):#可以换内核
        self.classifier = classifier
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 4), max_features=20000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, X):
        return self.classifier.predict(self.features([X]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)