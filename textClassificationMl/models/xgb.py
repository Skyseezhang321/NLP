#_*_coding:utf-8_*_
'''
@project: Exuding-NLP
@author: exudingtao
@time: 2020/3/1 8:52 下午
'''

from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class TextClassifierXGB():

    def __init__(self, classifier=XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=100,  # 弱分类树太少的话取不到更多的特征重要性
                                                silent=True, random_state=0, subsample=0.111,
                                                objective='binary:logistic')):
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