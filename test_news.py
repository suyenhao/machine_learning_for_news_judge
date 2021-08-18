# -*- coding: utf-8 -*-
"""
@author: su
"""
from sklearn.feature_extraction.text import CountVectorizer
import joblib

count_vectorizer = joblib.load('count_vectorizer.pkl')
classifier = joblib.load('classifier.pkl')


def classify(document):
    label = {0: '真新聞', 1: '假新聞'}
    document_text = count_vectorizer.transform([document])
    y = classifier.predict(document_text)[0]
    return label[y]


document = input("請輸入新聞描述:")
print("這則新聞為 " + classify(document) + ".")

