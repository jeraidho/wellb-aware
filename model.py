import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tools import evaluation, csv_gen, preprocess, vectorise
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import joblib

# read preprocessed .csv file or creates it
if 'preprocessed.csv' not in os.listdir('./data'):
    print('Necessary .csv file was not found')
    print('Creating file:')
    csv_gen()
df = pd.read_csv('./data/preprocessed.csv')
df = df.dropna()

# data splitting
seed = np.random.seed()
X, y = df['Normalised'], df['Target']
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)

# vectorisation training
tfidf = TfidfVectorizer()
X_train_transformed = tfidf.fit_transform(X_train)
X_test_transformed = tfidf.transform(X_test)

joblib.dump(tfidf, './data/vectoriser.pkl')

# train model
clf = MultinomialNB()
clf.fit(X_train_transformed, y_train)
evaluation(y_train, X_train_transformed, y_test, X_test_transformed, clf)

joblib.dump(clf, './data/classifier.pkl')

# print(clf.predict(vectorise([preprocess('я хочу убить себя')])))