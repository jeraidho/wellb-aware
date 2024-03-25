import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import pandas as pd
from pandarallel import pandarallel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import joblib

morph = MorphAnalyzer()
nltk.download('punkt')
sw = stopwords.words('russian')
sw.remove('не')


def preprocess(doc: str, normal_form=True) -> str:
    """
    Function to preprocess text in doc-variable with normal form.
    It tokenises doc, parse each token and gets the normal form of each token.
    :param doc: Text that is going to be preprocessed
    :param normal_form: True if normal_form transformation needed
    :return: Preprocessed doc
    """
    global morph, sw
    if not doc:
        raise ValueError('Document does not contain any characters')
    words = [morph.parse(word.lower())[0].normal_form if normal_form else word.lower()
             for word in word_tokenize(doc) if word.isalpha()]  # tokenization with normal form
    filtered = [word for word in words if word not in sw]  # stopwords filter
    return ' '.join(filtered)


def vectorise(data):
    """
    TF-IDF vectorisation by pre-saved vectoriser
    :param data: Doc that is to be vectorised by TF-IDF
    :return:
    """
    vectoriser = joblib.load('./data/vectoriser.pkl')  # load tf-idf vectoriser
    return vectoriser.transform(data)


def csv_gen():
    """
    Function to create .csv file needed for model
    :return: .csv file
    """
    if 'RU_dataset.json' not in os.listdir('./data'):
        raise ValueError('Dataset file does not exist')
    df = pd.read_json('./data/RU_dataset.json')
    df = df.rename(columns={0: 'Texts', 1: 'Annotation'})
    df['Target'] = df['Annotation'].apply(lambda x: 1 if x == 'suicide' else 0)
    df = df.drop_duplicates(subset=['Texts'])
    # function to parallise pandas
    pandarallel.initialize(progress_bar=True)
    # create tokenised and normalised docs
    df['Tokenised'] = df['Texts'].parallel_apply(preprocess, normal_form=False)
    df['Normalised'] = df['Texts'].parallel_apply(preprocess, normal_form=True)
    df = df.dropna()
    df.to_csv('./data/preprocessed.csv')


def evaluation(y_train, X_train, y_test, X_test, model):
    """
    Prints string with metrics for models evaluating
    :return: None
    """
    print(f"""
    Train:
    Accuracy: {round(accuracy_score(y_train, model.predict(X_train)), 3)}
    Precision: {round(precision_score(y_train, model.predict(X_train)), 3)}
    Recall: {round(recall_score(y_train, model.predict(X_train)), 3)}
    F1 score: {round(f1_score(y_train, model.predict(X_train)), 3)}

    Test:
    Accuracy: {round(accuracy_score(y_test, model.predict(X_test)), 3)}
    Precision: {round(precision_score(y_test, model.predict(X_test)), 3)}
    Recall: {round(recall_score(y_test, model.predict(X_test)), 3)}
    F1 score: {round(f1_score(y_test, model.predict(X_test)), 3)}
    """)
