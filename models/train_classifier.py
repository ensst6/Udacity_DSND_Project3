# import libraries
from sqlalchemy import create_engine
import sys
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
# these are for SVD/LSA
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def load_data(database_filepath):
    '''
    Open an SQLite database containing the messages and their categories.
    The message text goes into X, and each category (one-zero encoded;
    each message may have multiple categories) into a column of y

    Args:
        database_filepath (str): path to the SQLite database

    Returns:
        X (np array): message text
        y (np array): message categories (# messages x 35 columns)
        labels (list): labels for category columns
    '''
# connect to the SQLite database
    engine = create_engine('sqlite://'+database_filepath)
# extract the messages into X and the categories (multilabel output) into y
    df =pd.read_sql_table('messages', engine)
    X = df['message'].values
    Ydf = df.iloc[:, 4:]
    labels = Ydf.columns.to_list()
    y = Ydf.values

    return X, y, labels


def tokenize(text):
    '''
    Normalizes, tokenizes, and lemmatizes a string of text.
    Removes stopwords using nltk's stopwords dictionary.

    Args:
        text (str): a text string

    Returns:
        clean_tokens (list): a list of normalized, lemmatized text tokens
                             derived from the original text string
    '''
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize & remove stopwords
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not\
                    in stopwords.words('english')]

    return clean_tokens


def build_pipeline(clf, svd=False):
    if svd:
    # add on the steps to do LSA
        pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('svd', TruncatedSVD(100)),
                    ('nml', Normalizer(copy=False)),
                    ('multi_clf', MultiOutputClassifier(clf))
        ], verbose=True)
    else:
        pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('multi_clf', MultiOutputClassifier(clf))
        ], verbose=True)

    return pipeline

# took out print stmts here; either add here or as separate fxn code to make
# this into a df and print means, etc.
def get_results(model, y_test, y_pred, labels, cl_name, all_results):
    for i, label in enumerate(labels):
        result = classification_report(y_test[:,i], y_pred[:,i], output_dict=True)
        all_results.append([cl_name, label, result['0']['precision'], result['0']['recall'], \
                            result['0']['f1-score'], result['0']['support'], result['1']['precision'], \
                            result['1']['recall'], result['1']['f1-score'], result['1']['support'],\
                            result['accuracy'], result['macro avg']['precision'],\
                            result['macro avg']['recall'],result['macro avg']['f1-score']])
    return

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pickle.dump(model, model_filepath)

    return


#def main():
if __name__ == '__main__':
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, labels = load_data(database_filepath)


# split into train and validation sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        print('Building model...')
        parameters = [
                {
                'vect__ngram_range': [(1,1),(1,2)],
                'tfidf__use_idf': [True, False],
                'multi_clf__estimator__n_estimators':  [100, 200],
                'multi_clf__estimator__max_features': [0.5, "sqrt"]
        }]


        # create grid search object
        # when I run this on my machine, putting n_jobs = -1 here seems to work
        clf = RandomForestClassifier(random_state=42, n_jobs=-1)
        pipeline = build_pipeline(clf)
        # the multithreading option doesn't seem to work in iPython according to what I can find (failed for me)
        # on my machine n_jobs > 1 here fails with some picking error I can't figure out. but it runs on the Udacity VM
        cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

        print('Training model...')
        cv.fit(X_train, Y_train)
        Y_pred = cv.predict(X_test)

        print('Evaluating model...')
#        evaluate_model(model, X_test, Y_test, category_names)
        cv_results = []
        cl_name = str(type(clf)).split(".")[-1][:-2]   # thanks stack overflow
        get_results(cv, Y_test, Y_pred, labels, cl_name, cv_results)
        print("\nBest Parameters:", cv.best_params_)
        cv_df = pd.DataFrame(cv_results, columns=['classifier', 'category', 'precis_0', 'rcl_0', 'f1_0', 'support_0',\
                                                 'precis_1', 'rcl_1', 'f1_1', 'support_1','accuracy','ma_precision',\
                                                 'ma_recall', 'ma_f1'])
        summary = cv_df.groupby(['classifier'])[['accuracy', 'ma_precision', 'ma_recall', 'ma_f1']].mean()
        print('Results: \n', summary)
        print('Saving model...\n    MODEL: {}'.format(model_filepath))

        with open ('dist_test.pkl', 'wb') as outfile:
            pickle.dump(cv, outfile)
#        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


#if __name__ == '__main__':
#    main()
