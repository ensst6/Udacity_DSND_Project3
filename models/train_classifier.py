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
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
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
    '''
    Creates a scikit-learn pipeline to perform the machine learning analysis.
    First applies the text normalization steps, then runs the train_classifier
    in multi-output form to model the 35 message categories separately.

    Args:
        clf (sklearn object): the machine-learning train_classifier
        svd (boolean): If True, use singular-value decomposition on the text for
                       latent semantic analysis (LSA) dimensionality reduction.

    Returns:
        pipeline: sklearn Pipeline object
    '''
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

def get_results(model, y_test, y_pred, labels, cl_name):
    '''
    Prints the best parameters from GridSearchCV, along with name of classifier.
    Evaluates the results of the trained model using sklearn's classification_report
    (accuracy, precision, recall, and f1 score).
    Prints the macro-average values for each of the 35 category columns.
    Also calculates and prints the overall average for all categories.
    Stores all of the classification_report results for later manipulation.


    Args:
        model (sklearn object): the trained classifier
        y_test (numpy array): actual classification for each category
        y_pred (numpy array): predicted classification for each category
        labels (list): category names
        cl_name (string): classifier name

    Returns:
        gscv_df (pandas dataframe): results from classification_report.
                                    column names are result descriptions
                                    rows are results for each category
    '''

    print('\nClassifier:', cl_name)
    print('\nBest GridSearch Parameters:', model.best_params_)

    # get the classification results for each category.
    # saving to convert to dataframe for possible later use
    all_results = []
    for i, label in enumerate(labels):
        result = classification_report(y_test[:,i], y_pred[:,i], output_dict=True)
        all_results.append([cl_name, label, result['0']['precision'], result['0']['recall'], \
                            result['0']['f1-score'], result['0']['support'], result['1']['precision'], \
                            result['1']['recall'], result['1']['f1-score'], result['1']['support'],\
                            result['accuracy'], result['macro avg']['precision'],\
                            result['macro avg']['recall'],result['macro avg']['f1-score']])
        print('\nCategory: ',label)
        print('\nAccuracy: {:.4f}, Macro Avgs: Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}'\
              .format(result['accuracy'], result['macro avg']['precision'],\
              result['macro avg']['recall'],result['macro avg']['f1-score']))

    # saving all of these columns in case when want to look in more detail later
    gscv_df = pd.DataFrame(cv_results, columns=['classifier', 'category', 'precis_0', 'rcl_0', 'f1_0', 'support_0',\
                                                'precis_1', 'rcl_1', 'f1_1', 'support_1','accuracy','ma_precision',\
                                                'ma_recall', 'ma_f1'])
    # get the overall average results
    summary = gscv_df.groupby(['classifier'])[['accuracy', 'ma_precision', 'ma_recall', 'ma_f1']].mean()
    print('Averaged Results: \n', summary)

    return gscv_df

def save_model(model, model_filepath):
    '''
    Saves the final model as a Python pkl file.

    Args:
        model (sklearn object): the trained classifier
        model_filepath (string): path & filename for output file.
                                 should end in '.pkl'

    Returns:
        none
    '''
    with open (model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)

    return



if __name__ == '__main__':
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, labels = load_data(database_filepath)

# split into train and validation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        print('Building model...')

        # create classifer object and pipeline
        clf = RandomForestClassifier(random_state=42, n_jobs=-1) # mutlithreading works here
        pipeline = build_pipeline(clf)

        # create a gridsearch object to optimize the model
        parameters = [{'vect__ngram_range': [(1,1),(1,2)],
                       'tfidf__use_idf': [True, False],
                       'multi_clf__estimator__n_estimators':  [100, 200],
                       'multi_clf__estimator__max_features': [0.5, "sqrt"]}]
        # on my machine n_jobs > 1 here fails with some picking error I can't figure out. but it runs on the Udacity VM
        # I used cv=2 (instead of default 5) to get a somewhat reasonable runtime
        gscv = GridSearchCV(pipeline, param_grid=parameters, cv=2, scoring='f1_macro')

        print('Training model...')
        gscv.fit(X_train, y_train)
        y_pred = gscv.predict(X_test)

        print('Evaluating model...')
        # extract classifer name
        cl_name = str(type(clf)).split(".")[-1][:-2]   # thanks stack overflow

        # this returns a dataframe in case we want to do more detailed analysis
        # for now, it's not being used for anything further
        gscv_df = get_results(gscv, y_test, y_pred, labels, cl_name)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(gscv, model_filepath)
        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')
