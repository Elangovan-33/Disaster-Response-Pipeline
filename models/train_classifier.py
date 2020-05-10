# import libraries
import sys

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.externals import joblib


def load_data(database_filepath):
    """
    Load the database into a Pandas dataframe and return X, Y
    --
    Inputs:
        database_filepath: a path to the database
    Outputs:
        X: Features to be fed into a ML model
        Y: Target values to be fed into a ML model
        category_names: Name of categories of the Y
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.columns[4:]
    
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the input text and return a clean tokens after normalization, lemmatization, and removing stopwords
    --
    Inputs:
        text: a piece of text
    Outputs:
        clean_tokens: a clean list of tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens


def build_model():
    """
    Build a full pipeline including CountVectorizer, tf-idf, MultiOutput RandomForestClassifier Classifier, and GridSearchCV  
    --
    Outputs:
        cv: a full Machine Learning pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    } 

    cv = GridSearchCV(pipeline, parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print precision, recall, fscore for all the categories for test set
    --
    Inputs:
        model: a trained model
        X_test: features of test set
        Y_test: target values of test set
        category_names: names of different target value categories
    """
    y_pred = model.predict(X_test)
    for i, category in enumerate(category_names): 
        precision, recall, fscore, _ = score(Y_test[:,i], y_pred[:,i], average='micro')
        print('{:25}  precision: {:.2%}  recall: {:.2%}  fscore: {:.2%}'.format(category, precision, recall, fscore))


def save_model(model, model_filepath):
    """
    Save the model 
    --
    Inputs:
        model: a trained model
        model_filepath: model file path
    """
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
