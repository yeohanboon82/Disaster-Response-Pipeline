import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath='../data/'):
    # load data from database
    print(database_filepath)
    engine = create_engine('sqlite:///'+database_filepath+'DisasterResponse.db')
    df = pd.read_sql_table('DisasterMessage', engine)  
    X = df['message']
    Y = df[df.columns[4::]]
    Y = np.array(Y)
    category_names = df.columns[4::]
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #List Hyperparameters that we want to tune. IDE can't support.
    '''
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'clf__estimator__bootstrap': [True, False],
        'clf__estimator__max_depth': [10, 30, 50, 70, 90, None],
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__n_estimators': [200, 600, 1000, 1400, 1800]
    } 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    '''
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    #Generate and print report
    for i,column in enumerate(category_names):
        y_test_column = Y_test[:,i]
        y_pred_column = Y_pred[:,i]
        num_classes = Y_test[:,i].max() + 1

        if num_classes == 3: 
            target_names = [column+'_0', column+'_1', column+'_2']      

        elif num_classes == 2: 
            target_names = [column+'_0', column+'_1']

        elif num_classes == 1: 
            target_names = [column+'_0']

        print(classification_report(y_test_column, y_pred_column, target_names=target_names))


def save_model(model, model_filepath=''):
    filename = model_filepath+'classifier.pkl'
    pickle.dump(model, open(filename, 'wb'))


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