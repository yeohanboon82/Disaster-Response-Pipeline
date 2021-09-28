import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterMessage', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_categories = df.drop(['id','message','original','genre'], axis=1)
    category_counts=df_categories.sum(axis=0)
    category_names = df_categories.columns
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker_color=['gold','lime','red']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'titlefont':{'size':25},
                'yaxis': {
                    'title': "Count",
                    'titlefont':{'size':20},
                    'tickfont':{'size':20}
                },
                'xaxis': {
                    'title': "Genre",
                    'titlefont':{'size':20},
                    'tickfont':{'size':20}
                }
            }
        },
        {
            'data': [
                Bar(
                    y=category_names,
                    x=category_counts,
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Categories Counts',
                'titlefont':{'size':25},
                'height':1200,
                'yaxis': {
                    'title': "Count",
                    'titlefont':{'size':20},
                    'tickfont':{'size':20},
                    'categoryorder':'total ascending'
                },
                'xaxis': {
                    'title': "Category",
                    'titlefont':{'size':20},
                    'tickfont':{'size':20}   ,
                    'domain':[0.2, 1]
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
