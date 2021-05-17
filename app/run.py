import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar
import joblib
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
engine = create_engine('sqlite:///../data/DisResp.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # keep the default genre example
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

# get the top five and bottom 5 categories (excluding "Related") by percent of messages
    totals = df.iloc[:, 5:].sum(axis = 0).sort_values()/(0.01*df.shape[0])

    # create visuals
    graph_one = []
    graph_one.append(
     Bar(
         x = genre_names,
         y = genre_counts
         )
     )

    layout_one = dict(title = 'Distribution of Message Genres',
               xaxis = dict(title = 'Genre'),
               yaxis = dict(title = 'Message Count'))


    # top five categories by %
    graph_two = []
    graph_two.append(
     Bar(
         x = list(totals[-5:].index),
         y = list(totals[-5:].values),
         marker_color = 'green'
         )
     )

    layout_two = dict(title = 'Five Most-Identified Categories (excludes Related)',
               xaxis = dict(title = 'Category'),
               yaxis = dict(title = 'Percent of Messages'))

    # bottom five categories by %
    graph_three = []
    graph_three.append(
     Bar(
         x = list(totals[0:5].index),
         y = list(totals[0:5].values),
         marker_color = 'red'
         )
     )

    layout_three = dict(title = 'Five Least-Identified Categories',
               xaxis = dict(title = 'Category'),
               yaxis = dict(title = 'Percent of Messages'))


    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))

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
