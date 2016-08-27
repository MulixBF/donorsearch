from flask import Flask, request, jsonify
from dill import dill
import re
from html2text import html2text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import RussianStemmer

app = Flask(__name__)


def preprocess(text):
    text = html2text(text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('[A-Za-z0-9]', '', text)
    text = text.lower()
    return text

vectorizer = None
with open('/media/data/donorsearch/model/randomforest/vectorizer.pkl', 'rb') as f:
    vectorizer = dill.load(f)

model = None
with open('/media/data/donorsearch/model/randomforest/model.pkl', 'rb') as f:
    model = dill.load(f)


@app.route('/predict', methods=['GET'])
def predict():
    string = request.args.get('s')
    string = preprocess(string)
    vect = vectorizer.transform([string])
    proba = model.predict_proba(vect)[0][1]

    return jsonify({
        'proba': proba,
        'result': 1 if proba > 0.5 else 0
    })

if __name__ == '__main__':
    app.run()
