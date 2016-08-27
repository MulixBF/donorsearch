from flask import Flask, request, jsonify
from dill import dill
import re
from html2text import html2text

app = Flask(__name__)


def preprocess(text):
    text = html2text(text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('[A-Za-z0-9]', '', text)
    text = text.lower()
    return text

vectorizer = None
with open('../model/vectorizer.pkl', 'rb') as f:
    vectorizer = dill.load(f)

model = None
with open('../model/model.pkl', 'rb') as f:
    model = dill.load(f)


@app.route('/predict', methods=['GET'])
def predict():
    string = request.args.get('s')
    string = preprocess(string)
    vect = vectorizer.transform([string])
    prior = model.class_count_[1] / sum(model.class_count_)
    proba = model.predict_proba(vect)[0][1]

    return jsonify({
        'prior': prior,
        'posterior': proba,
        'result': 1 if proba > prior else 0
    })

if __name__ == '__main__':
    app.run()
