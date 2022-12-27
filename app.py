from flask import Flask, render_template, request, jsonify
import nltk
import pickle
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)
ps = PorterStemmer()

model = pickle.load(open('runam_model.pkl', 'rb'))
tfidfvect = pickle.load(open('runam_tfidfvect.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index1():
    return render_template('index1.html')

def predict(text):
    headline = re.sub('[^a-zA-Z]', ' ', text)
    headline = headline.lower()
    headline = headline.split()
    headline = [ps.stem(word) for word in headline if not word in stopwords.words('english')]
    headline = ' '.join(headline)
    headline_vect = tfidfvect.transform([headline]).toarray()
    prediction = 'This news item is likely to be from an unverified news source, so be careful of what decision you make with it.' if model.predict(headline_vect) == 0 else 'This news item is likely to be from a verified news source.'
    return prediction

@app.route('/', methods=['POST'])
def webapp():
    text = request.form['text']
    prediction = predict(text)
    return render_template('index1.html', text=text, result=prediction)

@app.route('/about')
def about():
  return render_template('about.html')  



@app.route('/predict/', methods=['GET','POST'])
def api():
    text = request.args.get("text")
    prediction = predict(text)
    return jsonify(prediction=prediction)

ol like Postman or by making a request to the endpoint using Python's 'requests' library. For example:


import requests

url = 'http://localhost:5000/predict'
data = {'text': 'This is a news article about a political scandal.'}

response = requests.post(url, json=data)
prediction = response.json()['prediction']
print(prediction)  # Output: 'This news item is likely to be from an unverified news source, so be careful of what decision you make with it.'





if __name__ == "__main__":
    app.run()
