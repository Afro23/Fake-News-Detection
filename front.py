from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

app = Flask(__name__,template_folder='./templates',static_folder='./static')

loaded_model = pickle.load(open('assets/models/fmodel.py', 'rb'))
lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))
tfidf_v = pickle.load(open('assets/models/lr_tfidf.sav', 'rb'))
corpus = []

def fake_news_det(news):
    review = news
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    review = review.lower()
    review = nltk.word_tokenize(review)
    for y in review :
        if y not in stpwrds :
            corpus.append(lemmatizer.lemmatize(y))
    input_data = [' '.join(corpus)]
    #vectorized_input_data = tfidf_v.fit_transform(input_data)
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 0:
        return(print("Prediction of the News :  Looking Fake⚠ News📰 "))
    else:
        return(print("Prediction of the News : Looking Real News📰 "))

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")



if __name__ == '__main__':
    app.run(debug=True)