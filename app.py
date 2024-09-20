from flask import Flask, render_template, request
import pandas as pd
import pickle
import dill
from src.components.recommender import ModelTrainer


application = Flask(__name__)
app = application

#Popularity based recommender
dic = pickle.load(open('pop.pkl', 'rb'))
popular_df = pd.DataFrame(dic)

book_title = list(popular_df['Book-Title'].values)
author = list(popular_df['Book-Author'].values)
image = list(popular_df['Image-URL-M'].values)
votes = list(popular_df['num_ratings'].values)
avg_rating = list(popular_df['avg_rating'].values)

#Collaborative based recommender
pt = dill.load(open('artifacts/pivot_table.pkl', 'rb'))
recommender = ModelTrainer()


# APP building
@app.route('/')
def index():
    return render_template("index.html", book_name = book_title, author_name = author, 
                           image_URL = image, voted = votes, ratings = avg_rating)

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods = ['POST'])
def recommend():
    user_input = request.form.get('user_input')
    books = recommender.initiate_recommendation(book_name = user_input, pivot=pt)
    return render_template('recommend.html', name = book_title, data = books )

if __name__ == '__main__':
    app.run(debug = True)