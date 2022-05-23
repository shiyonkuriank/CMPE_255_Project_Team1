# -*- coding: utf-8 -*-
"""Item_Based_Colaborative_Filtering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z54JXiVsI1zr_trGJ1l922ObqrJzUpfN
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files
uploaded = files.upload()

import csv
movies = pd.read_csv(
    'movies.csv', encoding='latin-1')

movies.head()

ratings = pd.read_csv(
    'ratings.csv',encoding='latin-1')
ratings.head()

df = ratings.filter(['rating','movieId'], axis = 1)

df['number_of_ratings'] = df.groupby('movieId')['rating'].count()

import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=df, height = 10)

data = ratings.merge(movies,on='movieId')
data.head()

df = (ratings.groupby(by=['movieId'])['rating'].count().reset_index().rename(columns={'rating':'totalRatingCount_movies'})
       [['movieId','totalRatingCount_movies']])
df.tail()

df['totalRatingCount_movies'].describe()

df1 = (ratings.groupby(by=['userId'])['rating'].count().reset_index().rename(columns={'rating':'totalRatingCount_users'})
       [['userId','totalRatingCount_users']])
df1.tail()

df1['totalRatingCount_users'].describe()

data1 = data.merge(df,on='movieId')
data2 = data1.merge(df1,on='userId')
data2.head()

popular_movies1 = data2.query('totalRatingCount_movies >= 205')
popular_movies = popular_movies1.query ('totalRatingCount_users >= 50')
popular_movies.head()

#pivot ratings into movie features
movie_user_rating_pivot = popular_movies.pivot(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)

# create mapper from movie title to index
movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(movies.set_index('movieId').loc[movie_user_rating_pivot.index].title))
}

movie_user_rating_pivot

from scipy.sparse import csr_matrix

# convert dataframe of movie features to scipy sparse matrix
movie_user_rating_matrix = csr_matrix(movie_user_rating_pivot.values)

# import libraries
from sklearn.neighbors import NearestNeighbors
# define model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# fit
model_knn.fit(movie_user_rating_matrix)

pip install fuzzywuzzy

from fuzzywuzzy import fuzz

def fuzzy_matching(mapper, fav_movie, verbose=True):
   
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]

def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    # inference
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    # get list of raw idx of recommendations
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))

my_favorite = 'Toy story'

make_recommendation(
    model_knn=model_knn,
    data=movie_user_rating_matrix,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)

pip install surprise

from surprise import Reader, Dataset
from surprise.model_selection import cross_validate

reader = Reader()

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.2)
trainsetfull = data.build_full_trainset()
print('Number of users: ', trainset.n_users, '\n')
print('Number of items: ', trainset.n_items, '\n')

trainset_iids = list(trainset.all_items())
iid_converter = lambda x: trainset.to_raw_iid(x)
trainset_raw_iids = list(map(iid_converter, trainset_iids))

from surprise import KNNBasic
my_k = 15
my_min_k = 5
my_sim_option = {
    'name':'pearson', 'user_based':False, 
    }
algo = KNNBasic(
    k = my_k, min_k = my_min_k, sim_option = my_sim_option
    )
algo.fit(trainset)

from surprise import accuracy
predictions = algo.test(testset)
accuracy.rmse(predictions)

from surprise.model_selection import cross_validate
results = cross_validate(
    algo = algo, data = data, measures=['RMSE'], 
    cv=5, return_train_measures=True
    )

results['test_rmse'].mean()

algo.fit(trainsetfull)

from surprise import KNNWithMeans
my_k = 15
my_min_k = 5
my_sim_option = {
    'name':'pearson', 'user_based':False, 
    }
algo = KNNWithMeans(
    k = my_k, min_k = my_min_k, sim_option = my_sim_option
    )
algo.fit(trainset)

from surprise import accuracy
predictions = algo.test(testset)
accuracy.rmse(predictions)