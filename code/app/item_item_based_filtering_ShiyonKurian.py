# -*- coding: utf-8 -*-
"""Item_Item_based_Filtering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yopNx_XR9v5l4NTRzR87Ui9XbCWUVAJt

**Item Based Collaborative Filtering**
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns

"""Here we use MovieLens Dataset for predicting the top 10 similar movies."""

from urllib.request import urlretrieve
import zipfile
import os
if not os.path.isfile('movielens.zip'):
    print("downloading dataset.....")
    urlretrieve("https://files.grouplens.org/datasets/movielens/ml-latest-small.zip", "movielens.zip")
    zip_ref = zipfile.ZipFile('movielens.zip', "r")
    zip_ref.extractall()
    print("dataset download completed .")
else:
    print("Required files are already present.")

from google.colab import files
uploaded = files.upload()

users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv(
    'u.user', sep='|', names=users_cols, encoding='ISO-8859-1')
users.head(10)

ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv(
    'u.data', sep='\t', names=ratings_cols, encoding='latin-1')
ratings.head()

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movies_cols = [
    'movie_id', 'title', 'release_date', "video_release_date", "imdb_url"
] + genre_cols
movies = pd.read_csv(
    'u.item', sep='|', names=movies_cols, encoding='latin-1')
movies = movies.drop(columns = 'imdb_url')
movies.head()

"""**Visulaization**

In this part we try to vizualize the total number of ratings for each unique rating.
"""

df = ratings.filter(['rating','movie_id'], axis = 1)

df['number_of_ratings'] = df.groupby('movie_id')['rating'].count()

import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=df, height = 10)

movielens = ratings.merge(movies, on='movie_id').merge(users, on='user_id')
movielens.head()

occupation_filter = alt.selection_multi(fields=["occupation"])
occupation_chart = alt.Chart().mark_bar().encode(
    x="count()",
    y=alt.Y("occupation:N"),
    color=alt.condition(
        occupation_filter,
        alt.Color("occupation:N", scale=alt.Scale(scheme='category20')),
        alt.value("lightgray")),
).properties(width=300, height=300, selection=occupation_filter)


def filtered_hist(field, label, filter):
    """Creates a layered chart of histograms.
    The first layer (light gray) contains the histogram of the full data, and the
    second contains the histogram of the filtered data.
    """
    base = alt.Chart().mark_bar().encode(
        x=alt.X(field, bin=alt.Bin(maxbins=10), title=label),
        y="count()",
    ).properties(
        width=300,
    )
    return alt.layer(
        base.transform_filter(filter),
        base.encode(color=alt.value('lightgreen'), opacity=alt.value(.7)),
    ).resolve_scale(y='independent')


def flatten_cols(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df
pd.DataFrame.flatten_cols = flatten_cols
#function for multipurpose use
def mask(df, key, function):
    return df[function(df[key])]
pd.DataFrame.mask = mask

users_ratings = (
    ratings
    .groupby('user_id', as_index=False)
    .agg({'rating': ['count', 'mean']})
    .flatten_cols()
    .merge(users, on='user_id')
)

alt.hconcat(
    filtered_hist('rating count', '# ratings / user', occupation_filter),
    filtered_hist('rating mean', 'mean user rating', occupation_filter),
    occupation_chart,
    data=users_ratings)

sns.set(rc={'figure.figsize':(20,10)})
genre_occurences = movies[genre_cols].sum().to_dict()


df2 = movies.melt(value_vars=genre_cols)
df2 = df2[df2["value"] != 0]
sns.countplot(data=df2, x="variable")

"""Item Based Filtering"""

cols = list(ratings.columns)
cols.remove('unix_timestamp')


rating = pd.merge(ratings[cols], movies[['title','movie_id']])
rating.head()

userRatings = rating.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

corrMatrix = userRatings.corr()
corrMatrix.head()

myRatings = userRatings.loc[20].dropna()
myRatings

"""We  will try to find the similarity for user 20"""

simCandidates = pd.Series(dtype = 'float64')

for i in range(0, len(myRatings.index)):

    sims = corrMatrix[myRatings.index[i]].dropna()
    sims = sims.map(lambda x: x * myRatings[i])
    simCandidates = simCandidates.append(sims)
     
simCandidates.sort_values(inplace = True, ascending = False)
print (simCandidates.head(10))

simCandidates.sort_values(inplace = True, ascending = False)
simCandidates.head(10)

filteredSims = simCandidates.drop(myRatings.index)
filteredSims.head(10)

"""Finding the RMSE score for Item based Filtering"""

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()

data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

"""Splitting the data to train and test"""

from surprise.model_selection import train_test_split

trainset, testset = train_test_split(data, test_size=0.2)

trainsetfull = data.build_full_trainset()

print('Number of users: ', trainset.n_users, '\n')
print('Number of items: ', trainset.n_items, '\n')

trainset_iids = list(trainset.all_items())
iid_converter = lambda x: trainset.to_raw_iid(x)
trainset_raw_iids = list(map(iid_converter, trainset_iids))

"""Uses KNNWithMeans algorithm to find the RMSE Score"""

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

from surprise.model_selection import cross_validate
results = cross_validate(
    algo = algo, data = data, measures=['RMSE'], 
    cv=5, return_train_measures=True
    )

results['test_rmse'].mean()

algo.fit(trainsetfull)
