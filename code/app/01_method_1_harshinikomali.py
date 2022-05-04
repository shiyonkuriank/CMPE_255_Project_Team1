#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy import spatial
get_ipython().run_line_magic('run', 'movies_preprocessing.ipynb import Add')


# In[4]:


def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['cast_bin']
    scoreB = b['cast_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    return genreDistance + directDistance + scoreDistance


# In[5]:


Similarity(3,160)


# In[6]:


print(movies.iloc[3])
print(movies.iloc[160])


# In[7]:


new_id = list(range(0,movies.shape[0]))
movies['new_id']=new_id
movies=movies[['title','genres','vote_average','genres_bin','cast_bin','new_id','director','director_bin']]
movies.head()


# In[8]:


import operator

def predict_score(name):
    #name = input('Enter a movie title: ')
    new_movie = movies[movies['title'].str.contains(name)].iloc[0].to_frame().T
    print('Selected Movie: ',new_movie.title.values[0])
    def getNeighbors(baseMovie, K):
        distances = []
    
        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    avgRating = 0
    neighbors = getNeighbors(new_movie, K)
    
    print('\nRecommended Movies: \n')
    for neighbor in neighbors:
        avgRating = avgRating+movies.iloc[neighbor[0]][2]  
        print( movies.iloc[neighbor[0]][0]+" | Genres: "+str(movies.iloc[neighbor[0]][1]).strip('[]').replace(' ','')+" | Rating: "+str(movies.iloc[neighbor[0]][2]))
    
    print('\n')
    avgRating = avgRating/K
    print('The predicted rating for %s is: %f' %(new_movie['title'].values[0],avgRating))
    print('The actual rating for %s is %f' %(new_movie['title'].values[0],new_movie['vote_average']))


# In[9]:


predict_score('Godfather')


# In[10]:


predict_score('Finding Nemo')


# In[11]:


predict_score('The Shawshank Redemption')


# In[ ]:




