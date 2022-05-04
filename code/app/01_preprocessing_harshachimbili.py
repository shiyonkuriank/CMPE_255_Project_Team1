#!/usr/bin/env python
# coding: utf-8

# In[622]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import plotly as py
from chart_studio import plotly
import plotly.express as px
import plotly.graph_objs as go
from wordcloud import WordCloud
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import math
import warnings
warnings.filterwarnings("ignore")


# In[571]:


filepath = "C:\\Users\\Checkout\\Documents\\SJSU Spring 2022\\CMPE 255\\Project\\team1-movie-recommendation-system\\code\\data\\Movies.csv"
movies = pd.read_csv(filepath,parse_dates=['release_date'])
movies.describe()


# In[572]:


display(movies.sample(10))


# In[573]:


movies.isnull().sum().sort_values(ascending=False)


# In[574]:


sns.distplot(movies['revenue'],color='g')
plt.title('Distribution of Revenue')
plt.xlabel('Revenue in USD')
plt.ylabel('Frequency in Log')
plt.show()


# In[575]:


movies['revenue'].replace(0,np.nan,inplace=True)
movies['revenue'].isnull().sum()


# In[576]:


movies.loc[movies.revenue.isnull(), 'revenue'] = movies.groupby(pd.Grouper(key='release_date', axis=0, freq='1Y'))['revenue'].transform('median')
movies['revenue'].isnull().sum()


# In[577]:


movies.drop(movies.index[movies.revenue.isna() == True],inplace=True)
display(movies.shape)
display(movies['revenue'].isnull().sum())


# In[578]:


movies.isnull().sum().sort_values(ascending=False)


# In[579]:


#drop movies with null values i genres
movies.dropna(subset=['genres'],inplace=True)
movies.isnull().sum().sort_values(ascending=False)


# In[580]:


#drop those rows with vote_count < 10 i.e. atleast 10 user should have voted for the movie
movies= movies[movies['vote_count'] > 10]
movies.isnull().sum().sort_values(ascending=False)
#replace null in cast,director,producer,production_companies,writercolumn with "Unknown"
movies['cast'].replace(np.nan,"Unknown",inplace=True)
movies['director'].replace(np.nan,"Unknown",inplace=True)
movies['producer'].replace(np.nan,"Unknown",inplace=True)
movies['production_companies'].replace(np.nan,"Unknown",inplace=True)
movies['writer'].replace(np.nan,"Unknown",inplace=True)
movies.isnull().sum().sort_values(ascending=False)


# In[581]:


movies.runtimes.fillna(movies.runtimes.mean(), inplace=True)
movies.isnull().sum().sort_values(ascending=False)


# In[582]:


def str_to_list(x):
    return list(x.split(","))


# In[583]:


#convert genres seperated by string to list 
movies['genres'] = movies['genres'].apply(str_to_list)
print(movies['genres'][0])


# In[584]:


#find the frequency of each genre
genre_frequency = {}
unique_genre_set = set()

for row in movies['genres']:
    for genre in row:
        unique_genre_set.add(genre)
        if genre in genre_frequency:
            genre_frequency[genre]+=1
        else:
            genre_frequency[genre]=1
unique_genre_list = list(unique_genre_set)
display(genre_frequency)


# In[585]:


#find the most prevalent genres
sorted_values = sorted(genre_frequency.values(),reverse=True) 
sorted_genre = {}

for i in sorted_values:
    for k in genre_frequency.keys():
        if genre_frequency[k] == i:
            sorted_genre[k] = genre_frequency[k]
            
plt.figure(figsize = (8,6))
sns.barplot(y=list(sorted_genre.keys()),x = list(sorted_genre.values()))


# In[586]:


#wordcloud for genres
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(genre_frequency)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)


# In[587]:


#one hot encoding for genre, director and cast columns
def binary(genre_list):
    binaryList = []
    
    for genre in unique_genre_list:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList


# In[588]:


movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))
movies['genres_bin'].head()
movies.isnull().sum()
display(movies)


# In[589]:


movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')


# In[591]:


for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i[:4]
    movies.loc[j,'cast'] = str(list2)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['cast'] = movies['cast'].str.split(',')
for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'cast'] = str(list2)
movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')


# In[592]:


castList = []
for index, row in movies.iterrows():
    cast = row["cast"]
    
    for i in cast:
        if i not in castList:
            castList.append(i)


# In[593]:


def binary(cast_list):
    binaryList = []
    
    for genre in castList:
        if genre in cast_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList


# In[594]:


movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))
movies['cast_bin'].head()


# In[595]:


def xstr(s):
    if s is None:
        return ''
    return str(s)
movies['director'] = movies['director'].apply(xstr)


# In[596]:


plt.subplots(figsize=(12,10))
ax = movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.5, i, v,fontsize=12,color='white',weight='bold')
plt.title('Directors with highest movies')
plt.show()


# In[597]:


directorList=[]
for i in movies['director']:
    if i not in directorList:
        directorList.append(i)


# In[598]:


def binary(director_list):
    binaryList = []  
    for direct in directorList:
        if direct in director_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList


# In[599]:


movies['director_bin'] = movies['director'].apply(lambda x: binary(x))
movies.head()


# In[600]:


f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(movies['title'].index,movies['vote_count'],color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()


# In[601]:


movies['year'] = movies['release_date'].dt.year
year_rev = movies[['year','revenue']].groupby('year').mean()
year_rev.plot(figsize=(18,8))


# In[602]:


#Top 10 movies with highest vote count
movies[['title','revenue', 'year','vote_count']].sort_values('vote_count', ascending=False).head(10)


# In[603]:


#displaying the top 10 movies according to average voting
movies[movies['vote_count'] > 3000][['title','revenue', 'year','vote_average']].sort_values('vote_average', ascending=False).head(10)


# In[604]:


#Movie runtime trend
sns.distplot(movies['runtimes'])


# In[605]:


#movies released by year
all_year = movies.groupby('year')['title'].count()
all_year.plot(figsize=(18,5))


# In[612]:


#average vote count per decade
def extract_decade(x):
    return str(math.floor(x/10)*10)+"s"
movies["decade"] = movies["year"].apply(extract_decade)
movies = movies.sort_values(by=['decade'], ascending=True)
movies_by_vote = movies.groupby(['decade']).vote_count.sum().reset_index()
bar_data = [go.Bar(x=movies_by_vote['decade']                   , y=movies_by_vote["vote_count"],
                     marker=dict(
                        color='rgb(127,188,65)'
                    ))]

py.offline.iplot({ 'data': bar_data,
            'layout': {
               'title': 'Vote Count for each Decade',
               'xaxis': {
                 'title': 'Decade'},
               'yaxis': {
                'title': 'Total Votes'}
        }})


# In[615]:


#finding directors who directed more than 5 movies with highest ratings
director_df = movies.groupby('director', as_index=False)
director_df = director_df.mean()
name_counts = movies['director'].value_counts().to_dict() # dictionary of director and number of rows/movies per
director_df['film_count'] = director_df['director'].map(name_counts)
director_df['director+count'] = director_df['director'].map(str) + " (" + director_df['film_count'].map(str) + ")"
dir_means = director_df[['director+count',  'film_count']]
dir_subset = director_df[director_df['film_count'] > 5]
top10rat = dir_subset.sort_values(ascending = False, by = 'vote_average')[['director+count', 'vote_average']].head(10)
top10rat.head(3)


# In[620]:


#director vs ratings plot

data = [go.Bar(
            x=top10rat['vote_average'],
            y=top10rat['director+count'],
            orientation = 'h',
            marker=dict(
            color='rgb(67,162,202)'
        )
)]

layout = dict(
        title='Average Movie Rating for Directors who have directed more than 5 movies',
        margin=go.Margin(
        l=210,
        r=100,
        b=100,
        t=100,
        pad=1),
            xaxis=dict(
            title='Average Rating'
        ),
    
        yaxis=dict(
            title='&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Director (Number of Movies)',
            tickfont=dict(
                size=12,
            )
        )
    
    )

fig = go.Figure(data = data, layout = layout)

iplot(fig)


# In[ ]:




