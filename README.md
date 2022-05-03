# CMPE_255_Project_Team1
### Project title: Movie Recommendation System ###
### __Team Members__ ###

[shiyon](https://github.com/shiyonkuriank)

[Harshini](https://github.com/HarshiniKomali)

[Amika](https://github.com/AmikaMehta123)

[Sai Harsha](https://github.com/sreeharsha-glitch)
### Dataset ###
The dataset we use for our project is found at  https://grouplens.org/datasets/movielens/100k/. It consists of about 100,000 ratings from more than 900 users. Each user has rated at least 20 movies. 
We also use https://data.world/patriciag/tmdb-data dataset which contains extensive data about movies, tv shows and ratings from the users.
### Problem Description ###
We all watch different movies and it is one of the best entertaining activities we can get engaged in. But sometimes it becomes difficult to decide on which movie to watch when provided with a lot of options to choose from. Generally, we would ask our friends or colleagues who have an idea of our interests to recommend movies. We also watch those movies which are popular among most users or have a good rating on the rating websites. We tend to watch those movies suggested by our favorite personalities. But, it is not always feasible to ask someone for suggestions. So it would be convenient if there is a recommendation system that can recommend movies that suit our interests.
### Potential Methods Used ###
Recommendation systems are widely used in almost every application that basically performs the filtering of items in order to recommend the user with the most similar product the user prefers to like. In our project, we are trying to recommend the movies which the user might like based on the input provided by the user. So, we filter out the movies out of the given movies from the dataset. 
There are three types of recommendation systems:
Collaborative Filtering
Content-Based Filtering
Hybrid Recommendation Systems
 
Collaborative Filtering:
There are several types of collaborative filtering. 
User based collaborative filtering: This works by trying to search for lookalike customers and recommend movies based on what the person’s lookalike has watched.
Item based collaborative filtering: This is similar to user-based but instead of finding the lookalike, we find item lookalike.
Other algorithms: There are other approaches which use market-basket analysis, which work by looking for combination of items that occur together frequently in a transaction.
Content-based Filtering:
These filtering methods are based on the description of an item and a profile of the user’s preferred choices. In a content-based recommendation system, keywords are used to describe the items, besides, a user profile is built to state the type of item this user likes. In other words, the algorithms try to recommend products that are similar to the ones that a user has liked in the past.
One of the major issues with the recommendation system that uses collaborative filtering is cold start and data sparsity. Here we use one of the Supervised Learning technique K-Nearest Neighbors approach in order to decrease the error rate and address the cold start and data sparsity issues.
So, we are using a Hybrid approach which uses both collaborative and content-based filtering. Hybrid approaches can be implemented in several ways, by making content-based and collaborative-based predictions separately and then combining them, by adding content-based capabilities to a collaborative-based approach (and vice versa), or by unifying the approaches into one model. Netflix is a good example of hybrid recommender systems.

