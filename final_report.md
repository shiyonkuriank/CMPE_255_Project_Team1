## Abstract
Every year new movies are released with a varied story-line or a genre which could be of potential interest to viewers. Various online movie or video streaming platforms can keep the customers engaged by recommending movies of the viewer's preference. A key research challenge for Recommender engines is makes more targeted recommendations. This paper presents Filtering approaches including Content-based, which recommends items (movies) to the user (viewer) based on their previous history/ preferences and Collaborative-based which uses opinions and actions of other similar users (viewers) to recommend items (movies). In Collaborative filtering, User-based, Item based, SVD, and SVD++ algorithms have been implemented and the performance evaluated. Finally, a hybrid recommendation engine that stacks both the Content-based and SVD filtering models is shown to have optimal performance and improved movie recommendations to retain active viewer engagement with the service.

## Introduction
In this hustling world, entertainment is a necessity for each one of us to refresh our mood and energy. Entertainment regains our confidence for work and we can work more enthusiastically. For revitalizing ourselves, we can listen to our preferred music or can watch movies of our choice. For watching favorable movies online, we can utilize movie recommendation systems, which are more reliable, since searching of preferred movies will require more and more time which one cannot afford to waste. In this paper, to improve the quality of a movie recommendation system, a Hybrid approach by combining content based filtering and collaborative filtering. Hybrid approach helps to get the advantages from both the approaches as well as tries to eliminate the drawbacks of both methods. What is a recommender system?
A recommender system is a filtration program whose aim is to provide the most relevant information to a user by discovering patterns in a dataset. The algorithm rates the items and shows the user the items that they would rate highly. An example of recommendation in action is when you visit amazon and you notice that some items are being recommended to you or when Netflix recommends certain movies to you. The domain-specific item is a movie; therefore, the main focus of our recommendation system is to filter and predict only those movies which a user would prefer given some data about the user him or herself. 
Different filtration strategies:<br>
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/filtration_strategies.png)
###### 1. Content-Based Filtering <br>
This filtration strategy is based on the data provided about the items. The algorithm recommends products that are similar to the ones that a user has liked in the past. This similarity (generally cosine similarity) is computed from the data we have about the items as well as the user’s past preferences. For example, if a user likes movies such as ‘The Prestige’ then we can recommend him the movies of ‘Christian Bale’ or movies with the genre ‘Thriller’ or maybe even movies directed by ‘Christopher Nolan’. So, what happens here the recommendation system checks the past preferences of the user and find the film “The Prestige”, then tries to find similar movies to that using the information available in the database such as the lead actors, the director, genre of the film, production house, etc. and based on this information find movies similar to “The Prestige”.

###### 2. Collaborative Filtering <br>
This filtration strategy is based on the combination of the user’s behavior and comparing and contrasting that with another users’ behavior in the database. The history of all users plays an important role in this algorithm. The main difference between content-based filtering and collaborative filtering that in the latter, the interaction of all users with the items influences the recommendation algorithm while for content-based filtering only the concerned user’s data is considered. There are multiple ways to implement collaborative filtering but the main concept to be grasped is that in collaborative filtering multiple user’s data influences the outcome of the recommendation. and doesn’t depend on only one user’s data for modeling. There are 2 types of collaborative filtering algorithms:
• User-based Collaborative filtering <br>
The basic idea here is to find users that have similar past preference patterns as the user ‘A’ has had and then recommending him or her items liked by those similar users which ‘A’ has not encountered yet. This is achieved by making a matrix of items each user has rated/viewed/liked/clicked depending upon the task at hand, and then computing the similarity score between the users and finally recommending items that the concerned user isn’t aware of but users similar to him/her are and liked it. For example, if the user ‘A’ likes ‘Batman Begins’, ‘Justice League’ and ‘The Avengers’ while the user ‘B’ likes ‘Batman Begins’, ‘Justice League’ and ‘Thor’ then they have similar interests because we know that these movies belong to the super-hero genre. So, there is a high probability that the user ‘A’ would like ‘Thor’ and the user ‘B’ would like The Avengers’.
• Item-based Collaborative Filtering <br>
The concept in this case is to find similar movies instead of similar users and then recommending similar movies to that ‘A’ has had in his/her past preferences. This is executed by finding every pair of items that were rated/viewed/liked/clicked by the same user, then measuring the similarity of those rated/viewed/liked/clicked across all user who rated/viewed/liked/clicked both, and finally recommending them based on similarity scores. Here, for example, we take 2 movies ‘A’ and ‘B’ and check their ratings by all users who have rated both the movies and based on the similarity of these ratings, and based on this rating similarity by users who have rated both we find similar movies. So, if most common users have rated ‘A’ and ‘B’ both similarly and it is highly probable that ‘A’ and ‘B’ are similar, therefore if someone has watched and liked ‘A’ they should be recommended ‘B’ and vice versa. • Other algorithms: There are other approaches like market basket analysis, which works by looking for combinations of items that occur together frequently in transactions.

###### 3. Hybrid Recommendation System <br>
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/hybrid.png) <br>
Recent research has demonstrated that a hybrid approach, combining collaborative filtering and content-based filtering could be more effective in some cases. Hybrid approaches can be implemented in several ways, by making content-based and collaborative-based predictions separately and then combining them, by adding content-based capabilities to a collaborative-based approach (and vice versa), or by unifying the approaches into one model. Netflix is a good example of the use of hybrid recommender systems. The website makes recommendations by comparing the watching and searching habits of similar users (i.e. collaborative filtering) as well as by offering movies that share characteristics with films that a user has rated highly (content-based filtering).

## Methods and Analysis
###### 1. Item Item Based Collaborative Filtering
Item-based collaborative filtering is a recommendation system that uses item similarity and user ratings to make recommendations. This method is based on the idea that users give similar ratings to similar items. To develop an efficient recommender system, many methods are being investigated. Because it is domain-free, the Collaborative Filtering (CF) recommender system beats the Content-based recommender system. Item-based CF (IBCF) is a well-known technique in the field of CF that gives accurate suggestions and has been employed by applications that provide product recommendation systems. In this section we explain the implementation of Item-based Collaborative Filtering using Python to build a movie recommendation system.

![1_3ALliiz9hG79_2xopzgyrQ](https://user-images.githubusercontent.com/90216358/169903447-11c90259-e859-477d-a535-4af89de19eb5.png)


As shown in the figure above, The similarities between movies are calculated first. Second, movies similar to those already rated are examined and recommended based on the computed similarities.
Item based collaborative Filtering models are developed using machine learning algorithms to predict movies for one user. Here we use KNN Model. The k-nearest neighbors (KNN) approach relies on item feature similarity rather than making any assumptions about the underlying data distribution. When a KNN predicts a movie, it calculates the 'distance' between the target movie and all other movies in its database. The top 'k' nearest neighbor movies (with the shortest distance measure) are then returned as the most similar movie choices.
Lets move to the implementation of the Item-based Collaborative Filtering using Python.


###### 1.1 Implementation with KNN and KNNWith Means Model for Prediction of 10 Movies
The dataset we have used is https://files.grouplens.org/datasets/movielens/ml-latest-small.zip. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.

All the necessary libraries are imported at the beginning for simplicity. 
As we'll be coding in Google Colab, we'll need to upload the movies and ratings csv files first. Once the file is uploaded we read the file contents using Pandas.

Data Preprocessing
The 'rating' file contains ratings given by users to the movies. Seaborn is used to visualize the number of ratings provided by different users for different ratings. Figure shows the distribution of the ratings.csv file.
![image](https://user-images.githubusercontent.com/90216358/169903960-90ac436c-8929-40a8-86a0-b924401510f4.png)
Fig 1.
The ‘movies’ file contains the movie's id, title, and genres. Based on the movieId, both the movies and ratings are combined. The movies are then grouped and the total rating count for each is calculated. This will aid in comprehending the dispersion of movie ratings. There are many movies with only one rating provided by the user. We are filtering the movies in such a way that movies with minimum 3 ratings are only considered. Similarly, we group the users in such a way that those rated greater than or equal to 50.

KNN collaborative filtering algorithm, is a combination of both collaborative filtering algorithm and KNN algorithm.
We want the data for K-Nearest Neighbors to be in an array, with each row representing a movie and each column representing a separate user. We'll pivot the data frame to a wide format with movies as rows and users as columns to reshape it. Since we'll be conducting linear algebra operations for calculating distances between vectors, we'll fill in the missing observations with 0s . Finally, we convert the dataframe's values into a scipy sparse matrix for faster calculations.
![image](https://user-images.githubusercontent.com/90216358/169904155-b0cd2e27-11a4-4221-a0fc-11ea5e90bb42.png)

The KNN algorithm is used to select the neighbors. In this project we use k value to be 10. Ten movies which are similar in distance with the input movie by the user are calculated and recommended to the user.

COSINE COMPUTING
Here we use cosine similarity, a method that computes the closeness between two movies by figuring the cosine of the point between the two vectors

where Ai and Bi are parts of vectors A and B respectively.
The subsequent closeness ranges from −1 meaning precisely inverse, to 1 meaning precisely the equivalent, with 0 demonstrating symmetry or decorrelation, while in the middle qualities show halfway similarity or uniqueness. For content coordinating, the characteristic vectors A and B are generally the term recurrence vectors of the reports. Cosine closeness can be viewed as a technique for normalizing document length during comparison.
We calculate cosine similarity between the input movie and the other movies.
KNN NEAREST NEIGHBOR SELECTION
After the calculation of similarity as Similarity between movies, then the algorithm selects a number of movies that are nearest to the input movie. Here, the KNN model selects 10 movies that are similar to the input movie. Select just the 10  high similitudes as neighbors. As shown in figure below.

![image](https://user-images.githubusercontent.com/90216358/169904265-c2e7b9c9-3066-4856-b0f2-3463fb783534.png)

To test the model, we gave a sample input of  'Toy story' to see if our model gives 10 recommendations. The figure below shows the output of our implementation.

![image](https://user-images.githubusercontent.com/90216358/169904362-16ffce39-bb12-4547-bcee-2c4f17d16b7d.png)

Thus our KNN model successfully recommended 10 movies that were similar to the input movie.

RMSE SCORE
The root mean squared error, or RMSE, is the most used statistic for measuring the performance of a prediction model. The primary concept is to compare the model's predictions to actual observed data to see how bad/wrong they are. As a result, a high RMSE is "bad," whereas a low RMSE is "excellent."
Surprise is a Python scikit for creating and evaluating recommender systems using explicit rating data. It provides tools for evaluating, analyzing, and comparing the performance of algorithms. The surprise package has many builtin methods. One Of that is RMSE. For the movie recommendation system that we have built, we are calculating the RMSE score to see the effectiveness of the model.
We import, Reader, Dataset and cross_validate functions to perform this. We split the dataset into train and test data. Since we have used the KNN model, we need to import the KNNBasic and compute the RMSE. The RMSE score for KNN Model is shown in the figure below.

![image](https://user-images.githubusercontent.com/90216358/169904459-d8cc837a-5104-4a43-baf3-b92894a461ec.png)

KNNWithMeans is a basic collaborative filtering algorithm that considers each user's mean ratings. The RMSE score for KNN Model is shown in the figure below.

![image](https://user-images.githubusercontent.com/90216358/169904556-5bcedaaa-fd0c-432a-a6cd-ee7af596a921.png)

