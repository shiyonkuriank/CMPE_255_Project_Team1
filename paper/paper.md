## Introduction: 
In this hustling world, entertainment is a necessity for each one of us to refresh our mood and energy. Entertainment regains our confidence for work and we can work more enthusiastically. For revitalizing ourselves, we can listen to our preferred music or can watch movies of our choice. For watching favorable movies online, we can utilize movie recommendation systems, which are more reliable, since searching of preferred movies will require more and more time which one cannot afford to waste. In this paper, to improve the quality of a movie recommendation system, a Hybrid approach by combining content based filtering and collaborative filtering. Hybrid approach helps to get the advantages from both the approaches as well as tries to eliminate the drawbacks of both methods.
What is a recommender system?<br>
A recommender system is a filtration program whose aim is to provide the most relevant information to a user by discovering patterns in a dataset. The algorithm rates the items and shows the user the items that they would rate highly. An example of recommendation in action is when you visit amazon and you notice that some items are being recommended to you or when Netflix recommends certain movies to you.  The domain-specific item is a movie; therefore, the main focus of our recommendation system is to filter and predict only those movies which a user would prefer given some data about the user him or herself. 
Different filtration strategies: <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/filtration_strategies.png)

###### 1. Content-based Filtering: <br>
This filtration strategy is based on the data provided about the items. The algorithm recommends products that are similar to the ones that a user has liked in the past. This similarity (generally cosine similarity) is computed from the data we have about the items as well as the user’s past preferences. For example, if a user likes movies such as ‘The Prestige’ then we can recommend him the movies of ‘Christian Bale’ or movies with the genre ‘Thriller’ or maybe even movies directed by ‘Christopher Nolan’.  So, what happens here the recommendation system checks the past preferences of the user and find the film “The Prestige”, then tries to find similar movies to that using the information available in the database such as the lead actors, the director, genre of the film, production house, etc. and based on this information find movies similar to “The Prestige”.
###### 2. Collaborative Filtering: <br>
This filtration strategy is based on the combination of the user’s behavior and comparing and contrasting that with another users’ behavior in the database. The history of all users plays an important role in this algorithm. The main difference between content-based filtering and collaborative filtering that in the latter, the interaction of all users with the items influences the recommendation algorithm while for content-based filtering only the concerned user’s data is considered.
There are multiple ways to implement collaborative filtering but the main concept to be grasped is that in collaborative filtering multiple user’s data influences the outcome of the recommendation. and doesn’t depend on only one user’s data for modeling. 
There are 2 types of collaborative filtering algorithms:<br>
•	User-based Collaborative filtering <br>
The basic idea here is to find users that have similar past preference patterns as the user ‘A’ has had and then recommending him or her items liked by those similar users which ‘A’ has not encountered yet. This is achieved by making a matrix of items each user has rated/viewed/liked/clicked depending upon the task at hand, and then computing the similarity score between the users and finally recommending items that the concerned user isn’t aware of but users similar to him/her are and liked it.
For example, if the user ‘A’ likes ‘Batman Begins’, ‘Justice League’ and ‘The Avengers’ while the user ‘B’ likes ‘Batman Begins’, ‘Justice League’ and ‘Thor’ then they have similar interests because we know that these movies belong to the super-hero genre. So, there is a high probability that the user ‘A’ would like ‘Thor’ and the user ‘B’ would like The Avengers’. <br>
•	Item-based Collaborative Filtering <br>
The concept in this case is to find similar movies instead of similar users and then recommending similar movies to that ‘A’ has had in his/her past preferences. This is executed by finding every pair of items that were rated/viewed/liked/clicked by the same user, then measuring the similarity of those rated/viewed/liked/clicked across all user who rated/viewed/liked/clicked both, and finally recommending them based on similarity scores.
Here, for example, we take 2 movies ‘A’ and ‘B’ and check their ratings by all users who have rated both the movies and based on the similarity of these ratings, and based on this rating similarity by users who have rated both we find similar movies. So, if most common users have rated ‘A’ and ‘B’ both similarly and it is highly probable that ‘A’ and ‘B’ are similar, therefore if someone has watched and liked ‘A’ they should be recommended ‘B’ and vice versa.
•	Other algorithms: There are other approaches like market basket analysis, which works by looking for combinations of items that occur together frequently in transactions. 
###### 3. Hybrid Recommendation System <br>
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/hybrid.png)

Recent research has demonstrated that a hybrid approach, combining collaborative filtering and content-based filtering could be more effective in some cases. Hybrid approaches can be implemented in several ways, by making content-based and collaborative-based predictions separately and then combining them, by adding content-based capabilities to a collaborative-based approach (and vice versa), or by unifying the approaches into one model.
Netflix is a good example of the use of hybrid recommender systems. The website makes recommendations by comparing the watching and searching habits of similar users (i.e. collaborative filtering) as well as by offering movies that share characteristics with films that a user has rated highly (content-based filtering). <br>

We used the movies dataset from data.world website. It contains title, cast and crew, genres, revenue, number of votes and average rating for a particular movie. <br >
We dropped those movies which have missing values in genre because it is difficult impute a value for it. <br >
Below is the revenue distribution plot <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/revenue.png)

We can see from the plot there are a lot of 0 values for revenue. So, impute revenue with median revenue of the movies released in that particular year. <br >

We performed one hot encoding on genre, director and cast columns as they are comma separated and it is difficult to use it. <br >

Below is the plot and word  cloud for the popular genres <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/genre_plot_1.png)
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/genre_plot_2.png)

The below plot shows the directors with highest number of movies directed. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/director_high_movies.png)

This is the plot between vote count and movie. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/vote_count.png)

Below is the plot between number of movies released per year. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/years_movies_released.png)

The table shows the top 10 movies based on number of votes. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/top_ten_vote.png)

The table below shows the top 10 highest rated movies. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/top_ten_rating.png)

The plot below shows how the vote count changes over the decades. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/vote_count_decade.png)

The below plot shows the highest average rating of directors who directed at least 5 movies. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/average_rating_director.png)

###### Analysis: <br >
1.	Drama is the most popular genre followed by comedy and thriller.
2.	Woody Allen is the director who directed the highest number of movies.
3.	2009 is the year which has the highest number of movies released.
4.	The vote count increases till 2000s and decreased in the 2010s.
5.	The highest average ratings of directors is almost 7.0.


## Methods: 
Given the task of recommending movies to the users, below are the methods we used to tackle the problem. <br >
### Method 1: KNN collaborative filtering algorithm
KNN collaborative filtering algorithm, is a combination of both collaborative filtering algorithm and KNN algorithm.<br >
The KNN algorithm is used to select the neighbors. We find k movies which are similar in distance with the input movie by the user. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/knn.png)

###### COSINE COMPUTING
The method computes the closeness between two users by figuring the cosine of the point between the two vectors <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/cosine_distance.png)

where Ai and Bi are parts of vector A and B respectively.  <br >
The subsequent closeness ranges from −1 meaning precisely inverse, to 1 meaning precisely the equivalent, with 0 demonstrating symmetry or decorrelation, while in the middle of qualities show halfway similarity or uniqueness. For content coordinating, the characteristic vectors An and B are generally the term recurrence vectors of the reports. Cosine closeness can be viewed as a technique for normalizing document length during comparison. <br >
We calculate cosine similarity of genres, director, cast between the input movie and the other movies. <br >
###### KNN NEAREST NEIGHBOR SELECTION
After the calculation of similarity as Similarity(movieA, movieB) between movies, then the algorithm selects a number movies that are nearest to the input movie. <br > 
Select just the most K high similitude as neighbors. As shown in figure below. <br >
![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/knn_2.png)

We are also predicting the rating of the input movie by computing the average rating of its k neighbor movies.
###### Steps:
1. Clean the dataset by imputing null values is possible and selecting only those features which are useful.
   Here we chose, vote_count, average_rating, title, revenue, cast, director, genres.
2. We chose k = 10 to show top 10 movie suggestions the user might be interested in.
3. The input is the movie entered by the user. Compute the cosine distance between every movie in the dataset and the input movie.
4. Output top 10 movies which have higher similitude.
5. Compute the average of the ratings of the selected 10 movies and use that value as the predicted rating for the input movie.
The below figures shows the movie results for three movies.<br >
   For "Godfather"<br >
   ![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/godfather.png)

   For "Finding Nemo" <br >
   ![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/finding_nemo.png)

   For "The Shawshank Redemption"<br >
   ![alt text](https://github.com/shiyonkuriank/CMPE_255_Project_Team1/blob/main/images/shawshank.png)

###### Advantages:
1. Since we are recommending movies based on similarity based on genre, director and cast, there are high chances that the user might watch the recommended movies. <br >
2. Easier to recommend movies. <br >
###### Disadvantages:
1. This method considers only the movies and their average ratings, genres, cast and crew but do not consider what the particular user liked or disliked.
    So, it cannot precisely recommend movies based on the preference.
2. We chose K = 10 arbitrarily to show top 10 movies. But in practice, K should be determined because the value of K affects the outcome.

### K -Means Clustering on Movie Dataset ###
K means is one of the most popular Unsupervised Machine Learning Algorithms Used for Solving Classification Problems. K Means segregates the unlabeled data into various groups, called clusters, based on having similar features, common patterns. It is an Iterative algorithm that divides a group of n datasets into k subgroups /clusters based on the similarity and their mean distance from the centroid of that particular subgroup/ formed. K, here is the pre-defined number of clusters to be formed by the Algorithm. If K=3, It means the number of clusters to be formed from the dataset is 3. The number of clusters that we choose for the algorithm shouldn’t be random. Each and Every cluster is formed by calculating and comparing the mean distances of each data point within a cluster from its centroid.

Implementation:
Following steps explain how to perform k- Means clustering with python.
Import all the relevant libraries.
Load the data using pandas
Select the necessary features that need to be analyzed for forming the cluster.
In this step, we use the inbuilt feature kMeans and perform the clustering analysis. The clustered data is the form of an array.
With the help of the matplotlib library in python, the clusters are visualized as scatter plots with all the necessary features.








