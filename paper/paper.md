## Introduction: 
What is a recommender system?
A recommender system is a filtration program whose aim is to provide the most relevant information to a user by discovering patterns in a dataset. The algorithm rates the items and shows the user the items that they would rate highly. An example of recommendation in action is when you visit amazon and you notice that some items are being recommended to you or when Netflix recommends certain movies to you.  The domain-specific item is a movie; therefore, the main focus of our recommendation system is to filter and predict only those movies which a user would prefer given some data about the user him or herself. 
Different filtration strategies: <br >

######1. Content-based Filtering: <br>
This filtration strategy is based on the data provided about the items. The algorithm recommends products that are similar to the ones that a user has liked in the past. This similarity (generally cosine similarity) is computed from the data we have about the items as well as the user’s past preferences. For example, if a user likes movies such as ‘The Prestige’ then we can recommend him the movies of ‘Christian Bale’ or movies with the genre ‘Thriller’ or maybe even movies directed by ‘Christopher Nolan’.  So, what happens here the recommendation system checks the past preferences of the user and find the film “The Prestige”, then tries to find similar movies to that using the information available in the database such as the lead actors, the director, genre of the film, production house, etc. and based on this information find movies similar to “The Prestige”.
######2. Collaborative Filtering: <br>
This filtration strategy is based on the combination of the user’s behavior and comparing and contrasting that with another users’ behavior in the database. The history of all users plays an important role in this algorithm. The main difference between content-based filtering and collaborative filtering that in the latter, the interaction of all users with the items influences the recommendation algorithm while for content-based filtering only the concerned user’s data is considered.
There are multiple ways to implement collaborative filtering but the main concept to be grasped is that in collaborative filtering multiple user’s data influences the outcome of the recommendation. and doesn’t depend on only one user’s data for modeling. 
There are 2 types of collaborative filtering algorithms:<br>
•	User-based Collaborative filtering<br>
The basic idea here is to find users that have similar past preference patterns as the user ‘A’ has had and then recommending him or her items liked by those similar users which ‘A’ has not encountered yet. This is achieved by making a matrix of items each user has rated/viewed/liked/clicked depending upon the task at hand, and then computing the similarity score between the users and finally recommending items that the concerned user isn’t aware of but users similar to him/her are and liked it.
For example, if the user ‘A’ likes ‘Batman Begins’, ‘Justice League’ and ‘The Avengers’ while the user ‘B’ likes ‘Batman Begins’, ‘Justice League’ and ‘Thor’ then they have similar interests because we know that these movies belong to the super-hero genre. So, there is a high probability that the user ‘A’ would like ‘Thor’ and the user ‘B’ would like The Avengers’.
•	Item-based Collaborative Filtering<br>
The concept in this case is to find similar movies instead of similar users and then recommending similar movies to that ‘A’ has had in his/her past preferences. This is executed by finding every pair of items that were rated/viewed/liked/clicked by the same user, then measuring the similarity of those rated/viewed/liked/clicked across all user who rated/viewed/liked/clicked both, and finally recommending them based on similarity scores.
Here, for example, we take 2 movies ‘A’ and ‘B’ and check their ratings by all users who have rated both the movies and based on the similarity of these ratings, and based on this rating similarity by users who have rated both we find similar movies. So, if most common users have rated ‘A’ and ‘B’ both similarly and it is highly probable that ‘A’ and ‘B’ are similar, therefore if someone has watched and liked ‘A’ they should be recommended ‘B’ and vice versa.
•	Other algorithms: There are other approaches like market basket analysis, which works by looking for combinations of items that occur together frequently in transactions.
#######3. Hybrid Recommendation System <br>
![hybrid](https://user-images.githubusercontent.com/71619460/166626503-964e642c-ef0c-43bf-b733-7a15d8ae930a.png)
Recent research has demonstrated that a hybrid approach, combining collaborative filtering and content-based filtering could be more effective in some cases. Hybrid approaches can be implemented in several ways, by making content-based and collaborative-based predictions separately and then combining them, by adding content-based capabilities to a collaborative-based approach (and vice versa), or by unifying the approaches into one model.
Netflix is a good example of the use of hybrid recommender systems. The website makes recommendations by comparing the watching and searching habits of similar users (i.e. collaborative filtering) as well as by offering movies that share characteristics with films that a user has rated highly (content-based filtering).
## Methods: 
Given the task of recommending movies to the users, below are the methods we used to tackle the problem. <br >
### Method 1: KNN collaborative filtering algorithm
KNN collaborative filtering algorithm, is a combination of both collaborative filtering algorithm and KNN algorithm.<br >
The KNN algorithm is used to select the neighbors. We find k movies which are similar in distance with the input movie by the user. <br >
![alt text]

###### COSINE COMPUTING
The method computes the closeness between two users by figuring the cosine of the point between the two vectors <br >
![alt text]

where Ai and Bi are parts of vector A and B respectively.  <br >
The subsequent closeness ranges from −1 meaning precisely inverse, to 1 meaning precisely the equivalent, with 0 demonstrating symmetry or decorrelation, while in the middle of qualities show halfway similarity or uniqueness. For content coordinating, the characteristic vectors An and B are generally the term recurrence vectors of the reports. Cosine closeness can be viewed as a technique for normalizing document length during comparison. <br >
We calculate cosine similarity of genres, director, cast between the input movie and the other movies. <br >
###### KNN NEAREST NEIGHBOR SELECTION
After the calculation of similarity as Similarity(movieA, movieB) between movies, then the algorithm selects a number movies that are nearest to the input movie. <br > 
Select just the most K high similitude as neighbors. As shown in figure below. <br >
![alt text]

We are also predicting the rating of the input movie by computing the average rating of its k neighbor movies.






