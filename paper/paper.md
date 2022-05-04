## Introduction: 
What is a recommender system?
A recommender system is a filtration program whose aim is to provide the most relevant information to a user by discovering patterns in a dataset. The algorithm rates the items and shows the user the items that they would rate highly. An example of recommendation in action is when you visit amazon and you notice that some items are being recommended to you or when Netflix recommends certain movies to you.  The domain-specific item is a movie; therefore, the main focus of our recommendation system is to filter and predict only those movies which a user would prefer given some data about the user him or herself. 
Different filtration strategies: <br >
## Methods: 
Given the task of recommending movies to the users, below are the methods we used to tackle the problem. <br >
### Method 1: KNN collaborative filtering algorithm
KNN collaborative filtering algorithm, is a combination of both collaborative filtering algorithm and KNN algorithm.<br >
The KNN algorithm is used to select the neighbors. We find k movies which are similar in distance with the input movie by the user. <br >
###### COSINE COMPUTING
The method computes the closeness between two users by figuring the cosine of the point between the two vectors <br >

where Ai and Bi are parts of vector A and B respectively.  <br >
The subsequent closeness ranges from −1 meaning precisely inverse, to 1 meaning precisely the equivalent, with 0 demonstrating symmetry or decorrelation, while in the middle of qualities show halfway similarity or uniqueness. For content coordinating, the characteristic vectors An and B are generally the term recurrence vectors of the reports. Cosine closeness can be viewed as a technique for normalizing document length during comparison. <br >
We calculate cosine similarity of genres, director, cast between the input movie and the other movies. <br >
###### KNN NEAREST NEIGHBOR SELECTION
After the calculation of similarity as Similarity(movieA, movieB) between movies, then the algorithm selects a number movies that are nearest to the input movie. <br > 
Select just the most K high similitude as neighbors. As shown in figure below. <br >

### Method 2: K -Means Clustering on Movie Dataset
K means is one of the most popular Unsupervised Machine Learning Algorithms Used for Solving Classification Problems. K Means segregates the unlabeled data into various groups, called clusters, based on having similar features, common patterns. It is an Iterative algorithm that divides a group of n datasets into k subgroups /clusters based on the similarity and their mean distance from the centroid of that particular subgroup/ formed. K, here is the pre-defined number of clusters to be formed by the Algorithm. If K=3, It means the number of clusters to be formed from the dataset is 3. The number of clusters that we choose for the algorithm shouldn’t be random. Each and Every cluster is formed by calculating and comparing the mean distances of each data point within a cluster from its centroid.

Implementation:
Following steps explain how to perform k- Means clustering with python.
1. Import all the relevant libraries.
2. Load the data using pandas
3. Select the necessary features that need to be analyzed for forming the cluster.
4. In this step, we use the inbuilt feature kMeans and perform the clustering analysis. The clustered data is the form of an array.
5. With the help of the matplotlib library in python, the clusters are visualized as scatter plots with all the necessary features.





