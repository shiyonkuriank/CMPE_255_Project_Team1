#### Introduction: ###
What is a recommender system?
A recommender system is a filtration program whose aim is to provide the most relevant information to a user by discovering patterns in a dataset. The algorithm rates the items and shows the user the items that they would rate highly. An example of recommendation in action is when you visit amazon and you notice that some items are being recommended to you or when Netflix recommends certain movies to you.  The domain-specific item is a movie; therefore, the main focus of our recommendation system is to filter and predict only those movies which a user would prefer given some data about the user him or herself. 
Different filtration strategies: 
### Methods: <br >
Given the task of recommending movies to the users, below are the methods we used to tackle the problem. <br >
### Method 1: Movie recommendation using KNN classifier ###

### Method 2: K -Means Clustering on Movie Dataset: ###


K means is one of the most popular Unsupervised Machine Learning Algorithms Used for Solving Classification Problems. K Means segregates the unlabeled data into various groups, called clusters, based on having similar features, common patterns. It is an Iterative algorithm that divides a group of n datasets into k subgroups /clusters based on the similarity and their mean distance from the centroid of that particular subgroup/ formed. K, here is the pre-defined number of clusters to be formed by the Algorithm. If K=3, It means the number of clusters to be formed from the dataset is 3. The number of clusters that we choose for the algorithm shouldn’t be random. Each and Every cluster is formed by calculating and comparing the mean distances of each data point within a cluster from its centroid.

Implementation:
Following steps explain how to perform k- Means clustering with python.
Import all the relevant libraries.
Load the data using pandas
Select the necessary features that need to be analyzed for forming the cluster.
In this step, we use the inbuilt feature kMeans and perform the clustering analysis. The clustered data is the form of an array.
With the help of the matplotlib library in python, the clusters are visualized as scatter plots with all the necessary features.
