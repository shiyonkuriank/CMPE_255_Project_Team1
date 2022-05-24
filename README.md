# CMPE_255_Project_Team1
### Project title: Movie Recommendation System ###
### __Team Members__ ###

[shiyon](https://github.com/shiyonkuriank)

[Harshini](https://github.com/HarshiniKomali)

[Amika](https://github.com/AmikaMehta123)

[Sai Harsha](https://github.com/sreeharsha-glitch)
### Dataset ###
The dataset we have used is https://files.grouplens.org/datasets/movielens/ml-latest-small.zip. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.
### Problem Description ###
We all watch different movies and it is one of the best entertaining activities we can get engaged in. But sometimes it becomes difficult to decide on which movie to watch when provided with a lot of options to choose from. Generally, we would ask our friends or colleagues who have an idea of our interests to recommend movies. We also watch those movies which are popular among most users or have a good rating on the rating websites. We tend to watch those movies suggested by our favorite personalities. But, it is not always feasible to ask someone for suggestions. So it would be convenient if there is a recommendation system that can recommend movies that suit our interests.
### Potential Methods Used ###
Recommendation systems are widely used in almost every application that basically performs the filtering of items in order to recommend the user with the most similar product the user prefers to like. In our project, we are trying to recommend the movies which the user might like based on the input provided by the user. So, we filter out the movies out of the given movies from the dataset. 
There are three types of recommendation systems:
Collaborative Filtering
Content-Based Filtering
Hybrid Recommendation Systems
### Steps to run the code
1. Download the final_code folder
2. Download the requirements.txt file
3. Run the command pip install -r requirements.txt to install the required modules.
4. Run the DataPreprocessing.ipynb file to download the dataset, clean the data and visualise it.
5. To run the content-based model, run the Content_Based_Filtering_Method.ipynb file
6. To run the different collaborative-models, run one of the ItemBasedCF_SVD.ipynb, ItemBasedCF_SVDp.ipynb, ItemBasedCF_KNN.ipynb, ItemBasedCF_KNNMeans.ipynb
7. Before running the Hybrid-based model, run the ItemBasedCF_SVD.ipynb and ItemBasedCF_SVDpp.ipynb if you have not run it before hand. This creates results from the two algorithms in svd_predictions.csv and svdpp_predictions.csv files. These will be used in the hybrid approach.
8. To run the hybrid model, run HybridRecommender.ipynb file. If you give an existing user, the model runs collabortaive filtering method. If you give a non-existing, you are asked to choose between option 1 which recommends based on Popularity and option 2 which recommends based on input movie.

