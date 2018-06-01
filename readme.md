# Dad Jokes

Analysis of dad jokes subreddit on Reddit.com using unsupervised learning to cluster jokes by words and supervised learning to attempt to predict score
 

## Quick start
  
To recreate the the project, complete the following:
1. Download Repo
2. Set up a google bigquery account and key
3. Modify ExtractData_BigQuery.py to link to your BigQuery account and key files
4. run Cluster_Jokes.py to produce joke clusters and visualizations
5. run Keras_dad_jokes.ipynb to create supervised deep learning model
6. joke_generator.ipynb was a (failed) attempt to generate jokes character by character using an LSTM architecture trained on a corpus of reddit jokes. 

## Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the 
appropriate packages, please install [Anaconda](https://www.continuum.io/downloads)
You will also need the following packages (install using pip install):
pandas-gbq
keras
tensorflow
gensim