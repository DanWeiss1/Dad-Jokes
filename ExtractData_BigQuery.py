import pandas
import pandas_gbq

# set up name to project
projectid = "reddit-jokes-204821"
# point to key file
key = 'your_big_query_key_file'
# pull jokes from 2017 into data frame filtering out removed and deleted
data_frame = pandas_gbq.read_gbq('''SELECT title, selftext, score, num_comments
FROM [fh-bigquery.reddit_posts.2017_12],
[fh-bigquery.reddit_posts.2017_11],
[fh-bigquery.reddit_posts.2017_10],
[fh-bigquery.reddit_posts.2017_09],
[fh-bigquery.reddit_posts.2017_08],
[fh-bigquery.reddit_posts.2017_07],
[fh-bigquery.reddit_posts.2017_06],
[fh-bigquery.reddit_posts.2017_05],
[fh-bigquery.reddit_posts.2017_04],
[fh-bigquery.reddit_posts.2017_03],
[fh-bigquery.reddit_posts.2017_02],
[fh-bigquery.reddit_posts.2017_01]
WHERE lower(subreddit) = "dadjokes" AND lower(selftext) NOT IN ('[removed]','[deleted]')
''', projectid, private_key=key)

# store jokes in PKL file for later analysis
data_frame.to_pickle('data/dadjokes.pkl')
