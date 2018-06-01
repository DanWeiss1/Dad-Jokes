import pandas

import gensim
import numpy
from gensim.parsing.preprocessing import strip_short, remove_stopwords, preprocess_string, strip_tags, strip_punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import MiniBatchKMeans, KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

pandas.set_option('display.max_colwidth',-1)
df = pandas.read_pickle('data/dadjokes.pkl')

df['selftext'] = df['selftext'].replace(r'\n',' ', regex=True) 

df['joke_text_raw'] = df['title'] + " " + df['selftext']
df['joke_text_process'] = df['joke_text_raw'].str.lower().apply(strip_punctuation)
df['joke_text_tokens']= (df['joke_text_raw']).apply(lambda x: preprocess_string(x,
                        [lambda x: x.lower(), strip_tags, strip_punctuation, strip_short]))

dups = df[df.duplicated(subset ='joke_text_process',keep=False)]
df = df.drop_duplicates(subset = 'joke_text_process')


tf_idf = TfidfVectorizer(max_df=1.0, min_df=1)

X = tf_idf.fit_transform(df['joke_text_process'])

svd = TruncatedSVD(random_state = 42)
svd.fit(X)
X_transform = svd.transform(X)


# k means determine k
distortions = []
K = range(1,11)
for k in K:
    kmeanModel = MiniBatchKMeans(n_clusters=k, init='k-means++',batch_size=3500,
    random_state=42).fit(X_transform)
    kmeanModel.fit(X_transform)
    distortions.append(kmeanModel.inertia_)
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.xticks(K)
plt.title('The Elbow Method showing the optimal k')
plt.savefig('Graphics/elbow.png')

kmeanModel = MiniBatchKMeans(n_clusters=4, init='k-means++',batch_size=3500,
          random_state=42).fit(X_transform)

for i in range(4):
    df['cluster_dist'] = kmeanModel.transform(X_transform)[:,i]
    df['cluster'] = kmeanModel.predict(X_transform)
    df_temp = df[df['cluster'] == i]
    print('Cluster' + str(i))
    print(df_temp.shape)
    print(df_temp.sort_values(by='cluster_dist', ascending=False)['joke_text_raw'].head(5))




tim = """you're American when you go into the bathroom, and you're 
American when you come out, but do you know what you are while you're in there?
European"""
ben = "What did the buffalo say to his son when he left for college? Bison"

tim_tran = tf_idf.transform([tim])
ben_tran = tf_idf.transform([ben])
plt.title("Joke Clusters by Top Principal Components")
plt.xlabel("1st principle component")
plt.ylabel("2nd principle component")
plt.scatter(X_transform[:,0],X_transform[:,1],c=kmeanModel.predict(X_transform), alpha=.1)
plt.text(svd.transform(ben_tran)[:,0],svd.transform(ben_tran)[:,1],"Ben",bbox=dict(facecolor='none', edgecolor='black'))
plt.text(svd.transform(tim_tran)[:,0],svd.transform(tim_tran)[:,1],"Tim",bbox=dict(facecolor='none', edgecolor='black'))
plt.savefig("Graphics/jokeclusters.png")
