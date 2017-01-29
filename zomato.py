from __future__ import division
from __future__ import print_function

import urllib2
import json
import binascii
from sys import stdout
import time


import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import MDS

nltk.download("stopwords")
nltk.download('punkt')


val_api_key = "0c80b8977020170ea8484bcca5b2f9ba"#"5cc0501c54b5188498321689ebd9ca8b"
key_api_key = "user-key"


api_url_search = "https://developers.zomato.com/api/v2.1/search?q=Delhi%20NCR&sort=rating&order=asc&start="
api_url_reviews = "https://developers.zomato.com/api/v2.1/reviews?res_id="
start = 0
rest_ids = []
titles = []
rest_ratings = []
reviews = []

def prints(s):
	stdout.write(s + "\r")
def normalizeStr(inp):
	return inp.encode('utf-8').decode('string_escape').decode("unicode_escape")

def getReviews(res_id):
	url = api_url_reviews + str(res_id)
	all_reviews = ""
	request = urllib2.Request(url, headers={key_api_key : val_api_key})
	contents = urllib2.urlopen(request).read()
	if contents != "":
		review_data = json.loads(contents)
		reviewsJson = review_data["user_reviews"]
		for review in reviewsJson:
			all_reviews += (review["review"]["review_text"]).encode('ascii', errors='ignore') + "\n"
	reviews.append(all_reviews)

print("Loading restaurants...")
for i in range(3):
	stdout.flush()
	prints(" %.1f" %((i+1)*100/3) + "%")
	api_url_search_temp = api_url_search + str(start)
	start += 20
	request = urllib2.Request(api_url_search_temp, headers={key_api_key : val_api_key})
	contents = urllib2.urlopen(request).read()
	restaurant_data = json.loads(contents)
	restaurants = restaurant_data["restaurants"]

	for restaurant in restaurants:
		rest_id = str(restaurant["restaurant"]["R"]["res_id"]).encode('ascii', errors='ignore')
		rating = (restaurant["restaurant"]["user_rating"]["aggregate_rating"]).encode('ascii', errors='ignore')
		rest_name = (restaurant["restaurant"]["name"] + " : " + restaurant["restaurant"]["location"]["locality"]).encode('ascii', errors='ignore')
		rest_ids.append(rest_id)
		titles.append(rest_name)
		rest_ratings.append(rating)

stdout.flush()
print("Loading reviews...")
review_count = 1
for rest_id in rest_ids :
	getReviews(rest_id)
	stdout.flush()
	prints(" %.1f" %(review_count*100/len(rest_ids)) + "%")
	review_count += 1
# print(len(restaurant_ids))
# print(restaurant_ids)
print(titles)
print(rest_ratings)
# print(reviews)

############################################## data analysis - clustering using k-means method ##############################################

stopwords = nltk.corpus.stopwords.words('english')

stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    filtered_tokens = tokenize_only(text)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word.lower() for sentence in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sentence)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in reviews:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)


num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

# joblib.dump(km,  'doc_cluster.pkl')

# km = joblib.load('doc_cluster.pkl')
# clusters = km.labels_.tolist()


rest_reviews = { 'title': titles, 'reviews': reviews, 'cluster': clusters}

frame = pd.DataFrame(rest_reviews, index = [clusters] , columns = ['title', 'cluster'])

frame['cluster'].value_counts()


grouped = frame['title'].groupby(frame['cluster'])

# grouped.mean()

print()

order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :-1]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()
    print()
    
    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()
    
print()
print()

MDS()

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)

xs, ys = pos[:, 0], pos[:, 1]


cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

cluster_names = {0: 'time, hygiene, late', 
                 1: 'pathetic, worst, delivery', 
                 2: 'late, worst, wait', 
                 3: 'worst, experience, money', 
                 4: 'money, waste, bad'}


cluster_names = {0: 'time, hygiene, late', 
                 1: 'pathetic, worst, taste', 
                 2: 'late, time, wait', 
                 3: 'worst, experience, hygiene', 
                 4: 'money, waste, prices'}



df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles)) 


groups = df.groupby('label')


fig, ax = plt.subplots(figsize=(17, 9))
ax.margins(0.05)


for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',
        which='both',
        left='off',
        top='off',
        labelleft='off')
    
ax.legend(numpoints=1)

for i in range(len(df)):
    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

plt.show()


