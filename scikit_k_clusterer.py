from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import os
import io
import sys
import glob
import errno
import operator
import string
import re, math


path=input("Please enter path of your Corpus : ")
true_k=input("Please enter value of 'k' : ")

####################################
# Get contents of corpus as Tokens #
####################################


documents=[]
filenames=[os.path.join(path,fn) for fn in next(os.walk(path))[2]]
for corpus_file in filenames: 
    with open(corpus_file,errors="ignore") as myfile:
        data=" ".join(line.rstrip() for line in myfile)
        documents.append(data)


####################################
# Perform Clustering               #
####################################


true_k = int(true_k)
vectorizer = TfidfVectorizer(stop_words='english',decode_error='ignore')
X = vectorizer.fit_transform(documents)
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


######################################
# Print top 10 terms in each cluster #
######################################


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    j=0
    print ("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print (' %s' % terms[ind]),
    print
