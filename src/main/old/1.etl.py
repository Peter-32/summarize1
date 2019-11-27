# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Public modules
import re
import os
import nltk
import spacy
import pickle
import numpy as np
import pandas as pd
from os import path
import networkx as nx
import lightgbm as lgb
from pandas import read_csv
from sklearn.svm import SVC
from numpy.random import seed
import matplotlib.pyplot as plt
from sklearn import preprocessing
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
                                  OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, \
                            precision_score, recall_score, accuracy_score

# Functions
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Set seed
seed(32)

# Initialize Helper Objects
print("Initialize Helper Objects")
wnl = nltk.WordNetLemmatizer()
tfidf = TfidfVectorizer()
nlp = spacy.load('en_core_web_md')

# Extract
print("Extract")
with open("../../data/raw/bitcoin.txt") as file:
    doc = file.read()

# Transform
print("Transform")
doc = doc.lower()
tokens = word_tokenize(guide)
text = nltk.Text(tokens)
words = [re.sub(r'[^A-Za-z_\s]', '', w) for w in text]
words = [wnl.lemmatize(w) for w in words if w.strip() != ''] # Will take work later to get the original text back without lemmatize.
sents = nltk.sent_tokenize(guide)

# Get dictionaries
print("Dictionaries")
tfidf.fit(words)
tfidf_weights_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
embeddings_dict = {}
for val in tfidf_weights_dict:
    embeddings_dict[val] = nlp(val).vector

# Averaged Vectors
print("Embeddings")
sent_vectors = []
for sent in sents:
    tokens = word_tokenize(sent)
    text = nltk.Text(tokens)
    words = [re.sub(r'[^A-Za-z_\s]', '', w) for w in text]
    words = [wnl.lemmatize(w) for w in words if w.strip() != '']
    vector_sum, denominator = [0]*300, 0
    for word in words:
        try:
            vector_sum += embeddings_dict[word]*tfidf_weights_dict[word]
            denominator += tfidf_weights_dict[word]
        except:
            pass
    if denominator != 0:
        sent_vectors.append(vector_sum/denominator)
    else:
        sent_vectors.append(vector_sum)

# Similarity Matrix
print("Similarity matrix")
sim_mat = np.zeros([len(sents), len(sents)])
for i in range(len(sents)):
    for j in range(len(sents)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sent_vectors[i].reshape(1,300), sent_vectors[j].reshape(1,300))[0,0]
print(sim_mat)

# Graph
print("GraphX")
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
print(scores)

save_obj(scores, "scores")


print(len(sent_vectors))
print(sent_vectors[-3:])
print(scores)
# words = [wnl.lemmatize(w) for w in words]
# words = [w for w in words if w not in the_stopwords and w != '']





print(guide[-400:])
