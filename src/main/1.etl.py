# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Public modules
import re
import pandas as pd
from pandas import read_csv
from numpy.random import seed
from sklearn.model_selection import train_test_split
import re
import pandas as pd
from pandas import read_csv
from numpy.random import seed
from sklearn.model_selection import train_test_split
import numpy as np
from os import path
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import precision_recall_curve, confusion_matrix, \
                            precision_score, recall_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from numpy.random import seed
import lightgbm as lgb
import re
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.random_projection import GaussianRandomProjection
from pandas import read_csv
from numpy.random import seed
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, \
                                  OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
import nltk
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Set seed - ensures that the datasets are split the same way if re-run
seed(32)

# Initialize
wnl = nltk.WordNetLemmatizer()
the_stopwords = stopwords.words('english')
tfidf = TfidfVectorizer()
nlp = spacy.load('en_core_web_md')

# Extract
with open("../../data/raw/guide.txt") as file:
    guide = file.read()

# Transform
guide = guide.lower()
tokens = word_tokenize(guide)
text = nltk.Text(tokens)
words = [re.sub(r'[^A-Za-z_\s]', '', w) for w in text]
words = [wnl.lemmatize(w) for w in words if w.strip() != ''] # Will take work later to get the original text back without lemmatize.
sents = nltk.sent_tokenize(guide)
print(sents[-10:])

# Get dictionaries
tfidf.fit(words)
tfidf_weights_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
embeddings_dict = {}
for val in tfidf_weights_dict:
    embeddings_dict[val] = nlp(val).vector

# Averaged Vectors
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
sim_mat = np.zeros([len(sents), len(sents)])
for i in range(len(sents)):
    for j in range(len(sents)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sent_vectors[i].reshape(1,300), sent_vectors[j].reshape(1,300))[0,0]
print(sim_mat)

# Graph
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)



print(len(sent_vectors))
print(sent_vectors[-3:])
print(scores)
# words = [wnl.lemmatize(w) for w in words]
# words = [w for w in words if w not in the_stopwords and w != '']





print(guide[-400:])