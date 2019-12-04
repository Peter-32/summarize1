# Add this project to the path
import os; import sys; currDir = os.path.dirname(os.path.realpath("__file__"))
rootDir = os.path.abspath(os.path.join(currDir, '..')); sys.path.insert(1, rootDir)

# Warnings
import warnings
warnings.filterwarnings("ignore")

# Public modules
import re
import nltk
import spacy
from os import path
import networkx as nx
import numpy as np
import pyperclip as clip
from numpy.random import seed
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set seed
seed(32)

# Initialize Helper Objects
print("Longer text leads to a longer runtime")
print("Progress: 0%")
wnl = nltk.WordNetLemmatizer()
tfidf = TfidfVectorizer()
nlp = spacy.load('en_core_web_sm')

# Clean Sentences
doc = clip.paste()
doc = doc.lower()
sents = nltk.sent_tokenize(doc)
processed_sents = []
for sent in sents:
    words = word_tokenize(sent)
    words = [re.sub(r'[^A-Za-z_\s]', '', w) for w in words]
    words = [wnl.lemmatize(w) for w in words if w.strip() != '']
    processed_sent = " ".join(words)
    processed_sents.append(processed_sent)

# Create Dictionaries
print("Progress: 20%")
tfidf.fit(processed_sents)
tfidf_weights_dict, embeddings_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_)), {}
vocabulary = tfidf_weights_dict.keys()
for word in vocabulary:
    embeddings_dict[word] = nlp(word).vector

# Convert Sentences into Numbers
print("Progress: 40%")
sent_vectors = []
for sent in processed_sents:
    vector_sum, denominator = [0]*96, 0
    for word in sent.split(" "):
        try:
            vector_sum += embeddings_dict[word]*tfidf_weights_dict[word]
            denominator += tfidf_weights_dict[word]
        except:
            pass
    if denominator != 0:
        sent_vectors.append(vector_sum/denominator)
    else:
        sent_vectors.append(vector_sum)

# Sentence Similarity
print("Progress: 60%")
sent_vectors = np.array(sent_vectors)
sent_vectors = StandardScaler().fit_transform(sent_vectors)
sent_vectors = MinMaxScaler().fit_transform(sent_vectors)
distances = pdist(sent_vectors, metric='euclidean')
sentence_similarity_matrix = squareform(distances)

# Graph and PageRank
print("Progress: 80%")
graph = nx.from_numpy_array(sentence_similarity_matrix)
scores = nx.pagerank(graph)
clip.copy("\n\n".join([sents[score] for score in [x for (x,y) in sorted([(x,y) for (x,y) in scores.items()], key= lambda x: float(x[1]), reverse=True)]]))
print("Done - results are in the clipboard")
