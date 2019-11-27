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
from numpy import zeros
import pyperclip as clip
from numpy.random import seed
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Set seed
seed(32)

# Initialize Helper Objects
print("Initialize Helper Objects")
wnl = nltk.WordNetLemmatizer()
tfidf = TfidfVectorizer()
nlp = spacy.load('en_core_web_md')

# Extract
print("Extract")
doc = clip.paste()

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

# Similarity Matrix (copied this from an article)
print("Similarity matrix")
sim_mat = zeros([len(sents), len(sents)])
for i in range(len(sents)):
    for j in range(len(sents)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sent_vectors[i].reshape(1,300), sent_vectors[j].reshape(1,300))[0,0]
print(sim_mat)

# Graph (copied this from an article)
print("GraphX")
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

# Sort scores
scores = [(x,y) for (x,y) in scores.items()]
scores = sorted(scores, key= lambda x: float(x[1]), reverse=True)
scores = [x for (x,y) in scores]
output = "\n\n".join([sents[score] for score in scores])[0:500]

# Save results to clipboard
clip.copy(output)
print("Done - results are in the clipboard")
