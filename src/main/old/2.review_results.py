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

# Extract
print("Extract")
with open("../../data/raw/bitcoin.txt") as file:
    doc = file.read()

# Transform
print("Transform")
sents = nltk.sent_tokenize(doc)


scores = load_obj("scores")

scores = [(x,y) for (x,y) in scores.items()]
scores = sorted(scores, key= lambda x: float(x[1]), reverse=True)
scores = [x for (x,y) in scores]
output = "\n\n".join([sents[score] for score in scores])[0:10000]

# Write
f = open("../../data/output/output2.txt", "w")
f.write(output)
f.close()

# for i in sorted (scores.values()) :
#      print(i, end = " ")
