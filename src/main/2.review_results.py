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


def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Set seed - ensures that the datasets are split the same way if re-run
seed(32)

# Initialize
print("Initialize")

# Extract
print("Extract")
with open("../../data/raw/guide.txt") as file:
    guide = file.read()

# Transform
print("Transform")
sents = nltk.sent_tokenize(guide)


scores = load_obj("scores")

scores = [(x,y) for (x,y) in scores.items()]
scores = sorted(scores, key= lambda x: float(x[1]), reverse=True)
scores = [x for (x,y) in scores]
output = "\n\n".join([sents[score] for score in scores])[0:1000]

# Write
f = open("../../data/output/output.txt", "w")
f.write(output)
f.close()

# for i in sorted (scores.values()) :
#      print(i, end = " ")
