# ***************************************************************************************
# ***************************************************************************************
#
# Coding Dojo 
#  - Casptone Project.
# 
# Topic: 
# - Cyber-Security
#      - Intrussion Analysis.
#
#
# Students:
# - Fracisco Piedra S.
# - Sharon Alvarado B.
#
# Year:
# - 2021
#
# Dataset Taken from:
# - http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
#
# ***************************************************************************************
# ***************************************************************************************

# ***************************************************************************************
# IMPORT NECESSARY LIBRARIES
# ***************************************************************************************
#
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
import warnings
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, accuracy_score, recall_score, precision_score
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import sys
from tqdm import tqdm
import time
import os
import pylab as py

# ***************************************************************************************
#		Ignore all warnings
# ***************************************************************************************
#
warnings.filterwarnings("ignore")

# ***************************************************************************************
#		Load Data Set
# ***************************************************************************************
#
r = requests.get('http://127.0.0.1:8000/api/cyberrecords')
x = r.json()
dataset = pd.DataFrame(x['teams'])

r = requests.get('http://127.0.0.1:8000/api/cybertests')
x = r.json()
dataset_temp = pd.DataFrame(x['teams'])

#dataset = pd.read_csv('Train_data.csv')
#dataset_temp = pd.read_csv('Train_data.csv')

# ***************************************************************************************
#      See Data Frame Information
# ***************************************************************************************
#