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
import requests 

# ***************************************************************************************
#   Ignore all warnings
# ***************************************************************************************
#
warnings.filterwarnings("ignore")

# ***************************************************************************************
#   Load Data Set
# ***************************************************************************************
#
URL_1 = 'http://127.0.0.1:8000/api/cyberrecords'
dataset = pd.read_json(URL_1)

URL_2 = 'http://127.0.0.1:8000/api/cyberrecords'
dataset_temp = pd.read_json(URL_2)


# ***************************************************************************************
#      See Data Frame Information
# ***************************************************************************************
print("\n\n\n************************************************************************")
print('Variables')
print("************************************************************************\n")
print(dataset.head())
print("\n\n\n************************************************************************")
print('Data Types')
print("************************************************************************\n")
print(dataset.info())
print("\n\n\n************************************************************************")
print('Data Shape')
print("************************************************************************\n")
print(dataset.shape)
print("\n\n\n************************************************************************")
print('Statistics')
print("************************************************************************\n")
print(dataset.describe())
print("\n\n\n************************************************************************")
print('Null Values')
print("************************************************************************\n")
print(dataset.isnull().sum())
print("\n\n\n************************************************************************")
print('DataFrame Size')
print("************************************************************************\n")
print(len(dataset))
print("************************************************************************\n\n\n")

# ***************************************************************************************
#      Pre-Processing
# ***************************************************************************************
X = dataset
y = dataset_temp

#******************************************
# Removing Duplicates and NAs
#X.drop_duplicates(inplace = True)
#X.dropna(inplace = True)

X = dataset.drop(['id','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','class','created_at','updated_at'], axis=1)
y = dataset['class']

dataset_temp = dataset_temp.drop(['class'], axis=1)

X = X.drop(['protocol_type','service','flag'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 33)

features_mean=list(X_test.columns[4:18])

print(features_mean)

plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(7,8))
axes = axes.ravel()
for idx,ax in enumerate(axes):
    ax.figure
    binwidth= (max(X_train[features_mean[idx]]) - min(X_train[features_mean[idx]]))/50
    #print(binwidth)
    ax.hist([X_test[features_mean[idx]],X_train[features_mean[idx]]], bins=np.arange(min(X_train[features_mean[idx]]), max(X_train[features_mean[idx]]) + binwidth, binwidth) , alpha=0.5,stacked=True, label=['Test','Train'],color=['#800D39','#EDDD4A'])
    ax.legend(loc='upper right')
    ax.set_title(features_mean[idx])
plt.tight_layout()
plt.show()

print(X_train.info())

#******************************************
# Treating Numeric Data Only
# 
X_train_numerics = X_train.select_dtypes(exclude = 'object')
X_train_cols = X_train.columns
X_train_numerics.columns = X_train_cols

#******************************************
# Treating Numeric Data Only
# 
X_test_numerics = X_test.select_dtypes(exclude = 'object')
X_test_cols = X_test.columns
X_test_numerics.columns = X_test_cols

#******************************************
# Scaling our Data
#
from sklearn.preprocessing import StandardScaler
from scipy import stats
ss = StandardScaler()
X_train_numeric = pd.DataFrame(ss.fit_transform(X_train_numerics))
X_train_numeric.set_index(X_train.index, inplace = True)
X_train_numeric.columns = X_train_cols

print(X_train_numeric.head())

ss = StandardScaler()
X_test_numeric = pd.DataFrame(ss.fit_transform(X_test_numerics))
X_test_numeric.set_index(X_test.index, inplace = True)
X_test_numeric.columns = X_test_cols

print(X_test_numeric.head())

#******************************************
# Removing Outliers
#
X_train_numeric = X_train_numeric[(np.abs(stats.zscore(X_train_numeric)) < 2.5).all(axis = 1)]
print(X_train_numeric.head())

X_test_numeric = X_test_numeric[(np.abs(stats.zscore(X_test_numeric)) < 2.5).all(axis = 1)]
print(X_test_numeric.head())

#******************************************
# Categorical Data
#
X_train_cat = dataset_temp.select_dtypes(include = 'object')

print(X_train_cat.head())

X_test_cat = dataset_temp.select_dtypes(include = 'object')

print(X_test_cat.head())

#******************************************
# Encoding Categorical Data
#
from sklearn import preprocessing
import category_encoders as ce

le = preprocessing.LabelEncoder()

#******************************************
# Create object for one-hot encoding
#
encoder_ptr_type = ce.OneHotEncoder(cols='protocol_type',handle_unknown='return_nan',return_df=True,use_cat_names=True)
encoder_serv = ce.OneHotEncoder(cols='service',handle_unknown='return_nan',return_df=True,use_cat_names=True)
encoder_flag = ce.OneHotEncoder(cols='flag',handle_unknown='return_nan',return_df=True,use_cat_names=True)

#Fit and transform Data
data_encoded = encoder_ptr_type.fit_transform(X_train_cat)
data_encoded = encoder_serv.fit_transform(data_encoded)
data_encoded = encoder_flag.fit_transform(data_encoded)

print(data_encoded)

#Fit and transform Data
data_encoded_tst = encoder_ptr_type.fit_transform(X_test_cat)
data_encoded_tst = encoder_serv.fit_transform(data_encoded_tst)
data_encoded_tst = encoder_flag.fit_transform(data_encoded_tst)

print(data_encoded_tst)

X_train_prep = pd.merge(data_encoded, X_train_numeric, left_index = True, right_index = True)
y_train = y_train.loc[X_train_prep.index]
encoder_class = ce.OneHotEncoder(cols='class',handle_unknown='return_nan',return_df=True,use_cat_names=True)
y_train_encoded = encoder_class.fit_transform(y_train)
y_train = y_train_encoded

print(X_train_prep.shape)
print(y_train.shape)

X_test_prep = pd.merge(data_encoded, X_test_numeric, left_index = True, right_index = True)
y_test = y_test.loc[X_test_prep.index]
encoder_class = ce.OneHotEncoder(cols='class',handle_unknown='return_nan',return_df=True,use_cat_names=True)
y_test_encoded = encoder_class.fit_transform(y_test)
y_test = y_test_encoded

print(X_test_prep.shape)
print(y_test.shape)

#******************************************
# Building the Linear Regression Model
#
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(X_train_prep, y_train)
y_hat_train = lr_model.predict(X_train_prep)
residuals = (y_train - y_hat_train)

#Seventh Step: Model Evaluation
#There are three primary metrics used to evaluate linear models. 
#These are: Mean absolute error (MAE), Mean squared error (MSE), or Root mean squared error (RMSE).

#Import metrics library
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
print(f"r^2: {r2_score(y_train, y_hat_train)}")

#Print result of MAE
#The Mean absolute error represents the average of the absolute difference between the actual and predicted values in the dataset. It measures the average of the residuals in the dataset.
print('MAE:',metrics.mean_absolute_error(y_train, y_hat_train))

#Print result of MSE
#Mean Squared Error represents the average of the squared difference between the original and predicted values in the data set. It measures the variance of the residuals.
print('MSE:',metrics.mean_squared_error(y_train, y_hat_train))

#Print result of RMSE
#Root Mean Squared Error is the square root of Mean Squared error. It measures the standard deviation of residuals.
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_hat_train)))

#The lower value of MAE, MSE, and RMSE implies higher accuracy of a regression model.
for x in range(0, 10):
  print('\n')

from statsmodels.graphics.gofplots import qqplot
qqplot(residuals, line = 'q')
py.show()

residuals = y_hat_train - y_train
plt.scatter(y_hat_train, residuals)
plt.show()

# # Use our model to make a prediction for Traffic Class
print('+*+*+*+*+*+*')
print(X_test.info())
#X_train, X_test, y_train, y_test 

y_pred = lr_model.predict(X_test_prep)
print('predicted response:', np.trunc(abs(y_pred)), sep='\n')
#print(y_pred.shape)

# for x in y_pred:
#   val = np.trunc(abs(np.around(x[0])))
#   print(val)

results_dataset = pd.DataFrame(columns = ['DURATION', 'SRC_BYTES', 'DST_BYTES','COUNT','SRV_COUNT','SAME_SRV_RATE','DIFF_SRV_RATE','SRV_DIFF_HOST_RATE','DST_HOST_COUNT','DST_HOST_SRV_COUNT','DST_HOST_SAME_SRV_RATE','DST_HOST_DIFF_SRV_RATE','DST_HOST_SAME_SRV_PORT_RATE','DST_HOST_SRV_DIFF_HOST_RATE','DST_HOST_SERROR_RATE','DST_HOST_SRV_SERROR_RATE','CLASS'])

i = 0
for x in y_pred:
  val = np.trunc(abs(np.around(x[0])))
  if (val == 1.0):
    print('Potential Thread',X_test['duration'].values[i])
    results_dataset = results_dataset.append({'DURATION' : X_test['duration'].values[i],
      'SRC_BYTES' : X_test['src_bytes'].values[i],
      'DST_BYTES' : X_test['dst_bytes'].values[i],
      'COUNT' : X_test['count'].values[i],
      'SRV_COUNT' : X_test['srv_count'].values[i],
      'SAME_SRV_RATE' : X_test['same_srv_rate'].values[i],
      'DIFF_SRV_RATE' : X_test['diff_srv_rate'].values[i],
      'SRV_DIFF_HOST_RATE' : X_test['srv_diff_host_rate'].values[i],
      'DST_HOST_COUNT' : X_test['dst_host_count'].values[i],
      'DST_HOST_SRV_COUNT' : X_test['dst_host_srv_count'].values[i],
      'DST_HOST_SAME_SRV_RATE' : X_test['dst_host_same_srv_rate'].values[i],
      'DST_HOST_DIFF_SRV_RATE' : X_test['dst_host_diff_srv_rate'].values[i],
      'DST_HOST_SAME_SRV_PORT_RATE' : X_test['dst_host_same_src_port_rate'].values[i],
      'DST_HOST_SRV_DIFF_HOST_RATE' : X_test['dst_host_srv_diff_host_rate'].values[i],
      'DST_HOST_SERROR_RATE' : X_test['dst_host_serror_rate'].values[i],
      'DST_HOST_SRV_SERROR_RATE' : X_test['dst_host_srv_serror_rate'].values[i],
      'CLASS' : 'Potential Thread'},ignore_index = True)
  else:
    print('Normal Package',i)
    results_dataset = results_dataset.append({'DURATION' : X_test['duration'].values[i],
      'SRC_BYTES' : X_test['src_bytes'].values[i],
      'DST_BYTES' : X_test['dst_bytes'].values[i],
      'COUNT' : X_test['count'].values[i],
      'SRV_COUNT' : X_test['srv_count'].values[i],
      'SAME_SRV_RATE' : X_test['same_srv_rate'].values[i],
      'DIFF_SRV_RATE' : X_test['diff_srv_rate'].values[i],
      'SRV_DIFF_HOST_RATE' : X_test['srv_diff_host_rate'].values[i],
      'DST_HOST_COUNT' : X_test['dst_host_count'].values[i],
      'DST_HOST_SRV_COUNT' : X_test['dst_host_srv_count'].values[i],
      'DST_HOST_SAME_SRV_RATE' : X_test['dst_host_same_srv_rate'].values[i],
      'DST_HOST_DIFF_SRV_RATE' : X_test['dst_host_diff_srv_rate'].values[i],
      'DST_HOST_SAME_SRV_PORT_RATE' : X_test['dst_host_same_src_port_rate'].values[i],
      'DST_HOST_SRV_DIFF_HOST_RATE' : X_test['dst_host_srv_diff_host_rate'].values[i],
      'DST_HOST_SERROR_RATE' : X_test['dst_host_serror_rate'].values[i],
      'DST_HOST_SRV_SERROR_RATE' : X_test['dst_host_srv_serror_rate'].values[i],
      'CLASS' : 'Normal Package'},ignore_index = True)
  i=i+1

print(results_dataset)

# le = preprocessing.LabelEncoder()
# le.fit(y)
#LabelEncoder()
#print(le.classes_)

print(y_pred.shape)

results_dataset.to_csv('Final_Results.csv')


#***************************
#***************************
#  Plots from the Exported Data
#***************************
#***************************

# 

