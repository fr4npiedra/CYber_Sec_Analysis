import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import DataSet from CSV.

URL_1 = 'http://127.0.0.1:8000/api/cyberrecords'
df = pd.read_json(URL_1)

df = df.head(1000)

# 
# ***************************************************************************************
#      See Data Frame Information
# ***************************************************************************************
print("\n************************************************************************")
print('Variables')
print("************************************************************************\n")
print(df.head())
print("\n\n************************************************************************")
print('Data Types')
print("************************************************************************\n")
print(df.info())
print("\n\n************************************************************************")
print('Data Shape')
print("************************************************************************\n")
print(df.shape)
print("\n\n************************************************************************")
print('Statistics')
print("************************************************************************\n")
print(df.describe())
print("\n\n************************************************************************")
print('Null Values')
print("************************************************************************\n")
print(df.isnull().sum())
print("\n\n************************************************************************")
print('DataFrame Size')
print("************************************************************************\n")
print(len(df))
print("************************************************************************\n\n")


#***************************
#***************************
#  Plots from the Exported Data
#***************************
#***************************
#df = df#

############################
# Source Bytes vs Class 
# Detection of Anomalies.
# We can compare the difference in the Source Bytes Size, between the Normal and Suspicious packages.
############################
sns.set(style = 'whitegrid')
sns.stripplot(x="class", y="src_bytes",hue="protocol_type", data=df, jitter=0.2, marker="D", palette="Set2",edgecolor="gray", alpha=.75)
plt.savefig('./plots/01. SRC_BYTES-CLASS.pdf')
plt.show()


############################
# Source Bytes vs Class 
# Detection of Anomalies.
# We can compare the difference in the Source Bytes Size, between the Normal and Suspicious packages.
############################

ax = sns.boxplot(data=tips, x="count", y="protocol_type", whis=np.inf)
ax = sns.stripplot(data=tips, x="count", y="protocol_type", color=".3")
