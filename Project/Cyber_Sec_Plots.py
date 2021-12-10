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
plt.savefig('./plots/01. SRC_BYTES-CLASS.jpg')
plt.show()


############################
# Protocol Type
# Count= number of connections to the same destination host as the current connection in the past 2 seconds (quant.).
# Protocol Type. Protocol used in connection.
############################

ax = sns.boxplot(data=df, x="count", y="protocol_type", whis=np.inf)
ax = sns.stripplot(data=df, x="count", y="protocol_type", color=".3")
plt.savefig('./plots/02. COUNT-PROTOCOL_TYPE.pdf')
plt.savefig('./plots/02. COUNT-PROTOCOL_TYPE.jpg')
plt.show()

############################
# Protocol Type, Class, Duration
# Count= number of connections to the same destination host as the current connection in the past 2 seconds (quant.).
# Protocol Type. Protocol used in connection.
############################
g = sns.catplot(x="class", y="duration",
                hue="class", col="protocol_type",
                data=df, kind="strip",
                height=4, aspect=.7);
plt.savefig('./plots/03. DURATION-CLASS-PROTOCOL_TYPE.pdf')
plt.savefig('./plots/03. DURATION-CLASS-PROTOCOL_TYPE.jpg')
plt.show()

############################
# Service, Class, Duration
# Count= number of connections to the same destination host as the current connection in the past 2 seconds (quant.).
# Protocol Type. Protocol used in connection.
############################
g = sns.catplot(x="class", y="duration",
                hue="flag", col="protocol_type",
                data=df, kind="strip",
                height=4, aspect=.7);
plt.savefig('./plots/04. DURATION-CLASS-PRT_TYPE_Flag.pdf')
plt.savefig('./plots/04. DURATION-CLASS-PRT_TYPE_Flag.jpg')
plt.show()


############################
# 32.	Dst host count: count of the connections having the same destination IP address (quant.) 
# 33.	Dst host srv count: count of connections having the same port number (quant.) 
# 4.	Flag: status of the connection (e.g. REJ = connection rejected) (cat., 11 categories) 
# 2.	Protocol type: Protocol used in connection (cat., 3 categories) 
############################

sns.scatterplot(data=df, x="dst_host_count", y="dst_host_srv_count", hue="flag", style="protocol_type", palette="deep",hue_norm=(0, 10))
plt.savefig('./plots/05. DURATION_SRCBYTES_FLAG.pdf')
plt.savefig('./plots/05. DURATION_SRCBYTES_FLAG.jpg')
plt.show()


############################
# 32.	Fag
# 
############################
sns.boxplot(x='flag', y='dst_host_same_srv_rate', data=df)
plt.savefig('./plots/06. FLAG_DST_HOST_SAME_SRV.pdf')
plt.savefig('./plots/06. FLAG_DST_HOST_SAME_SRV.jpg')
plt.show()