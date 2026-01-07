

# ## Loading needed Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings("ignore")


# ## Loading the dataset

# In[2]:


df = pd.read_csv('shopping_trends.csv')
df


# ## Data cleaning

# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.isnull().sum()


# In[6]:


df.isnull().sum().sum()


# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.describe()


# ## Feature Enginnering
# 

# In[10]:


total_rows = df.shape[0]
unique_customers = df['Customer ID'].nunique()

print("Total rows:", total_rows)
print("Unique Customer IDs:", unique_customers)


# In[11]:


df_data= df.copy()


# In[12]:


df_data.drop(columns=['Customer ID'], inplace=True)


# In[13]:


# RENAME COLUMN FOR CLARITY
# ===============================
df_data.rename(columns={'Purchase Amount (USD)': 'Spending'}, inplace=True)

# Remove unrealistic ages
if 'Age' in df_data.columns:
    df_data = df_data[df_data['Age'] <= 90]

# 1. Spending Level (Quartiles)
df_data['Spending_Level'] = pd.qcut(
    df_data['Spending'],
    q=4,
    labels=['Low', 'Medium', 'High', 'Very_High']
)

# 2. High Spender Flag
df_data['High_Spender'] = (df_data['Spending'] > df_data['Spending'].median()).astype(int)

# 3. Age Groups
df_data['Age_Group'] = pd.cut(
    df_data['Age'],
    bins=[0, 25, 35, 50, 65, 100],
    labels=['Youth', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior']
)

# 4. Discount Binary
if 'Discount Applied' in df_data.columns:
    df_data['Discount_Used'] = df_data['Discount Applied'].map({'Yes': 1, 'No': 0})
    df_data.drop(columns=['Discount Applied'], inplace=True)


# In[14]:


df_data


# In[15]:


df_data.dtypes


# In[16]:


numeric_features = df_data.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df_data.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline([
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)
)
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

X = preprocessor.fit_transform(df_data)
print("Final feature matrix shape:", X.shape)


# In[17]:


numeric_df = df_data.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix (Numeric Features Only)")
plt.show()


# In[18]:


pca = PCA(n_components=3, random_state=42)
X_pca = pd.DataFrame(
    pca.fit_transform(X),
    columns=['PCA1', 'PCA2', 'PCA3'],
    index=df_data.index
)


# In[19]:


k_values = range(2, 11)
silhouette_scores = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    silhouette_scores.append(score)
    print(f"K={k} → Silhouette Score = {score:.4f}")


# In[20]:


# Plot silhouette scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o')
plt.axvline(x=8, color='red', linestyle='--', label='Chosen K = 8')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by K")
plt.legend()
plt.grid(True)
plt.show()


# In[21]:


kmeans_8 = KMeans(n_clusters=8, random_state=42)
df_data['Cluster_K8'] = kmeans_8.fit_predict(X_pca)

print("\nFinal Silhouette Score (K=8):",
      silhouette_score(X_pca, df_data['Cluster_K8']))


# In[22]:


df_data['PCA1'] = X_pca['PCA1']
df_data['PCA2'] = X_pca['PCA2']

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster_K8',
    data=df_data,
    palette='tab10'
)

plt.title("Customer Segmentation using KMeans (K = 8)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()


# In[23]:


from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X_pca, kmeans_8.labels_)
print("Davies-Bouldin Index:", db_score)


# In[24]:


import numpy as np

X = X_pca.values
labels = kmeans_8.labels_
centers = kmeans_8.cluster_centers_

# Distance of each point to its centroid
intra_distances = []
for i in range(len(X)):
    center = centers[labels[i]]
    dist = np.linalg.norm(X[i] - center)
    intra_distances.append(dist)

avg_intra = np.mean(intra_distances)
print("Average intra-cluster distance:", avg_intra)


# In[25]:


from itertools import combinations

inter_distances = []
for c1, c2 in combinations(centers, 2):
    dist = np.linalg.norm(c1 - c2)
    inter_distances.append(dist)

avg_inter = np.mean(inter_distances)
print("Average inter-cluster distance:", avg_inter)


# In[26]:


from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score

X_sample, labels_sample = resample(
    X_pca,
    kmeans_8.labels_,
    n_samples=int(0.8 * len(X_pca)),
    random_state=42
)

kmeans_test = KMeans(n_clusters=8, random_state=42)
new_labels = kmeans_test.fit_predict(X_sample)

ari = adjusted_rand_score(labels_sample[:len(new_labels)], new_labels)
print("ARI Stability Score:", ari)


# In[27]:


threshold = np.percentile(intra_distances, 75)
well_clustered = np.mean(np.array(intra_distances) < threshold)
print("Well-clustered percentage:", well_clustered * 100)


# In[28]:


cluster_profile = df_data.groupby('Cluster_K8').agg({
    'Spending': 'mean',
    'Age': 'mean',
    'High_Spender': 'mean',
    'Discount_Used': 'mean'
}).round(2)

print("\nCluster Profile:")
print(cluster_profile)


# In[29]:


df_data


# In[30]:


plt.figure(figsize=(6,4))
sns.countplot(x=df_data["Cluster_K8"], palette='tab10')
plt.title("Distribution of Clusters (K = 8)")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()


# | Cluster Type                     | Meaning                   |
# | -------------------------------- | ------------------------- |
# | Price-Sensitive Budget Shoppers  | Low spend, high discounts |
# | Occasional Low-Engagement Buyers | Infrequent, low value     |
# | Young Trend-Oriented Shoppers    | Younger, moderate spend   |
# | Family-Oriented Value Buyers     | Practical, consistent     |
# | Premium Loyal Customers          | High spend, low discount  |
# | High-Value Premium Buyers        | Top revenue customers     |
# | Deal-Driven Shoppers             | Promotion sensitive       |
# | Exploratory / Niche Customers    | Unusual combinations      |
# 

# In[31]:


def assign_persona(row):

    spending = row['Spending']
    discount = row['Discount_Used']
    age = row['Age']

    if spending < 40 and discount > 0.6:
        return 'Price-Sensitive Budget Shoppers'

    elif spending < 40:
        return 'Occasional Low-Engagement Buyers'

    elif 40 <= spending < 60 and age < 35:
        return 'Young Trend-Oriented Shoppers'

    elif 40 <= spending < 60:
        return 'Family-Oriented Value Buyers'

    elif 60 <= spending < 80 and discount < 0.3:
        return 'Premium Loyal Customers'

    elif spending >= 80:
        return 'High-Value Premium Buyers'

    elif discount > 0.5:
        return 'Deal-Driven Shoppers'

    else:
        return 'Exploratory / Niche Customers'


# In[32]:


df_data['Customer_Persona'] = df_data.apply(assign_persona, axis=1)


# In[33]:


df_data


# In[36]:


# PCA for visualization
pca = PCA(n_components=3, random_state=42)
X_pca  = pd.DataFrame(pca.fit_transform(X), columns=['PCA1','PCA2','PCA3'], index=df.index)
print("PCA explained variance ratios:", pca.explained_variance_ratio_)


# In[37]:


print(df_data.columns.tolist())


# In[38]:


print(type(X_pca))


# In[39]:


print(df_data.columns)


# ## Anomaly Detection

# In[40]:


df_data.dtypes


# In[41]:


df_data['Frequency of Purchases'].value_counts(dropna=False)


# In[42]:


df_data['Frequency of Purchases'] = df_data['Frequency of Purchases'].astype(str)


# In[43]:


df_data['Frequency of Purchases']


# In[44]:


print(df_data['Frequency of Purchases'].unique())


# In[45]:


freq_map = {
    'Fortnightly' : 1,
    'Weekly' : 2 ,
    'Annually' : 3,
    'Quarterly' : 4,
    'Bi-Weekly' : 5, 
    'Monthly' : 6,
    'Every 3 Months' : 7
}

df_data['Frequency of Purchases'] = (
    df_data['Frequency of Purchases']
    .str.strip()
    .map(freq_map)
)


# In[46]:


print(df_data[['Age', 'Spending', 'Review Rating',
               'Frequency of Purchases', 'Discount_Used',
               'High_Spender']].dtypes)


# In[47]:


print(df_data.columns.tolist())


# In[48]:


df_data.isna().sum()


# In[49]:


cols_to_check = ['Age', 'Spending', 'Review Rating', 
                 'Frequency of Purchases', 'Discount_Used', 'High_Spender']

existing_cols = [col for col in cols_to_check if col in df_data.columns]

print(df_data[existing_cols].dtypes)


# In[50]:


df_data


# In[51]:


anomaly_features = ['Age', 'Frequency of Purchases', 'Review Rating', 'Spending', 'Discount_Used', 'High_Spender']


# In[52]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_anomaly = scaler.fit_transform(df_data[anomaly_features])


# In[53]:


from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,   # 5% anomalies
    random_state=42
)

df_data['Anomaly_Flag'] = iso_forest.fit_predict(X_anomaly)


# In[54]:


df_data.head(60)


# In[55]:


df_data['Anomaly'] = df_data['Anomaly_Flag'].map({
    1: 'Normal',
    -1: 'Anomaly'
})


# In[56]:


print(df_data['Anomaly'].value_counts())


# In[57]:


anomaly_cluster_summary = (
    df_data
    .groupby(['Cluster_K8', 'Anomaly'])
    .size()
    .unstack(fill_value=0)
)

print(anomaly_cluster_summary)


# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
sns.scatterplot(
    x='PCA1',
    y='PCA2',
    hue='Cluster_K8',
    style='Anomaly',
    data=df_data,
    palette='tab10'
)

plt.title("Customer Clusters with Anomaly Detection (K = 8)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster / Anomaly")
plt.show()


# In[59]:


cluster_anomaly_rate = (
    df_data.groupby('Cluster_K8')['Anomaly_Flag']
    .apply(lambda x: (x == -1).mean())
)

print(cluster_anomaly_rate)


# In[ ]:
out_file = "shopping_Analysis.csv"
df_data.to_csv(out_file, index=False)
print(f"Saved clustered data to {out_file}")




