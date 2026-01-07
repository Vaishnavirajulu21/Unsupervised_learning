

# ## Import Needed Libraries

# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")


# ## Load the dataset

# In[69]:


df = pd.read_csv("marketing_campaign.csv", sep="\t")
print(df.shape)
df.head(50)


# ## Data cleaning

# In[70]:


df.info()


# In[71]:


df.isnull().sum()


# In[72]:


df = df.dropna() # removing a null value in a income

df.shape


# In[73]:


df.isnull().sum()


# In[74]:


df.isnull().sum().sum()


# In[75]:


df.columns


# In[76]:


df.dtypes


# ## Feature Engineering

# In[77]:


df['Dt_Customer'] = pd.to_datetime(
    df['Dt_Customer'],
    dayfirst=True,
    errors='coerce'
)

dates=[]

for i in df['Dt_Customer']:
    i=i.date()
    dates.append(i)
    
print("The newest customer's enrolment date in the records:",max(dates))
print("The oldest customer's enrolment date in the records:",min(dates))


# In[78]:


print("Total categories in the feature Marital_Status:\n", df["Marital_Status"].value_counts(), "\n")
print("Total categories in the feature Education:\n", df["Education"].value_counts())


# In[79]:


# to get the current age 
df["Age"] = datetime.now().year - df["Year_Birth"]

# total sepending on all the items
df['Total_Spending'] = (
    df['MntWines'] +
    df['MntFruits'] +
    df['MntMeatProducts'] +
    df['MntFishProducts'] +
    df['MntSweetProducts'] +
    df['MntGoldProds']
)

#to understanding the living situations
df['Living_with'] = df['Marital_Status'].replace({"Married":"Partner","Together":"Partner", 
"Absurd":"Alone", "Widow":"Alone", "YOLO":"Alone", "Divorced":"Alone", "Single":"Alone" })                                                 

df["Children"]=df["Kidhome"]+df["Teenhome"]

df["Family_Size"] = df['Living_with'].replace({"Partner": 2 ,"Alone":1}) + df["Children"]

df['is_parent'] = np.where(df.Children> 0, 1, 0)

#Segmenting education levels in three groups
df["Education"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})

#For clarity
df=df.rename(columns={"MntWines": "Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweets","MntGoldProds":"Gold"})



df = df.drop(columns=["Marital_Status", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID"])


# In[80]:


df['Total_Purchases'] = (
    df['NumWebPurchases'] +
    df['NumCatalogPurchases'] +
    df['NumStorePurchases']
)


# In[81]:


df['Total_Campaign_Accepted'] = (
    df['AcceptedCmp1'] +
    df['AcceptedCmp2'] +
    df['AcceptedCmp3'] +
    df['AcceptedCmp4'] +
    df['AcceptedCmp5'] +
    df['Response']
)


# In[82]:


df.head(10)


# In[83]:


# Outlier removal using IQR for Income, Total_Spending, Age (if present)
def remove_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]

for col in ["Income","Total_Spending","Age"]:
    if col in df.columns:
        before = df.shape[0]
        df = remove_outliers_iqr(df, col)
        after = df.shape[0]
        print(f"Removed {before-after} rows as outliers for {col}")

print("Shape after outlier removal:", df.shape)


# In[84]:


df.describe()


# In[85]:


from matplotlib import colors
from matplotlib.colors import ListedColormap
#To plot some selected features 
#Setting up colors prefrences
sns.set(rc={"axes.facecolor":"#FFF9ED","figure.facecolor":"#FFF9ED"})
pallet = ["#682F2F", "#9E726F", "#B9C0C9", "#9F8A78", "#F3AB60"]
cmap = colors.ListedColormap(["#682F2F", "#9E726F", "#B9C0C9", "#9F8A78", "#F3AB60"])
#Plotting following features
To_Plot = [ "Income", "Recency", "Age", "Total_Spending", "is_parent"]
print("Reletive Plot Of Some Selected Features: A Data Subset")
plt.figure()
sns.pairplot(df[To_Plot], hue= "is_parent",palette= (["#682F2F","#F3AB60"]))
#Taking hue 
plt.show()


# In[86]:


df = df[(df["Age"]<90)]
df = df[(df["Income"]<600000)]
print("The total number of data-points after removing the outliers are:", len(df))


# In[87]:


#correlation matrix

corrmat = df.select_dtypes(include='number').corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    corrmat,
    cmap='coolwarm',
    annot=True,         
    linewidths=0.5
)
plt.title("Correlation Matrix (Numeric Features)")
plt.show()


# In[88]:


# Features you want for clustering
features = [
    'Education', 'Income', 'Kidhome', 'Teenhome', 'Recency',
    'Wines', 'Fruits', 'Meat', 'Fish', 'Sweets', 'Gold',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'Age',
    'Total_Spending', 'Living_with', 'Children'
]


X = df[features]

# Separate numeric & categorical
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Categorical:", categorical_cols)
print("Numeric:", numeric_cols)


# In[89]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)


# In[90]:


X_processed = preprocessor.fit_transform(X)

print("Final feature matrix shape:", X_processed.shape)


# In[91]:


pca = PCA(n_components=3, random_state=42)
pca_data = pca.fit_transform(X_processed)

pca_df = pd.DataFrame(
    pca_data,
    columns=['PCA1', 'PCA2', 'PCA3'],
    index=df.index
)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Explained:", pca.explained_variance_ratio_.sum())


# In[94]:


# Test K from 2 to 10
sil_scores = {}
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pca_df)
    sil = silhouette_score(pca_df, labels)
    sil_scores[k] = sil
    print(f"K={k} → Silhouette Score = {sil}")

best_k = max(sil_scores, key=sil_scores.get)
print("---"*40)
print("Best K based on Silhouette:", best_k)


# In[95]:


# Run KMeans for K=2 and K=4 and compute silhouette scores
k2 = KMeans(n_clusters=2, random_state=42)
df['Cluster_K2'] = k2.fit_predict(pca_df)
sil_k2 = silhouette_score(pca_df, df['Cluster_K2'])

k4 = KMeans(n_clusters=4, random_state=42)
df['Cluster_K4'] = k4.fit_predict(pca_df)
sil_k4 = silhouette_score(pca_df, df['Cluster_K4'])

print("Silhouette K=2:", round(sil_k2,4))
print("Silhouette K=4:", round(sil_k4,4))


# In[97]:


plt.plot(list(sil_scores.keys()), list(sil_scores.values()))
plt.title("Silhouette Score by K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.show()


# In[98]:


# PCA 2D comparison plot

fig, axes = plt.subplots(1,2, figsize=(15,6))

sns.scatterplot(x=pca_df['PCA1'], y=pca_df['PCA2'], hue=df['Cluster_K2'], palette='tab10', ax=axes[0], alpha=0.7)
axes[0].set_title('PCA Clusters (K=2)')

sns.scatterplot(x=pca_df['PCA1'], y=pca_df['PCA2'], hue=df['Cluster_K4'], palette='tab10', ax=axes[1], alpha=0.7)
axes[1].set_title('PCA Clusters (K=4)')

plt.show()


# In[99]:


# PCA 3D comparison plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(16,7))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c=df['Cluster_K2'], cmap='tab10', s=40)
ax1.set_title('3D PCA Clusters (K=2)')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'], c=df['Cluster_K4'], cmap='tab10', s=40)
ax2.set_title('3D PCA Clusters (K=4)')

plt.show()


# In[100]:


from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

print('Elbow Method to determine the optimal number of clusters:')

fig, ax = plt.subplots(figsize=(8,6))

visualizer = KElbowVisualizer(KMeans(), k=10, ax=ax)
visualizer.fit(pca_df)
visualizer.show()




# In[101]:


# Income vs Total_Spending comparison
fig, axes = plt.subplots(1,2, figsize=(15,6))

if 'Income' in df.columns and 'Total_Spending' in df.columns:
    sns.scatterplot(data=df, x='Income', y='Total_Spending', hue='Cluster_K2', palette='tab10', ax=axes[0], alpha=0.7)
    axes[0].set_title('Income vs Spending — K=2')

    sns.scatterplot(data=df, x='Income', y='Total_Spending', hue='Cluster_K4', palette='tab10', ax=axes[1], alpha=0.7)
    axes[1].set_title('Income vs Spending — K=4')
    plt.show()
else:
    print('Income or Total_Spending not in dataframe; skipping Income vs Spending plots.')


# In[102]:


# Countplot for K = 2

plt.figure(figsize=(6,4))
sns.countplot(x=df["Cluster_K2"], palette='tab10')
plt.title("Distribution of Clusters (K = 2)")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()


# In[103]:


# Countplot for K = 4


plt.figure(figsize=(6,4))
sns.countplot(x=df["Cluster_K4"], palette='tab10')
plt.title("Distribution of Clusters (K = 4)")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()


# In[104]:


# Cluster summary tables for K=2 and K=4
metrics = [c for c in ['Income','Total_Spending','Age','Total_Purchases','Children'] if c in df.columns]
summary_k2 = df.groupby('Cluster_K2')[metrics].mean().round(2)
summary_k4 = df.groupby('Cluster_K4')[metrics].mean().round(2)

print('\n===== Cluster Summary (K=2) =====\n')
display(summary_k2)

print('\n===== Cluster Summary (K=4) =====\n')
display(summary_k4)


# | Cluster       | Cluster Type             | Meaning (Business Interpretation)                                                                                                           |
# | ------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
# | **Cluster 0** | **High-Value Customers** | Higher income, higher total spending, frequent purchases. Core revenue contributors. Ideal for loyalty programs and premium offers.         |
# | **Cluster 1** | **Low-Value Customers**  | Lower income and spending, fewer purchases. More price-sensitive and less engaged. Best targeted with discounts or re-engagement campaigns. |
# 

# In[107]:


avg_income = df["Income"].mean()
avg_spend  = df["Total_Spending"].mean()

df["K2_Description"] = ""

for cluster in sorted(df["Cluster_K2"].unique()):
    
    mask = df["Cluster_K2"] == cluster
    income_mean = df[mask]["Income"].mean()
    spend_mean  = df[mask]["Total_Spending"].mean()
    
    if income_mean > avg_income and spend_mean > avg_spend:
        df.loc[mask, "K2_Description"] = "High Income – High Spending"
        
    elif income_mean > avg_income and spend_mean < avg_spend:
        df.loc[mask, "K2_Description"] = "High Income – Low Spending"
        
    elif income_mean < avg_income and spend_mean > avg_spend:
        df.loc[mask, "K2_Description"] = "Low Income – High Spending"
        
    else:
        df.loc[mask, "K2_Description"] = "Low Income – Low Spending"


# | Cluster       | Cluster Type                    | Meaning                                                                                                                      |
# | ------------- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
# | **Cluster 0** | **High Income – Low Spending**  | Financially capable but cautious spenders. Potential upsell targets with personalized recommendations.                       |
# | **Cluster 1** | **Low Income – Low Spending**   | Least engaged group. Occasional buyers with high price sensitivity. Best targeted with discounts or minimal marketing spend. |
# | **Cluster 2** | **High Income – High Spending** | Premium customers with strong purchasing power and high engagement. Best suited for exclusive offers and VIP programs.       |
# | **Cluster 3** | **Low Income – High Spending**  | Highly engaged despite lower income. Likely promotion-responsive and brand-loyal. Good candidates for rewards programs.      |

# In[108]:


df["K4_Description"] = ""

# Step 1: cluster means
k4_summary = df.groupby("Cluster_K4")[["Income", "Total_Spending"]].mean()

# Step 2: sort clusters from richest → poorest
k4_sorted = k4_summary.sort_values(["Income", "Total_Spending"], ascending=False)

# Step 3: labeling based on sorted order
labels = [
    "High Income – High Spending",
    "High Income – Low Spending",
    "Low Income – High Spending",
    "Low Income – Low Spending"
]

# map descriptions based on sorted index
mapping = {cluster: labels[i] for i, cluster in enumerate(k4_sorted.index)}

# apply the mapping
df["K4_Description"] = df["Cluster_K4"].map(mapping)


# In[109]:


df.drop(columns=['Cluster'], inplace=True, errors='ignore')


# In[110]:


df.head(50)


# In[111]:


# Save the final dataframe with cluster assignments
out_file = "customer_personality_Analysis.csv"
df.to_csv(out_file, index=False)
print(f"Saved clustered data to {out_file}")


# In[ ]:




