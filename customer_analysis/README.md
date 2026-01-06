🧠 Customer Personality Analysis using Unsupervised Learning
📌 Project Overview

This project focuses on customer segmentation using unsupervised machine learning techniques.
The objective is to identify meaningful customer groups based on demographics, income, and purchasing behavior to support data-driven marketing strategies.

📂 Dataset

Source: Marketing Campaign Dataset

The dataset includes:

Customer demographics

Income information

Product category spending

Purchase behavior (web, catalog, store)

Campaign response data

⚙️ Technologies Used

Python

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

🔧 Feature Engineering

The following features were derived to enhance clustering quality:

Age = Current year − Year of birth

Total_Spending = Sum of all product category spending

Living_with = Partner / Alone (derived from marital status)

Children = Kidhome + Teenhome

Family_Size

Total_Purchases

Total_Campaign_Accepted

Outliers in Income, Total_Spending, and Age were removed using the IQR method.

🧪 Data Preprocessing

Numerical features were standardized

Categorical features were One-Hot Encoded

A ColumnTransformer pipeline was used to ensure correct preprocessing

This avoids ordinal bias and ensures compatibility with distance-based algorithms.

📉 Dimensionality Reduction

Principal Component Analysis (PCA) was applied with:

n_components = 3

Used to reduce dimensionality before clustering

Helps retain meaningful variance while removing noise

🔍 Clustering Approach

KMeans clustering was performed for multiple values of K.

🧮 Cluster Validation

Silhouette Score was used to evaluate cluster quality

Tested for K = 2 to K = 10

Final models selected based on interpretability and business relevance

🧩 Final Cluster Models
🔹 K = 2 (High-Level Segmentation)

Used for strategic overview of customers.

Cluster interpretation is data-driven, based on average income and spending:

High Income – High Spending

High Income – Low Spending

Low Income – High Spending

Low Income – Low Spending

This approach avoids dependency on arbitrary cluster labels.

🔹 K = 4 (Detailed Segmentation)

Used for actionable marketing insights.

Customers are segmented into:

High Income – High Spending

High Income – Low Spending

Low Income – High Spending

Low Income – Low Spending

This enables targeted marketing and optimized campaign strategies.

📊 Visualization

PCA scatter plots were used to visualize cluster separation

Clear segmentation observed for K = 4

📁 Output

The final output file:

customer_personality_analysis_final.csv


Includes:

Original customer data

Cluster labels for K = 2 and K = 4

Business-interpretable cluster descriptions

💡 Key Learnings

Importance of correct encoding for clustering

Role of PCA in distance-based algorithms

Mathematical validation of clustering using silhouette score

Translating technical clusters into business insights

🚀 Future Improvements

Experiment with DBSCAN and Hierarchical clustering

Automate cluster naming using statistical rules

Build interactive dashboards

Deploy as a customer segmentation service

👩‍💻 Author

Vaishnavi
Biomedical Engineering Graduate
Aspiring Data Analyst | Python Developer | ML Enthusiast
