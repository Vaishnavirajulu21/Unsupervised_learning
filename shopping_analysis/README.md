# 🛍️ Customer Shopping Trends Analysis

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-green)
![Clustering](https://img.shields.io/badge/Clustering-KMeans-orange)
![Dimensionality Reduction](https://img.shields.io/badge/Dimensionality-PCA-purple)
![Anomaly Detection](https://img.shields.io/badge/Anomaly%20Detection-IsolationForest-red)
![Status](https://img.shields.io/badge/Project-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview
- **Label: Objective**  
  Analyze customer shopping behavior using unsupervised learning techniques

- **Label: Goal**  
  Identify meaningful customer segments and detect anomalous purchasing patterns

- **Label: Approach**  
  Feature engineering, clustering, dimensionality reduction, and anomaly detection

---

## 📂 Dataset Information
- **Label: Dataset Name**  
  Shopping Trends Dataset

- **Label: Total Records**  
  3,900 customers

- **Label: Data Type**  
  Retail transaction data

- **Label: Key Attributes**
  - Customer demographics
  - Purchase amount and frequency
  - Discounts and promotions
  - Payment and shipping methods
  - Review ratings

---

## 🛠️ Tools & Technologies
- **Label: Programming Language**
  - Python

- **Label: Data Analysis Libraries**
  - Pandas
  - NumPy

- **Label: Visualization Tools**
  - Matplotlib
  - Seaborn

- **Label: Machine Learning (Scikit-learn)**
  - KMeans
  - PCA
  - OneHotEncoder
  - StandardScaler
  - IsolationForest
  - Silhouette Score
  - Davies–Bouldin Index

---

## 🔍 Data Preprocessing
- **Label: Cleaning Steps**
  - Removed irrelevant identifiers (Customer ID)
  - Checked and handled unrealistic values

- **Label: Transformations**
  - Renamed columns for clarity
  - Encoded categorical features using One-Hot Encoding
  - Scaled numerical features using StandardScaler

---

## 🧠 Feature Engineering
- **Label: Spending**
  - Renamed from purchase amount

- **Label: Spending_Level**
  - Quartile-based segmentation

- **Label: High_Spender**
  - Binary indicator (1 = High spender)

- **Label: Age_Group**
  - Youth / Adult / Middle_Aged / Senior

- **Label: Discount_Used**
  - Binary conversion of discount usage

---

## 📉 Dimensionality Reduction (PCA)
- **Label: Technique**
  - Principal Component Analysis (PCA)

- **Label: Components Used**
  - PCA1, PCA2, PCA3

- **Label: Purpose**
  - Reduce dimensionality
  - Improve clustering performance
  - Enable 2D visualization

- **Label: Note**
  - PCA features are derived features, not original variables

---

## 🧩 Clustering Methodology
- **Label: Algorithm**
  - K-Means Clustering

- **Label: K Range Tested**
  - 2 to 10

- **Label: Final Selection**
  - K = 8

- **Label: Evaluation Metrics**
  - Silhouette Score
  - Davies–Bouldin Index
  - Cluster compactness and separation

---

## 👥 Customer Personas (K = 8)

- **Label: Price-Sensitive Budget Shoppers**
  - Description: Low spending, high discount usage

- **Label: Occasional Low-Engagement Buyers**
  - Description: Infrequent purchases with low transaction value

- **Label: Young Trend-Oriented Shoppers**
  - Description: Younger customers with moderate spending behavior

- **Label: Family-Oriented Value Buyers**
  - Description: Practical and consistent purchasing patterns

- **Label: Premium Loyal Customers**
  - Description: High spending with low discount dependency

- **Label: High-Value Premium Buyers**
  - Description: Top revenue contributors with frequent purchases

- **Label: Deal-Driven Shoppers**
  - Description: Highly promotion-sensitive customers

- **Label: Exploratory / Niche Customers**
  - Description: Mixed or unusual shopping behavior

---

## 🚨 Anomaly Detection
- **Label: Algorithm**
  - Isolation Forest

- **Label: Contamination Rate**
  - 5%

- **Label: Features Used**
  - Age
  - Spending
  - Review Rating
  - Purchase Frequency
  - Discount Usage

- **Label: Outcome**
  - Identified abnormal customer behavior
  - Visualized anomalies within clusters

---

## 📊 Visualizations
- **Label: Included Plots**
  - Correlation heatmap
  - Silhouette score vs number of clusters
  - PCA scatter plots with cluster labels
  - Cluster size distribution
  - Anomaly overlay plots

---

## 👩‍💻 Author
- **Vaishnavi**
- Aspiring Data Scientist | ML Enthusiast

⭐ If you find this project useful, feel free to star the repository!
