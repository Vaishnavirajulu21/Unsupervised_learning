# 🧠 Customer Personality Analysis using Unsupervised Learning 
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-green)
![Clustering](https://img.shields.io/badge/Algorithm-KMeans-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
## 📌 Project Overview

This project focuses on customer segmentation using unsupervised machine learning techniques.  
The objective is to identify meaningful customer groups based on demographics, income, and purchasing behavior to support data-driven marketing strategies.

---

## 📂 Dataset

**Source:** Marketing Campaign Dataset  

The dataset includes:
- Customer demographics  
- Income information  
- Product category spending  
- Purchase behavior (web, catalog, store)  
- Campaign response data  

---

## ⚙️ Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 🔧 Feature Engineering
- **Age** calculated from year of birth  
- **Total_Spending** as sum of all product category spending  
- **Living_with** derived from marital status  
- **Children** = Kidhome + Teenhome  
- **Family_Size**
- **Total_Purchases**
- **Total_Campaign_Accepted**

---

## 🧪 Data Preprocessing
- Handled missing values  
- Removed outliers using IQR method  
- Applied **One-Hot Encoding** for categorical features  
- Standardized numerical features using **StandardScaler**

---

## 📉 Dimensionality Reduction
- Applied **PCA (Principal Component Analysis)**
- Reduced features to 3 principal components
- Improved clustering efficiency and visualization

---

## 🔍 Clustering Technique
- Applied **KMeans clustering**
- Tested multiple values of **K (2–10)**
- Evaluated using **Silhouette Score**

---

## 📊 Cluster Interpretation

### 🔹 K = 2 (High-level Segmentation)
- Broad customer grouping
- Useful for strategic insights

### 🔹 K = 4 (Detailed Segmentation)
- High Income – High Spending  
- High Income – Low Spending  
- Low Income – High Spending  
- Low Income – Low Spending  

---

## 📈 Visualizations
- PCA scatter plots
- Correlation heatmaps
- Cluster distribution plots

---

## 📁 Output
- Final dataset with cluster labels
- Business-friendly cluster descriptions

---

## 👩‍💻 Author
**Vaishnavi**  
Biomedical Engineering Graduate  
Aspiring Data Analyst | Python Developer | ML Enthusiast  

⭐ If you find this project useful, feel free to star the repository!
