# Experiment 3: Dimensionality Reduction Using PCA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# 1. Load Titanic Dataset
# ---------------------------------------------------------

df = sns.load_dataset("titanic")

print("========== DATASET INFORMATION ==========")
print("Dataset loaded successfully")
print("Shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

# ---------------------------------------------------------
# 2. Select Numerical Features
# ---------------------------------------------------------

features = ['age', 'fare', 'sibsp', 'parch', 'pclass']

X = df[features]

print("\n========== SELECTED FEATURES ==========")
print(X.head())

# ---------------------------------------------------------
# 3. Handle Missing Values
# ---------------------------------------------------------

print("\n========== MISSING VALUES BEFORE ==========")
print(X.isnull().sum())

# Remove rows containing missing values
X = X.dropna()

print("\n========== MISSING VALUES AFTER ==========")
print(X.isnull().sum())

print("\nShape after removing missing values:", X.shape)

# ---------------------------------------------------------
# 4. Standardize the Features
# ---------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\n========== STANDARDIZED DATA ==========")
print(pd.DataFrame(X_scaled, columns=features).head())

# ---------------------------------------------------------
# 5. Apply PCA - Two Components
# ---------------------------------------------------------

pca_2 = PCA(n_components=2)

X_pca_2 = pca_2.fit_transform(X_scaled)

print("\n========== PCA WITH 2 COMPONENTS ==========")

print("Principal Components:")
print(X_pca_2[:5])

print("\nExplained Variance Ratio:")
print(pca_2.explained_variance_ratio_)

print("\nTotal Explained Variance:",
      sum(pca_2.explained_variance_ratio_))

# ---------------------------------------------------------
# 6. Visualize First Two Principal Components
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca_2[:, 0],
    X_pca_2[:, 1],
    alpha=0.6
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Titanic Dataset")

plt.grid(True)
plt.show()

# ---------------------------------------------------------
# Exercise 1: Apply PCA using Three Components
# ---------------------------------------------------------

pca_3 = PCA(n_components=3)

X_pca_3 = pca_3.fit_transform(X_scaled)

print("\n========== PCA WITH 3 COMPONENTS ==========")

print("Shape before PCA:", X_scaled.shape)
print("Shape after PCA:", X_pca_3.shape)

print("\nExplained Variance Ratio:")
print(pca_3.explained_variance_ratio_)

print("\nTotal Explained Variance:",
      sum(pca_3.explained_variance_ratio_))

# ---------------------------------------------------------
# Exercise 2: Plot Explained Variance Ratio
# ---------------------------------------------------------

pca_all = PCA()

X_pca_all = pca_all.fit_transform(X_scaled)

explained_variance = pca_all.explained_variance_ratio_

print("\n========== EXPLAINED VARIANCE ==========")

for i, variance in enumerate(explained_variance):
    print(f"PC{i+1}: {variance:.4f}")

plt.figure(figsize=(8, 6))

plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance Ratio of Principal Components")

plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(axis="y")

plt.show()

# ---------------------------------------------------------
# Exercise 3: Compare Dataset Before and After PCA
# ---------------------------------------------------------

print("\n========== BEFORE AND AFTER PCA ==========")

print("Original number of features:", X.shape[1])
print("Features after PCA:", X_pca_2.shape[1])

print("\nOriginal Features:")
print(features)

print("\nPCA Features:")
print(["PC1", "PC2"])

# ---------------------------------------------------------
# PCA Component Loadings
# ---------------------------------------------------------

print("\n========== PCA COMPONENT LOADINGS ==========")

loadings = pd.DataFrame(
    pca_2.components_,
    columns=features,
    index=["PC1", "PC2"]
)

print(loadings)

# ---------------------------------------------------------
# Exercise 4: Interpret First Two Components
# ---------------------------------------------------------

print("\n========== INTERPRETATION ==========")

print("PC1 loadings:")
print(loadings.loc["PC1"])

print("\nPC2 loadings:")
print(loadings.loc["PC2"])

# ---------------------------------------------------------
# Exercise 5: Apply PCA to Another Dataset
# ---------------------------------------------------------

from sklearn.datasets import load_iris

iris = load_iris()

iris_X = iris.data

iris_scaler = StandardScaler()

iris_scaled = iris_scaler.fit_transform(iris_X)

iris_pca = PCA(n_components=2)

iris_pca_data = iris_pca.fit_transform(iris_scaled)

print("\n========== PCA ON IRIS DATASET ==========")

print("Original Iris shape:", iris_X.shape)
print("Shape after PCA:", iris_pca_data.shape)

print("\nExplained Variance Ratio:")
print(iris_pca.explained_variance_ratio_)

plt.figure(figsize=(8, 6))

plt.scatter(
    iris_pca_data[:, 0],
    iris_pca_data[:, 1],
    c=iris.target,
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of Iris Dataset")

plt.grid(True)
plt.show()