# ============================================
# Experiment 2
# Exploratory Data Analysis (EDA),
# Descriptive Statistics and Data Visualization
# Using Titanic Dataset
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# Load Titanic Dataset
# --------------------------------------------
# If you have titanic.csv in the same folder:
# df = pd.read_csv("titanic.csv")

# Otherwise use Seaborn
df = sns.load_dataset("titanic")

# ============================================
# DATASET OVERVIEW
# ============================================

print("========== FIRST FIVE RECORDS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFORMATION ==========")
df.info()

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# ============================================
# DESCRIPTIVE STATISTICS
# ============================================

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== FARE COLUMN STATISTICS ==========")
print(df["fare"].describe())

print("\nMean Fare:", df["fare"].mean())
print("Median Fare:", df["fare"].median())
print("Mode Fare:")
print(df["fare"].mode())
print("Standard Deviation:", df["fare"].std())

# ============================================
# BAR CHART
# ============================================

plt.figure(figsize=(6,5))
sns.countplot(x="class", data=df)

plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

# ============================================
# HISTOGRAM - AGE
# ============================================

plt.figure(figsize=(7,5))
sns.histplot(df["age"].dropna(), bins=20, kde=True)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# ============================================
# HISTOGRAM - FARE
# ============================================

plt.figure(figsize=(7,5))
sns.histplot(df["fare"], bins=20, kde=True)

plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# ============================================
# BOX PLOT - AGE
# ============================================

plt.figure(figsize=(6,5))
sns.boxplot(y=df["age"])

plt.title("Box Plot of Age")
plt.show()

# ============================================
# BOX PLOT - FARE
# ============================================

plt.figure(figsize=(6,5))
sns.boxplot(y=df["fare"])

plt.title("Box Plot of Fare")
plt.show()

# ============================================
# SCATTER PLOT
# ============================================

plt.figure(figsize=(7,5))
sns.scatterplot(data=df, x="age", y="fare", hue="survived")

plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# ============================================
# CORRELATION HEATMAP
# ============================================

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.show()

# ============================================
# OBSERVATIONS
# ============================================

print("\n========== OBSERVATIONS ==========")

print("1. Dataset loaded successfully.")
print("2. Missing values are present mainly in Age, Deck and Embarked.")
print("3. Most passengers travelled in Third Class.")
print("4. Age distribution is concentrated between 20 and 40 years.")
print("5. Fare contains several outliers.")
print("6. Correlation between Age and Fare is weak.")
print("7. Heatmap shows relationships among numerical variables.")