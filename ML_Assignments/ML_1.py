# ============================================================
# HOUSE PRICE PREDICTION
# Kaggle - House Prices: Advanced Regression Techniques
# EDA + Data Preprocessing + Linear Regression
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd                  # Used for loading and manipulating data
import numpy as np                   # Used for numerical calculations
import matplotlib.pyplot as plt      # Used for creating plots
import seaborn as sns                # Used for statistical visualizations

from scipy import stats              # Used for statistical tests such as t-test

# train_test_split divides data into training and testing sets
from sklearn.model_selection import train_test_split

# StandardScaler standardizes features to mean 0 and standard deviation 1
from sklearn.preprocessing import StandardScaler

# LinearRegression creates the linear regression model
from sklearn.linear_model import LinearRegression

# Metrics used to evaluate regression performance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")    # Hides warning messages

sns.set_theme(style="whitegrid")     # Gives plots a clean white-grid style


# ============================================================
# PART A: DATA LOADING & BASIC INSPECTION
# ============================================================

# Read the Kaggle CSV file into a Pandas DataFrame
df = pd.read_csv("train.csv")


# Display the first 10 rows to understand the dataset
print("=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)
print(df.head(10))


# df.shape returns (number of rows, number of columns)
print("\nDataset Shape:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# Display the datatype of every column
# Example: int64, float64, object
print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)
print(df.dtypes)


# Select all numerical columns
# np.number includes int and float columns
numerical_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

# Select all categorical/text columns
# "object" generally represents text data in Pandas
categorical_cols = df.select_dtypes(
    include="object"
).columns.tolist()

print("\nNumber of numerical columns:", len(numerical_cols))
print("Number of categorical columns:", len(categorical_cols))

print("\nNumerical Columns:", numerical_cols)
print("\nCategorical Columns:", categorical_cols)


# df.info() gives:
# - column names
# - number of non-null values
# - datatypes
# - memory usage
print("\n" + "=" * 70)
print("DATA TYPE INFORMATION")
print("=" * 70)
df.info()


# ============================================================
# PART B: UNIVARIATE ANALYSIS
# ============================================================

# Univariate analysis means studying ONE variable at a time

# Five important numerical features required by the assignment
key_features = [
    "LotArea",       # Size of the property lot
    "GrLivArea",     # Above-ground living area
    "SalePrice",     # Target variable: house selling price
    "YearBuilt",     # Year the house was built
    "OverallQual"    # Overall material/finish quality
]


# ------------------------------------------------------------
# HISTOGRAMS
# ------------------------------------------------------------

# Create a large figure for all five histograms
plt.figure(figsize=(15, 10))

# Loop through every selected feature
for i, f in enumerate(key_features, 1):

    # Create a 2 x 3 grid of plots
    plt.subplot(2, 3, i)

    # Histogram shows the distribution of numerical values
    # kde=True adds a smooth density curve
    # bins=30 divides the values into 30 intervals
    sns.histplot(
        df[f],
        kde=True,
        bins=30
    )

    plt.title(f"Distribution of {f}")
    plt.xlabel(f)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# SKEWNESS
# ------------------------------------------------------------

# Skewness measures the asymmetry of a distribution
#
# Skewness ≈ 0  -> approximately symmetric
# Positive      -> right-skewed
# Negative      -> left-skewed

print("\n" + "=" * 70)
print("SKEWNESS OF KEY FEATURES")
print("=" * 70)

for f in key_features:

    # Calculate skewness of the feature
    s = df[f].skew()

    print(f"{f}: {s:.3f}")

    # These are practical rules of thumb for interpreting skewness
    if abs(s) < 0.5:
        print("  -> Approximately symmetric / normal")
    elif abs(s) < 1:
        print("  -> Moderately skewed")
    else:
        print("  -> Highly skewed")


# ------------------------------------------------------------
# CATEGORICAL FEATURES
# ------------------------------------------------------------

# Count plots show how frequently each category occurs
# We analyze three categorical features
for f in ["Neighborhood", "HouseStyle", "BldgType"]:

    plt.figure(figsize=(12, 5))

    # value_counts() orders categories by frequency
    # countplot creates a bar chart
    sns.countplot(
        data=df,
        x=f,
        order=df[f].value_counts().index
    )

    plt.title(f"Distribution of {f}")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# TOP 10 MOST FREQUENT CATEGORIES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 MOST FREQUENT CATEGORIES")
print("=" * 70)

# Empty list to store:
# feature name, category name and frequency
category_counts = []

# Go through every categorical column
for col in categorical_cols:

    # value_counts() counts each category
    for category, count in df[col].value_counts().items():

        category_counts.append([
            col,
            category,
            count
        ])


# Convert the list into a DataFrame
top10 = pd.DataFrame(
    category_counts,
    columns=["Feature", "Category", "Count"]
)

# Sort by frequency from highest to lowest
# Then select the first 10
top10 = top10.sort_values(
    "Count",
    ascending=False
).head(10)

print(top10)


# ------------------------------------------------------------
# CATEGORY DOMINANCE / IMBALANCE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORY DOMINANCE")
print("=" * 70)

for col in categorical_cols:

    # normalize=True gives proportions instead of counts
    counts = df[col].value_counts(normalize=True)

    # First category is the most frequent category
    most_common = counts.index[0]

    # Convert proportion into percentage
    percentage = counts.iloc[0] * 100

    print(
        f"{col}: {most_common} = {percentage:.2f}%"
    )


# ============================================================
# PART C: BIVARIATE ANALYSIS
# ============================================================

# Bivariate analysis means studying the relationship
# between TWO variables


# ------------------------------------------------------------
# GrLivArea VS SalePrice
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

# Scatter plot:
# X-axis = living area
# Y-axis = house price
# Each point represents one house
sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("GrLivArea vs SalePrice")
plt.xlabel("Above Ground Living Area")
plt.ylabel("Sale Price")
plt.show()


# A positive trend is generally expected:
# larger living area -> higher SalePrice


# ------------------------------------------------------------
# SALEPRICE VS OVERALLQUAL
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

# Boxplot compares SalePrice distributions
# for different OverallQual values
sns.boxplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("SalePrice Across Overall Quality")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")
plt.show()


# Generally, higher OverallQual corresponds
# to higher house prices.


# ------------------------------------------------------------
# CORRELATION
# ------------------------------------------------------------

# Keep only numerical columns because
# Pearson correlation requires numerical values
numeric_df = df.select_dtypes(include=np.number)

# Calculate correlation of every numerical feature with SalePrice
correlations = (
    numeric_df
    .corr()["SalePrice"]
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("CORRELATION WITH SALEPRICE")
print("=" * 70)
print(correlations)


# .abs() considers both positive and negative correlations
# .head(11) includes SalePrice itself + 10 other features
top_corr = (
    correlations
    .abs()
    .sort_values(ascending=False)
    .head(11)
    .index
)


# ------------------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(12, 9))

# Heatmap represents correlation using colors
# annot=True displays the numerical correlation values
# fmt=".2f" displays values up to 2 decimal places
sns.heatmap(
    df[top_corr].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Top 10 Features Correlated with SalePrice")
plt.show()


# ------------------------------------------------------------
# COMPARE TWO NEIGHBORHOODS
# ------------------------------------------------------------

# Select two common neighborhoods for comparison
neighborhoods = ["NAmes", "CollgCr"]

# Keep only houses from these two neighborhoods
data = df[
    df["Neighborhood"].isin(neighborhoods)
]

print("\n" + "=" * 70)
print("AVERAGE SALE PRICE BY NEIGHBORHOOD")
print("=" * 70)

# groupby() creates one group for each neighborhood
# mean() calculates the average SalePrice
print(
    data
    .groupby("Neighborhood")["SalePrice"]
    .mean()
)


# ------------------------------------------------------------
# T-TEST BETWEEN NEIGHBORHOODS
# ------------------------------------------------------------

# Extract SalePrice for each neighborhood
g1 = df[
    df["Neighborhood"] == neighborhoods[0]
]["SalePrice"]

g2 = df[
    df["Neighborhood"] == neighborhoods[1]
]["SalePrice"]


# Independent two-sample t-test
# equal_var=False performs Welch's t-test
t, p = stats.ttest_ind(
    g1,
    g2,
    equal_var=False
)

print("\nT-test:")
print("t-statistic:", t)
print("p-value:", p)


# Hypothesis test:
# H0: The two population means are equal
# H1: The two population means are different
#
# If p < 0.05 -> statistically significant difference
# If p >= 0.05 -> no statistically significant evidence

if p < 0.05:
    print("Significant difference in average SalePrice.")
else:
    print("No significant difference in average SalePrice.")


# ------------------------------------------------------------
# NEWER VS OLDER HOMES
# ------------------------------------------------------------

# np.where works like an if-else condition
#
# If YearBuilt >= 2000:
#     "Newer Homes"
# Otherwise:
#     "Older Homes"

df["HomeAgeGroup"] = np.where(
    df["YearBuilt"] >= 2000,
    "Newer Homes",
    "Older Homes"
)


# ------------------------------------------------------------
# BOXPLOT: NEW VS OLD
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="HomeAgeGroup",
    y="SalePrice"
)

plt.title("SalePrice: Newer vs Older Homes")
plt.xlabel("Home Age Group")
plt.ylabel("Sale Price")

plt.show()


# ------------------------------------------------------------
# T-TEST: NEW VS OLD
# ------------------------------------------------------------

# Create two SalePrice groups
new = df[
    df["YearBuilt"] >= 2000
]["SalePrice"]

old = df[
    df["YearBuilt"] < 2000
]["SalePrice"]


# Test whether their average SalePrice differs significantly
t, p = stats.ttest_ind(
    new,
    old,
    equal_var=False
)

print("\n" + "=" * 70)
print("T-TEST: NEWER VS OLDER HOMES")
print("=" * 70)

print("Newer homes average:", new.mean())
print("Older homes average:", old.mean())
print("t-statistic:", t)
print("p-value:", p)

if p < 0.05:
    print("Result: Significant difference in mean SalePrice.")
else:
    print("Result: No significant difference in mean SalePrice.")


# ============================================================
# PART D: MISSING VALUE TREATMENT
# ============================================================

# ------------------------------------------------------------
# FIND MISSING VALUES
# ------------------------------------------------------------

# isnull() returns True for missing values
# sum() counts the missing values
# sort_values() sorts from highest to lowest
missing = (
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

# Display only columns that actually contain missing values
print(
    missing[
        missing > 0
    ]
)


# Display the five columns with the most missing values
print("\nTop 5 columns with most missing values:")
print(missing.head(5))


# ------------------------------------------------------------
# LOTFRONTAGE: NEIGHBORHOOD-WISE MEDIAN IMPUTATION
# ------------------------------------------------------------

# LotFrontage is related to the neighborhood.
# Therefore, missing values are filled using
# the median LotFrontage of that neighborhood.

df["LotFrontage"] = (
    df.groupby("Neighborhood")["LotFrontage"]
    .transform(
        lambda x: x.fillna(x.median())
    )
)


# If any values are still missing,
# use the overall LotFrontage median
df["LotFrontage"] = df["LotFrontage"].fillna(
    df["LotFrontage"].median()
)


# ------------------------------------------------------------
# NUMERICAL COLUMNS: MEDIAN IMPUTATION
# ------------------------------------------------------------

# For every numerical column:
# If missing values exist, replace them with the median.

for col in df.select_dtypes(include=np.number):

    if df[col].isnull().any():

        df[col] = df[col].fillna(
            df[col].median()
        )


# ------------------------------------------------------------
# CATEGORICAL COLUMNS: "Missing"
# ------------------------------------------------------------

# For categorical columns, replace missing values
# with the category "Missing".

for col in df.select_dtypes(include="object"):

    if df[col].isnull().any():

        df[col] = df[col].fillna("Missing")


# ------------------------------------------------------------
# VERIFY MISSING VALUES
# ------------------------------------------------------------

# First sum() counts missing values column-wise.
# Second sum() gives the total number of missing values.
remaining_missing = df.isnull().sum().sum()

print("\n" + "=" * 70)
print("MISSING VALUES AFTER IMPUTATION")
print("=" * 70)

print("Total missing values:", remaining_missing)

# Ideally the output should be 0.


# ============================================================
# PART E: OUTLIER DETECTION & TREATMENT
# ============================================================


# ------------------------------------------------------------
# IQR OUTLIER FUNCTION
# ------------------------------------------------------------

def iqr_outliers(data, col):

    # Q1 = 25th percentile
    # Q3 = 75th percentile
    q1, q3 = data[col].quantile([0.25, 0.75])

    # IQR = spread of the middle 50% of the data
    iqr = q3 - q1

    # Standard IQR outlier boundaries
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr

    # Select observations outside the boundaries
    outliers = data[
        (data[col] < low) |
        (data[col] > high)
    ]

    # Return all useful information
    return outliers, low, high


# ------------------------------------------------------------
# DETECT OUTLIERS IN LOTAREA AND GRLIVAREA
# ------------------------------------------------------------

for col in ["LotArea", "GrLivArea"]:

    # Run the IQR function
    out, low, high = iqr_outliers(df, col)

    # Calculate Q1 and Q3 for displaying
    q1, q3 = df[col].quantile([0.25, 0.75])

    print("\n" + "=" * 70)
    print(col.upper(), "OUTLIERS")
    print("=" * 70)

    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", q3 - q1)
    print("Lower bound:", low)
    print("Upper bound:", high)

    # Number of observations outside the IQR limits
    print("Number of outliers:", len(out))


    # --------------------------------------------------------
    # CAP OUTLIERS
    # --------------------------------------------------------

    # clip() limits values to the specified boundaries.
    #
    # Values below low become low.
    # Values above high become high.
    #
    # We cap instead of deleting rows so that
    # valid houses are not completely removed.
    df[col] = df[col].clip(
        low,
        high
    )


# ------------------------------------------------------------
# RE-PLOT AFTER OUTLIER CAPPING
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("GrLivArea vs SalePrice After Outlier Capping")
plt.xlabel("Above Ground Living Area")
plt.ylabel("Sale Price")

plt.show()


# ============================================================
# PART F: FEATURE ENCODING & SCALING
# ============================================================


# ------------------------------------------------------------
# SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

# X contains independent variables/features
X = df.drop(
    columns=["SalePrice"]
)

# y contains the dependent variable/target
y = df["SalePrice"]


# ------------------------------------------------------------
# REMOVE ID
# ------------------------------------------------------------

# Id is only an identifier.
# It does not represent a useful house characteristic.
if "Id" in X:
    X = X.drop(columns=["Id"])


# ------------------------------------------------------------
# ONE-HOT ENCODING
# ------------------------------------------------------------

# Find categorical columns
cat = X.select_dtypes(
    include="object"
).columns


# Convert categorical variables into 0/1 dummy variables
#
# drop_first=True removes one category from each variable
# to avoid the dummy variable trap / perfect multicollinearity.
X_encoded = pd.get_dummies(
    X,
    columns=cat,
    drop_first=True
)


# Convert all values to float
# This makes the entire dataset numerical.
X_encoded = X_encoded.astype(float)


# ------------------------------------------------------------
# STANDARDIZATION
# ------------------------------------------------------------

# StandardScaler uses:
#
# z = (x - mean) / standard deviation
#
# After scaling:
# mean ≈ 0
# standard deviation ≈ 1

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_encoded)


# Convert the NumPy array back into a DataFrame
# so that column names are preserved.
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X_encoded.columns
)


print("\n" + "=" * 70)
print("PROCESSED DATASET")
print("=" * 70)

print("Original feature count:", X.shape[1])
print("Encoded feature count:", X_encoded.shape[1])
print("Scaled dataset shape:", X_scaled.shape)


# ------------------------------------------------------------
# SAVE PROCESSED DATASET
# ------------------------------------------------------------

# Copy scaled features
processed_df = X_scaled.copy()

# Add the original SalePrice target back
processed_df["SalePrice"] = y.values

# Save as CSV
# index=False prevents Pandas from adding an extra index column.
processed_df.to_csv(
    "house_prices_clean.csv",
    index=False
)

print("\nSaved as: house_prices_clean.csv")


# ============================================================
# PART G: LINEAR REGRESSION
# ============================================================


# Select 10 important numerical features for regression
features = [
    "OverallQual",     # Overall quality
    "GrLivArea",       # Above-ground living area
    "GarageCars",      # Garage capacity
    "GarageArea",      # Garage size
    "TotalBsmtSF",     # Total basement area
    "1stFlrSF",        # First floor area
    "FullBath",        # Number of full bathrooms
    "TotRmsAbvGrd",    # Total rooms above ground
    "YearBuilt",       # Construction year
    "YearRemodAdd"     # Remodeling year
]


# X = selected independent variables
# y = target variable
X = df[features]
y = df["SalePrice"]


# ------------------------------------------------------------
# TRAIN-TEST SPLIT
# ------------------------------------------------------------

# 80% of data is used for training
# 20% is used for testing
#
# random_state=42 ensures the same split every time.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ------------------------------------------------------------
# STANDARDIZATION
# ------------------------------------------------------------

scaler = StandardScaler()

# IMPORTANT:
# fit_transform() is used only on training data.
# The scaler learns the training mean and standard deviation.
X_train_scaled = scaler.fit_transform(X_train)

# Only transform test data.
# We do NOT fit again because that would cause data leakage.
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# CREATE LINEAR REGRESSION MODEL
# ------------------------------------------------------------

model = LinearRegression()


# Train the model
#
# Linear Regression learns coefficients:
#
# SalePrice =
# β0 + β1X1 + β2X2 + ... + β10X10
#
# The coefficients are chosen to minimize
# the sum of squared prediction errors.
model.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

# Use the trained model to predict prices
# for houses in the test set.
y_pred = model.predict(
    X_test_scaled
)


# ============================================================
# MODEL EVALUATION
# ============================================================

# ------------------------------------------------------------
# MAE - MEAN ABSOLUTE ERROR
# ------------------------------------------------------------

# Formula:
#
# MAE = average(|actual - predicted|)
#
# It represents the average absolute prediction error.
mae = mean_absolute_error(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# MSE - MEAN SQUARED ERROR
# ------------------------------------------------------------

# Formula:
#
# MSE = average((actual - predicted)^2)
#
# Large errors are penalized more strongly.
mse = mean_squared_error(
    y_test,
    y_pred
)


# ------------------------------------------------------------
# RMSE - ROOT MEAN SQUARED ERROR
# ------------------------------------------------------------

# RMSE = sqrt(MSE)
#
# RMSE is in the same units as SalePrice,
# making it easier to interpret.
rmse = np.sqrt(mse)


# ------------------------------------------------------------
# R-SQUARED
# ------------------------------------------------------------

# R² measures how much of the variation in SalePrice
# is explained by the regression model.
#
# R² = 1 means perfect prediction.
# Higher R² is generally better.
r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 70)
print("LINEAR REGRESSION RESULTS")
print("=" * 70)

print(f"Mean Absolute Error    : {mae:.2f}")
print(f"Mean Squared Error     : {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score               : {r2:.4f}")


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(8, 6))

# Each point represents one test observation.
# X-axis = actual price
# Y-axis = predicted price
sns.scatterplot(
    x=y_test,
    y=y_pred
)

plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Actual vs Predicted SalePrice")


# Find the minimum and maximum value
# between actual and predicted prices.
minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
)


# Draw the ideal prediction line y = x.
#
# If predictions were perfect,
# every point would lie on this line.
plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()
plt.show()


# ============================================================
# FINAL SAVE
# ============================================================

# Save the 10 regression features + SalePrice.
#
# NOTE:
# This overwrites the earlier house_prices_clean.csv.
# This preserves the behavior of the original code.
processed_df = df[
    features + ["SalePrice"]
].copy()

processed_df.to_csv(
    "house_prices_clean.csv",
    index=False
)

print("\nSaved as: house_prices_clean.csv")