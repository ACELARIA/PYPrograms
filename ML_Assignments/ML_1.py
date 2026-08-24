# ============================================================
# HOUSE PRICE PREDICTION
# Kaggle - House Prices: Advanced Regression Techniques
# Data Preprocessing, EDA and Linear Regression
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ============================================================
# PART A: DATA LOADING & BASIC INSPECTION
# ============================================================

# Load dataset
df = pd.read_csv("train.csv")

print("=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)

print(df.head(10))


# ------------------------------------------------------------
# Number of rows and columns
# ------------------------------------------------------------

print("\nDataset Shape:")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ------------------------------------------------------------
# Data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# Identify numerical and categorical columns
# ------------------------------------------------------------

numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include="object").columns.tolist()

print("\nNumber of numerical columns:", len(numerical_cols))
print("Number of categorical columns:", len(categorical_cols))

print("\nNumerical Columns:")
print(numerical_cols)

print("\nCategorical Columns:")
print(categorical_cols)


# ------------------------------------------------------------
# Check for incorrect data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPE INFORMATION")
print("=" * 70)

df.info()


# ============================================================
# PART B: UNIVARIATE ANALYSIS
# ============================================================

# ------------------------------------------------------------
# Numerical Features - Histograms
# ------------------------------------------------------------

key_features = [
    "LotArea",
    "GrLivArea",
    "SalePrice",
    "YearBuilt",
    "OverallQual"
]

plt.figure(figsize=(15, 10))

for i, feature in enumerate(key_features, 1):

    plt.subplot(2, 3, i)

    sns.histplot(
        df[feature],
        kde=True,
        bins=30
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Calculate skewness
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SKEWNESS OF KEY FEATURES")
print("=" * 70)

for feature in key_features:

    skewness = df[feature].skew()

    print(f"{feature}: {skewness:.3f}")

    if abs(skewness) < 0.5:
        print("  -> Approximately symmetric / normal")
    elif abs(skewness) < 1:
        print("  -> Moderately skewed")
    else:
        print("  -> Highly skewed")


# ------------------------------------------------------------
# Categorical Features
# ------------------------------------------------------------

categorical_features = [
    "Neighborhood",
    "HouseStyle",
    "BldgType"
]

for feature in categorical_features:

    plt.figure(figsize=(12, 5))

    sns.countplot(
        data=df,
        x=feature,
        order=df[feature].value_counts().index
    )

    plt.title(f"Distribution of {feature}")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Value counts for categorical features
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 MOST FREQUENT CATEGORIES")
print("=" * 70)

category_counts = []

for col in categorical_cols:

    counts = df[col].value_counts()

    for category, count in counts.items():

        category_counts.append(
            [col, category, count]
        )

category_counts_df = pd.DataFrame(
    category_counts,
    columns=["Feature", "Category", "Count"]
)

top_10_categories = (
    category_counts_df
    .sort_values("Count", ascending=False)
    .head(10)
)

print(top_10_categories)


# ------------------------------------------------------------
# Identify imbalanced categorical variables
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORY DOMINANCE")
print("=" * 70)

for col in categorical_cols:

    counts = df[col].value_counts(normalize=True)

    most_common_category = counts.index[0]
    percentage = counts.iloc[0] * 100

    print(
        f"{col}: {most_common_category} "
        f"= {percentage:.2f}%"
    )


# ============================================================
# PART C: BIVARIATE ANALYSIS
# ============================================================

# ------------------------------------------------------------
# Q7: GrLivArea vs SalePrice
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("GrLivArea vs SalePrice")
plt.xlabel("Above Ground Living Area")
plt.ylabel("Sale Price")

plt.show()


# ------------------------------------------------------------
# Boxplot: SalePrice vs OverallQual
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("SalePrice Across Overall Quality")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")

plt.show()


# ------------------------------------------------------------
# Correlation with SalePrice
# ------------------------------------------------------------

numeric_df = df.select_dtypes(include=np.number)

correlations = (
    numeric_df
    .corr()["SalePrice"]
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("CORRELATION WITH SALEPRICE")
print("=" * 70)

print(correlations)


# Top 10 features correlated with SalePrice
top_correlated = (
    correlations
    .abs()
    .sort_values(ascending=False)
    .head(11)
    .index
)

# Include SalePrice itself, hence 11
top_corr_df = df[top_correlated].corr()


# ------------------------------------------------------------
# Correlation Heatmap
# ------------------------------------------------------------

plt.figure(figsize=(12, 9))

sns.heatmap(
    top_corr_df,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Top 10 Features Correlated with SalePrice")

plt.show()


# ------------------------------------------------------------
# Compare two neighborhoods
# ------------------------------------------------------------

# Choosing two large/common neighborhoods
neighborhoods = ["NAmes", "CollgCr"]

neighborhood_data = df[
    df["Neighborhood"].isin(neighborhoods)
]

print("\n" + "=" * 70)
print("AVERAGE SALE PRICE BY NEIGHBORHOOD")
print("=" * 70)

print(
    neighborhood_data
    .groupby("Neighborhood")["SalePrice"]
    .mean()
)


# ------------------------------------------------------------
# T-test between neighborhoods
# ------------------------------------------------------------

group1 = df[
    df["Neighborhood"] == neighborhoods[0]
]["SalePrice"]

group2 = df[
    df["Neighborhood"] == neighborhoods[1]
]["SalePrice"]

t_stat, p_value = stats.ttest_ind(
    group1,
    group2,
    equal_var=False
)

print("\nT-test:")
print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Significant difference in average SalePrice.")
else:
    print("No significant difference in average SalePrice.")


# ------------------------------------------------------------
# New homes vs Old homes
# ------------------------------------------------------------

df["HomeAgeGroup"] = np.where(
    df["YearBuilt"] >= 2000,
    "Newer Homes",
    "Older Homes"
)


# Boxplot

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
# T-test
# ------------------------------------------------------------

newer_homes = df[
    df["YearBuilt"] >= 2000
]["SalePrice"]

older_homes = df[
    df["YearBuilt"] < 2000
]["SalePrice"]

t_stat, p_value = stats.ttest_ind(
    newer_homes,
    older_homes,
    equal_var=False
)

print("\n" + "=" * 70)
print("T-TEST: NEWER VS OLDER HOMES")
print("=" * 70)

print("Newer homes average:",
      newer_homes.mean())

print("Older homes average:",
      older_homes.mean())

print("t-statistic:", t_stat)
print("p-value:", p_value)

if p_value < 0.05:
    print("Result: Significant difference in mean SalePrice.")
else:
    print("Result: No significant difference in mean SalePrice.")


# ============================================================
# PART D: MISSING VALUE TREATMENT
# ============================================================

# ------------------------------------------------------------
# Missing values in each column
# ------------------------------------------------------------

missing_values = df.isnull().sum()

missing_values = (
    missing_values
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    missing_values[
        missing_values > 0
    ]
)


# ------------------------------------------------------------
# Top 5 columns with most missing values
# ------------------------------------------------------------

print("\nTop 5 columns with most missing values:")

print(
    missing_values
    .head(5)
)


# ------------------------------------------------------------
# LotFrontage grouped median imputation
# ------------------------------------------------------------

df["LotFrontage"] = (
    df.groupby("Neighborhood")["LotFrontage"]
    .transform(
        lambda x: x.fillna(x.median())
    )
)


# If any LotFrontage values still remain missing,
# fill them with the overall median.

df["LotFrontage"] = df["LotFrontage"].fillna(
    df["LotFrontage"].median()
)


# ------------------------------------------------------------
# Numerical columns - median imputation
# ------------------------------------------------------------

numerical_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

for col in numerical_cols:

    if df[col].isnull().sum() > 0:

        df[col] = df[col].fillna(
            df[col].median()
        )


# ------------------------------------------------------------
# Categorical columns
# ------------------------------------------------------------

categorical_cols = df.select_dtypes(
    include="object"
).columns.tolist()

for col in categorical_cols:

    if df[col].isnull().sum() > 0:

        df[col] = df[col].fillna("Missing")


# ------------------------------------------------------------
# Verify missing values
# ------------------------------------------------------------

remaining_missing = df.isnull().sum().sum()

print("\n" + "=" * 70)
print("MISSING VALUES AFTER IMPUTATION")
print("=" * 70)

print("Total missing values:", remaining_missing)


# ============================================================
# PART E: OUTLIER DETECTION & TREATMENT
# ============================================================

# ------------------------------------------------------------
# IQR function
# ------------------------------------------------------------

def detect_outliers_iqr(data, column):

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_bound) |
        (data[column] > upper_bound)
    ]

    return outliers, lower_bound, upper_bound


# ------------------------------------------------------------
# LotArea outliers
# ------------------------------------------------------------

lot_outliers, lot_lower, lot_upper = (
    detect_outliers_iqr(df, "LotArea")
)

print("\n" + "=" * 70)
print("LOTAREA OUTLIERS")
print("=" * 70)

print("Q1:", df["LotArea"].quantile(0.25))
print("Q3:", df["LotArea"].quantile(0.75))
print("IQR:",
      df["LotArea"].quantile(0.75) -
      df["LotArea"].quantile(0.25))

print("Lower bound:", lot_lower)
print("Upper bound:", lot_upper)

print("Number of outliers:",
      len(lot_outliers))


# ------------------------------------------------------------
# GrLivArea outliers
# ------------------------------------------------------------

grliv_outliers, grliv_lower, grliv_upper = (
    detect_outliers_iqr(df, "GrLivArea")
)

print("\n" + "=" * 70)
print("GRLIVAREA OUTLIERS")
print("=" * 70)

print("Q1:", df["GrLivArea"].quantile(0.25))
print("Q3:", df["GrLivArea"].quantile(0.75))
print("IQR:",
      df["GrLivArea"].quantile(0.75) -
      df["GrLivArea"].quantile(0.25))

print("Lower bound:", grliv_lower)
print("Upper bound:", grliv_upper)

print("Number of outliers:",
      len(grliv_outliers))


# ------------------------------------------------------------
# Cap extreme outliers
# ------------------------------------------------------------

# Instead of deleting rows, we cap values at
# the IQR lower/upper bounds.

df["LotArea"] = df["LotArea"].clip(
    lower=lot_lower,
    upper=lot_upper
)

df["GrLivArea"] = df["GrLivArea"].clip(
    lower=grliv_lower,
    upper=grliv_upper
)


# ------------------------------------------------------------
# Replot scatterplot
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title(
    "GrLivArea vs SalePrice After Outlier Capping"
)

plt.xlabel("Above Ground Living Area")
plt.ylabel("Sale Price")

plt.show()


# ============================================================
# PART F: FEATURE ENCODING & SCALING
# ============================================================

# ------------------------------------------------------------
# Separate target variable
# ------------------------------------------------------------

X = df.drop(
    columns=["SalePrice"]
)

y = df["SalePrice"]


# ------------------------------------------------------------
# Remove ID
# ------------------------------------------------------------

# Id is only an identifier and has no useful
# predictive meaning.

if "Id" in X.columns:

    X = X.drop(columns=["Id"])


# ------------------------------------------------------------
# One-Hot Encoding
# ------------------------------------------------------------

X_encoded = pd.get_dummies(
    X,
    columns=X.select_dtypes(
        include="object"
    ).columns,
    drop_first=True
)


# ------------------------------------------------------------
# Convert Boolean columns to integers
# ------------------------------------------------------------

X_encoded = X_encoded.astype(float)


# ------------------------------------------------------------
# Standardize numerical features
# ------------------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_encoded)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=X_encoded.columns
)


print("\n" + "=" * 70)
print("PROCESSED DATASET")
print("=" * 70)

print("Original feature count:",
      X.shape[1])

print("Encoded feature count:",
      X_encoded.shape[1])

print("Scaled dataset shape:",
      X_scaled.shape)


# ------------------------------------------------------------
# Save cleaned and processed dataset
# ------------------------------------------------------------

processed_df = X_scaled.copy()

processed_df["SalePrice"] = y.values

processed_df.to_csv(
    "house_prices_clean.csv",
    index=False
)

print("\nSaved as: house_prices_clean.csv")


# ============================================================
# PART G: LINEAR REGRESSION
# ============================================================

# Select the strongest numerical features
features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "TotRmsAbvGrd",
    "YearBuilt",
    "YearRemodAdd"
]

X = df[features]
y = df["SalePrice"]


# ------------------------------------------------------------
# Train-Test Split
# ------------------------------------------------------------

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
# Standardization
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# Linear Regression Model
# ------------------------------------------------------------

model = LinearRegression()

model.fit(
    X_train_scaled,
    y_train
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n" + "=" * 70)
print("LINEAR REGRESSION RESULTS")
print("=" * 70)

print(f"Mean Absolute Error    : {mae:.2f}")
print(f"Mean Squared Error     : {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score               : {r2:.4f}")


# ------------------------------------------------------------
# Actual vs Predicted
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test,
    y=y_pred
)

plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")

plt.title("Actual vs Predicted SalePrice")

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Save processed dataset
# ------------------------------------------------------------

processed_df = df[features + ["SalePrice"]].copy()

processed_df.to_csv(
    "house_prices_clean.csv",
    index=False
)

print("\nSaved as: house_prices_clean.csv")