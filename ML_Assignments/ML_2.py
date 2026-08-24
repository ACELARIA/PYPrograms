# ============================================================
# EXPERIMENT: SIMPLE AND MULTIPLE LINEAR REGRESSION
# Dataset: Ames Housing - Kaggle House Prices
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("train.csv")

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 10 rows:")
print(df.head(10))


# ============================================================
# PART A: EXPLORATORY DATA ANALYSIS
# ============================================================

# ------------------------------------------------------------
# 1. Distribution of target variable
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.histplot(
    df["SalePrice"],
    kde=True,
    bins=30
)

plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")

plt.show()


# Calculate skewness

skewness = df["SalePrice"].skew()

print("\n" + "=" * 70)
print("TARGET SKEWNESS")
print("=" * 70)

print("SalePrice skewness:", skewness)

if skewness > 1:
    print("SalePrice is highly positively skewed.")
elif skewness > 0.5:
    print("SalePrice is moderately positively skewed.")
else:
    print("SalePrice is approximately symmetric.")


# ------------------------------------------------------------
# 2. Correlation matrix
# ------------------------------------------------------------

numeric_df = df.select_dtypes(
    include=np.number
)

correlation_matrix = numeric_df.corr()


plt.figure(figsize=(15, 12))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix")

plt.show()


# ------------------------------------------------------------
# Features most correlated with SalePrice
# ------------------------------------------------------------

correlation_with_target = (
    numeric_df.corr()["SalePrice"]
    .sort_values(ascending=False)
)

print("\n" + "=" * 70)
print("CORRELATION WITH SALEPRICE")
print("=" * 70)

print(correlation_with_target)


# ============================================================
# PART B: SIMPLE LINEAR REGRESSION
# ============================================================

# Select one predictor
# GrLivArea has a strong correlation with SalePrice.

X = df[["GrLivArea"]]
y = df["SalePrice"]


# ------------------------------------------------------------
# Hypothesis equation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SIMPLE LINEAR REGRESSION")
print("=" * 70)

print("Hypothesis equation:")
print("y = β0 + β1x + ε")

print("\nHere:")
print("y  = SalePrice")
print("x  = GrLivArea")
print("β0 = Intercept")
print("β1 = Slope")
print("ε  = Error term")


# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# Train model
# ------------------------------------------------------------

simple_model = LinearRegression()

simple_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Intercept and slope
# ------------------------------------------------------------

intercept = simple_model.intercept_

slope = simple_model.coef_[0]


print("\nIntercept (β0):", intercept)
print("Slope (β1):", slope)


# ------------------------------------------------------------
# Fitted equation
# ------------------------------------------------------------

print("\nFitted equation:")

print(
    f"SalePrice = {intercept:.2f} + "
    f"({slope:.2f} × GrLivArea)"
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

simple_predictions = simple_model.predict(X_test)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

simple_r2 = r2_score(
    y_test,
    simple_predictions
)

simple_mse = mean_squared_error(
    y_test,
    simple_predictions
)

simple_rmse = np.sqrt(simple_mse)


print("\nSimple Regression Performance:")
print("R²   :", simple_r2)
print("MSE  :", simple_mse)
print("RMSE :", simple_rmse)


# ------------------------------------------------------------
# Scatter plot + regression line
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    x=df["GrLivArea"],
    y=df["SalePrice"]
)

# Regression line
x_line = np.linspace(
    df["GrLivArea"].min(),
    df["GrLivArea"].max(),
    100
)

y_line = (
    intercept +
    slope * x_line
)

plt.plot(
    x_line,
    y_line,
    linestyle="--",
    linewidth=2
)

plt.title(
    "Simple Linear Regression: "
    "GrLivArea vs SalePrice"
)

plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")

plt.show()


# ------------------------------------------------------------
# Interpret slope
# ------------------------------------------------------------

print("\nSlope interpretation:")

print(
    f"For every 1 square unit increase in GrLivArea, "
    f"the predicted SalePrice increases by approximately "
    f"{slope:.2f} units, on average."
)


# ============================================================
# PART C: MULTIPLE LINEAR REGRESSION
# ============================================================

# Select at least 3 predictors
#
# These are strong predictors based on correlation.

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars"
]

X = df[features]
y = df["SalePrice"]


# ------------------------------------------------------------
# Train-test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# Train multiple regression model
# ------------------------------------------------------------

multiple_model = LinearRegression()

multiple_model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Coefficients
# ------------------------------------------------------------

multiple_intercept = multiple_model.intercept_

multiple_coefficients = multiple_model.coef_


print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 70)

print("Intercept:", multiple_intercept)

print("\nCoefficients:")

for feature, coefficient in zip(
    features,
    multiple_coefficients
):

    print(
        f"{feature}: {coefficient:.2f}"
    )


# ------------------------------------------------------------
# Fitted equation
# ------------------------------------------------------------

print("\nFitted equation:")

equation = (
    f"SalePrice = {multiple_intercept:.2f}"
)

for feature, coefficient in zip(
    features,
    multiple_coefficients
):

    sign = "+" if coefficient >= 0 else "-"

    equation += (
        f" {sign} "
        f"{abs(coefficient):.2f}({feature})"
    )

print(equation)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

multiple_predictions = multiple_model.predict(
    X_test
)


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

multiple_r2 = r2_score(
    y_test,
    multiple_predictions
)

multiple_mse = mean_squared_error(
    y_test,
    multiple_predictions
)

multiple_rmse = np.sqrt(
    multiple_mse
)


print("\nMultiple Regression Performance:")

print("R²   :", multiple_r2)

print("MSE  :", multiple_mse)

print("RMSE :", multiple_rmse)


# ============================================================
# COMPARE SIMPLE VS MULTIPLE REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Simple Linear Regression",
        "Multiple Linear Regression"
    ],

    "R2": [
        simple_r2,
        multiple_r2
    ],

    "MSE": [
        simple_mse,
        multiple_mse
    ],

    "RMSE": [
        simple_rmse,
        multiple_rmse
    ]
})

print(comparison)


# ------------------------------------------------------------
# Determine better model
# ------------------------------------------------------------

if multiple_r2 > simple_r2:
    print(
        "\nMultiple Linear Regression performs better "
        "because it has a higher R² score."
    )
else:
    print(
        "\nSimple Linear Regression performs better "
        "based on R²."
    )


# ============================================================
# COEFFICIENT INTERPRETATION
# ============================================================

print("\n" + "=" * 70)
print("COEFFICIENT INTERPRETATION")
print("=" * 70)

print(
    f"OverallQual coefficient = "
    f"{multiple_coefficients[0]:.2f}"
)

print(
    "Keeping GrLivArea and GarageCars constant, "
    "a one-unit increase in OverallQual is associated "
    f"with an average change of "
    f"{multiple_coefficients[0]:.2f} in SalePrice."
)