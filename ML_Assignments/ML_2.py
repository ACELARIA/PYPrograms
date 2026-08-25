# ============================================================
# SIMPLE AND MULTIPLE LINEAR REGRESSION - HOUSE PRICES
# ============================================================

# Import libraries:
# pandas -> data handling
# numpy -> numerical calculations
# matplotlib/seaborn -> graphs
# sklearn -> machine learning and evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ignore unnecessary warning messages
import warnings
warnings.filterwarnings("ignore")

# Set the style of Seaborn graphs
sns.set_theme(style="whitegrid")


# ============================================================
# LOAD DATASET
# ============================================================

# Read the Kaggle House Prices dataset
df = pd.read_csv("train.csv")

# Display basic information about the dataset
print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

# shape[0] = number of rows, shape[1] = number of columns
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Display the first 10 rows
print("\nFirst 10 rows:")
print(df.head(10))


# ============================================================
# PART A: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# EDA means understanding the data before building the model.


# -------------------- TARGET DISTRIBUTION --------------------

# Plot the distribution of SalePrice.
# Histogram shows frequency; KDE shows the smooth distribution curve.
sns.histplot(df["SalePrice"], kde=True, bins=30)

plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()


# -------------------- SKEWNESS --------------------

# Skewness measures how asymmetric the distribution is.
# Positive value -> right/positive skew
# Negative value -> left/negative skew
skewness = df["SalePrice"].skew()

print("\n" + "=" * 70)
print("TARGET SKEWNESS")
print("=" * 70)
print("SalePrice skewness:", skewness)

# Interpret the amount of positive skewness
if skewness > 1:
    print("SalePrice is highly positively skewed.")
elif skewness > 0.5:
    print("SalePrice is moderately positively skewed.")
else:
    print("SalePrice is approximately symmetric.")


# -------------------- CORRELATION MATRIX --------------------

# Select only numerical columns because correlation is calculated
# between numerical variables.
numeric_df = df.select_dtypes(include=np.number)

# Create a heatmap to visualize correlations between variables.
# Correlation ranges from -1 to +1.
# +1 -> strong positive relationship
# -1 -> strong negative relationship
#  0 -> little/no linear relationship

plt.figure(figsize=(15, 12))
sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)

plt.title("Correlation Matrix")
plt.show()


# -------------------- CORRELATION WITH TARGET --------------------

# Find how strongly every numerical feature is correlated
# with SalePrice and arrange from highest to lowest.
correlation = numeric_df.corr()["SalePrice"].sort_values(ascending=False)

print("\n" + "=" * 70)
print("CORRELATION WITH SALEPRICE")
print("=" * 70)
print(correlation)


# ============================================================
# PART B: SIMPLE LINEAR REGRESSION
# ============================================================

# Simple regression uses ONE predictor.
# GrLivArea = Above-ground living area.
# SalePrice = target we want to predict.

X = df[["GrLivArea"]]
y = df["SalePrice"]

print("\n" + "=" * 70)
print("SIMPLE LINEAR REGRESSION")
print("=" * 70)

# General hypothesis equation:
# y = β0 + β1x + ε
#
# β0 = intercept
# β1 = slope/coefficient
# ε  = error

print("Hypothesis equation:")
print("y = β0 + β1x + ε")

print("\nHere:")
print("y  = SalePrice")
print("x  = GrLivArea")
print("β0 = Intercept")
print("β1 = Slope")
print("ε  = Error term")


# -------------------- TRAIN-TEST SPLIT --------------------

# Divide data into:
# 80% training data -> used to learn the model
# 20% testing data  -> used to evaluate the model
#
# random_state=42 makes the split reproducible.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -------------------- TRAIN MODEL --------------------

# Create a Linear Regression model and train it.
# fit() learns the best intercept and coefficient
# from the training data.

simple_model = LinearRegression().fit(X_train, y_train)


# -------------------- INTERCEPT AND SLOPE --------------------

# intercept_ gives β0
# coef_[0] gives β1 because we have only one predictor

intercept = simple_model.intercept_
slope = simple_model.coef_[0]

print("\nIntercept (β0):", intercept)
print("Slope (β1):", slope)


# -------------------- FITTED EQUATION --------------------

# Display the actual equation learned by the model.
print("\nFitted equation:")
print(f"SalePrice = {intercept:.2f} + ({slope:.2f} × GrLivArea)")


# -------------------- PREDICTION --------------------

# predict() uses the trained model to predict SalePrice
# for previously unseen test data.

simple_predictions = simple_model.predict(X_test)


# -------------------- MODEL EVALUATION --------------------

# This function calculates three important metrics:
# R²   -> proportion of variance explained; higher is better
# MSE  -> average squared error; lower is better
# RMSE -> square root of MSE; lower is better

def evaluate(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    return r2_score(y_test, predictions), mse, np.sqrt(mse)


# Evaluate the simple regression model
simple_r2, simple_mse, simple_rmse = evaluate(
    y_test,
    simple_predictions
)

print("\nSimple Regression Performance:")
print("R²   :", simple_r2)
print("MSE  :", simple_mse)
print("RMSE :", simple_rmse)


# -------------------- REGRESSION PLOT --------------------

# Create 100 equally spaced X values to draw a smooth regression line.
x_line = np.linspace(
    df["GrLivArea"].min(),
    df["GrLivArea"].max(),
    100
)

# Calculate predicted Y values using:
# y = β0 + β1x
y_line = intercept + slope * x_line

# Plot actual data points
plt.figure(figsize=(9, 6))
sns.scatterplot(
    x=df["GrLivArea"],
    y=df["SalePrice"]
)

# Plot the regression line
plt.plot(
    x_line,
    y_line,
    linestyle="--",
    linewidth=2
)

plt.title("Simple Linear Regression: GrLivArea vs SalePrice")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.show()


# -------------------- SLOPE INTERPRETATION --------------------

# A slope tells us how much the predicted target changes
# when the predictor increases by one unit.

print("\nSlope interpretation:")
print(
    f"For every 1 square unit increase in GrLivArea, "
    f"the predicted SalePrice increases by approximately "
    f"{slope:.2f} units, on average."
)


# ============================================================
# PART C: MULTIPLE LINEAR REGRESSION
# ============================================================

# Multiple regression uses TWO OR MORE predictors.
# Here we use three:
# OverallQual -> overall material/finish quality
# GrLivArea   -> above-ground living area
# GarageCars  -> garage capacity in cars

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars"
]

X = df[features]


# -------------------- TRAIN-TEST SPLIT --------------------

# Again, use 80% for training and 20% for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -------------------- TRAIN MODEL --------------------

# Train a multiple linear regression model.
#
# General equation:
# y = β0 + β1x1 + β2x2 + β3x3

multiple_model = LinearRegression().fit(
    X_train,
    y_train
)


# -------------------- COEFFICIENTS --------------------

# Get the intercept β0 and coefficients β1, β2, β3.

multiple_intercept = multiple_model.intercept_
multiple_coefficients = multiple_model.coef_

print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 70)

print("Intercept:", multiple_intercept)

print("\nCoefficients:")

# Match each feature with its coefficient
for feature, coefficient in zip(
    features,
    multiple_coefficients
):
    print(f"{feature}: {coefficient:.2f}")


# -------------------- FITTED EQUATION --------------------

# Start the equation with the intercept.
equation = f"SalePrice = {multiple_intercept:.2f}"

# Add each feature and its coefficient.
for feature, coefficient in zip(
    features,
    multiple_coefficients
):

    # Determine whether the coefficient is positive or negative.
    sign = "+" if coefficient >= 0 else "-"

    equation += (
        f" {sign} "
        f"{abs(coefficient):.2f}({feature})"
    )

print("\nFitted equation:")
print(equation)


# -------------------- PREDICTION --------------------

# Predict SalePrice using the test data.
multiple_predictions = multiple_model.predict(X_test)


# -------------------- MODEL EVALUATION --------------------

# Calculate R², MSE and RMSE for multiple regression.

multiple_r2, multiple_mse, multiple_rmse = evaluate(
    y_test,
    multiple_predictions
)

print("\nMultiple Regression Performance:")
print("R²   :", multiple_r2)
print("MSE  :", multiple_mse)
print("RMSE :", multiple_rmse)


# ============================================================
# MODEL COMPARISON
# ============================================================

# Compare simple and multiple regression using:
# R²  -> higher is better
# MSE  -> lower is better
# RMSE -> lower is better

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


# Select the better model based on R².
# Higher R² means more variation in SalePrice is explained.

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

# Interpret the first coefficient: OverallQual.
# In multiple regression, we keep the other predictors constant.

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