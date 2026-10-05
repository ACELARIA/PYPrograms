# ============================================================
# ML Assignment 5
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    mean_squared_error, accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.datasets import load_iris


# ------------------------------------------------------------
# Common settings
# ------------------------------------------------------------
np.random.seed(42)


# ============================================================
# Q1. Batch Gradient Descent
# ============================================================

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add x0 = 1 for intercept
X_b = np.c_[np.ones((len(X), 1)), X]

eta = 0.1
n_iterations = 1000

# Random initial theta
theta = np.random.randn(2, 1)

mse_history_bgd = []

for iteration in range(n_iterations):
    gradients = (2 / len(X_b)) * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

    y_pred = X_b @ theta
    mse = mean_squared_error(y, y_pred)
    mse_history_bgd.append(mse)

print("\n================ Q1: BATCH GD ================")
print("Final theta [intercept, slope]:")
print(theta.ravel())
print("Final MSE:", mse_history_bgd[-1])

plt.figure(figsize=(8, 5))
plt.plot(range(1, n_iterations + 1), mse_history_bgd)
plt.xlabel("Iteration number")
plt.ylabel("MSE")
plt.title("Q1 - Batch Gradient Descent: MSE vs Iteration")
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Q2. Stochastic Gradient Descent
# ============================================================

# SGD parameters
t0 = 5
t1 = 50
n_epochs = 50

theta_sgd = np.random.randn(2, 1)
mse_history_sgd = []
updates_sgd = 0
rng = np.random.default_rng(42)

for epoch in range(n_epochs):
    # Shuffle training instances at the beginning of every epoch
    shuffled_indices = rng.permutation(len(X_b))

    for t, idx in enumerate(shuffled_indices):
        xi = X_b[idx:idx + 1]
        yi = y[idx:idx + 1]

        eta_t = t0 / (t + t1)

        gradients = 2 * xi.T @ (xi @ theta_sgd - yi)
        theta_sgd = theta_sgd - eta_t * gradients

        updates_sgd += 1

        # Record full-dataset MSE after each parameter update
        y_pred = X_b @ theta_sgd
        mse_history_sgd.append(mean_squared_error(y, y_pred))


print("\n================ Q2: STOCHASTIC GD ================")
print("Final theta [intercept, slope]:")
print(theta_sgd.ravel())
print("Final MSE:", mse_history_sgd[-1])
print("Total SGD parameter updates:", updates_sgd)


# Batch GD performed one parameter update per iteration.
updates_bgd = n_iterations
print("Total Batch GD parameter updates:", updates_bgd)


# Overlay MSE trajectories.
# Use update number on x-axis because SGD has 5000 updates and BGD has 1000.
plt.figure(figsize=(9, 5))
plt.plot(
    np.arange(1, len(mse_history_bgd) + 1),
    mse_history_bgd,
    label="Batch GD"
)
plt.plot(
    np.arange(1, len(mse_history_sgd) + 1),
    mse_history_sgd,
    alpha=0.7,
    label="Stochastic GD"
)
plt.xlabel("Parameter update")
plt.ylabel("Full-dataset MSE")
plt.title("Q2 - Batch GD vs Stochastic GD")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Q3. Mini-batch GD
# ============================================================

batch_size = 20
theta_mini = np.random.randn(2, 1)

mse_history_mini = []
epoch_mse_bgd = []
epoch_mse_sgd = []
epoch_mse_mini = []

# Re-run BGD so that we can record epoch-level MSE consistently.
theta_bgd_q3 = np.random.randn(2, 1)
bgd_reach_epoch = None

# For a fair comparison in Q3, use 50 epochs for all methods.
for epoch in range(50):
    gradients = (2 / len(X_b)) * X_b.T @ (X_b @ theta_bgd_q3 - y)
    theta_bgd_q3 -= eta * gradients

    mse = mean_squared_error(y, X_b @ theta_bgd_q3)
    epoch_mse_bgd.append(mse)

    if bgd_reach_epoch is None and mse < 1.0:
        bgd_reach_epoch = epoch + 1


# SGD: reset and run 50 epochs.
theta_sgd_q3 = np.random.randn(2, 1)
sgd_reach_epoch = None
rng_q3 = np.random.default_rng(42)

for epoch in range(50):
    shuffled_indices = rng_q3.permutation(len(X_b))

    for t, idx in enumerate(shuffled_indices):
        xi = X_b[idx:idx + 1]
        yi = y[idx:idx + 1]
        eta_t = t0 / (t + t1)

        gradients = 2 * xi.T @ (xi @ theta_sgd_q3 - yi)
        theta_sgd_q3 -= eta_t * gradients

    mse = mean_squared_error(y, X_b @ theta_sgd_q3)
    epoch_mse_sgd.append(mse)

    if sgd_reach_epoch is None and mse < 1.0:
        sgd_reach_epoch = epoch + 1


# Mini-batch GD: 50 epochs.
theta_mini = np.random.randn(2, 1)
mini_reach_epoch = None
rng_mini = np.random.default_rng(42)

for epoch in range(50):
    shuffled_indices = rng_mini.permutation(len(X_b))

    for start in range(0, len(X_b), batch_size):
        batch_indices = shuffled_indices[start:start + batch_size]
        xi = X_b[batch_indices]
        yi = y[batch_indices]

        gradients = (2 / len(xi)) * xi.T @ (xi @ theta_mini - yi)
        theta_mini -= eta * gradients

    mse = mean_squared_error(y, X_b @ theta_mini)
    epoch_mse_mini.append(mse)

    if mini_reach_epoch is None and mse < 1.0:
        mini_reach_epoch = epoch + 1


# Helper: if threshold is never reached
def threshold_result(value):
    return value if value is not None else "Not reached"


q3_table = pd.DataFrame({
    "Method": ["Batch GD", "Stochastic GD", "Mini-batch GD"],
    "Final Intercept": [
        theta_bgd_q3[0, 0],
        theta_sgd_q3[0, 0],
        theta_mini[0, 0]
    ],
    "Final Slope": [
        theta_bgd_q3[1, 0],
        theta_sgd_q3[1, 0],
        theta_mini[1, 0]
    ],
    "Epoch MSE < 1.0": [
        threshold_result(bgd_reach_epoch),
        threshold_result(sgd_reach_epoch),
        threshold_result(mini_reach_epoch)
    ]
})

print("\n================ Q3: COMPARISON TABLE ================")
print(q3_table.to_string(index=False))

plt.figure(figsize=(9, 5))
plt.plot(range(1, 51), epoch_mse_bgd, label="Batch GD")
plt.plot(range(1, 51), epoch_mse_sgd, label="Stochastic GD")
plt.plot(range(1, 51), epoch_mse_mini, label="Mini-batch GD")
plt.axhline(1.0, linestyle="--", label="MSE = 1.0")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Q3 - GD Variants")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Q4. Polynomial Regression: degrees 1, 2, 300
# ============================================================

rng = np.random.default_rng(42)

X_poly = np.sort(6 * rng.random((100, 1)) - 3, axis=0)
y_poly = (
    0.5 * X_poly**2
    + X_poly
    + 2
    + rng.standard_normal(X_poly.shape)
)

degrees = [1, 2, 300]

plt.figure(figsize=(10, 6))
plt.scatter(X_poly, y_poly, label="Training data", alpha=0.6)

x_plot = np.linspace(-3, 3, 500).reshape(-1, 1)

for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        LinearRegression()
    )
    model.fit(X_poly, y_poly)

    y_curve = model.predict(x_plot)
    plt.plot(x_plot, y_curve, label=f"Degree {degree}")

plt.xlabel("X")
plt.ylabel("y")
plt.title("Q4 - Polynomial Regression")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Learning curves for degree 1 and degree 300.
# RMSE is used, as requested.
def plot_learning_curve_degree(degree, X_data, y_data):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )

    # 5-fold CV. n_jobs omitted for compatibility.
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_data,
        y_data.ravel(),
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    train_rmse = -train_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_rmse, marker="o", label="Training RMSE")
    plt.plot(train_sizes, val_rmse, marker="o", label="Validation RMSE")
    plt.xlabel("Training-set size")
    plt.ylabel("RMSE")
    plt.title(f"Q4 - Learning Curve (Degree {degree})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


plot_learning_curve_degree(1, X_poly, y_poly)
plot_learning_curve_degree(300, X_poly, y_poly)


# ============================================================
# Q5. Wine Quality: Ridge, Lasso, ElasticNet
# ============================================================

wine = pd.read_csv("winequality-red.csv", sep=";")

X_wine = wine.drop(columns=["quality"])
y_wine = wine["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X_wine,
    y_wine,
    test_size=0.2,
    random_state=42
)

alphas = [0.001, 0.01, 0.1, 1, 10]

results = []

# Standardization is important for fair regularization.
# The target is not scaled.
for alpha in alphas:
    models = {
        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha)
        ),
        "Lasso": make_pipeline(
            StandardScaler(),
            Lasso(alpha=alpha, max_iter=100000)
        ),
        "ElasticNet": make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=100000)
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))

        results.append({
            "Model": name,
            "Alpha": alpha,
            "Test RMSE": rmse
        })

q5_table = pd.DataFrame(results)

print("\n================ Q5: WINE RMSE TABLE ================")
print(q5_table.to_string(index=False))


# Lasso coefficient paths.
# Fit standardized Lasso for every alpha.
feature_names = X_wine.columns
lasso_coefficients = {}

for alpha in alphas:
    lasso = make_pipeline(
        StandardScaler(),
        Lasso(alpha=alpha, max_iter=100000)
    )
    lasso.fit(X_train, y_train)

    # Coefficients are after StandardScaler, so all are comparable.
    coef = lasso.named_steps["lasso"].coef_
    lasso_coefficients[alpha] = coef

plt.figure(figsize=(12, 7))

for i, feature in enumerate(feature_names):
    values = [lasso_coefficients[a][i] for a in alphas]
    plt.plot(alphas, values, marker="o", label=feature)

plt.xscale("log")
plt.xlabel("Alpha")
plt.ylabel("Lasso coefficient")
plt.title("Q5 - Lasso Coefficient Value vs Alpha")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()


# "Reaches zero first" among the tested alpha values:
# Find the first alpha at which each coefficient is exactly/approximately zero.
zero_tol = 1e-10
zero_first = {}

for i, feature in enumerate(feature_names):
    for alpha in alphas:
        if abs(lasso_coefficients[alpha][i]) <= zero_tol:
            zero_first[feature] = alpha
            break

if zero_first:
    first_feature = min(zero_first, key=lambda f: zero_first[f])
    first_alpha = zero_first[first_feature]
    print(
        f"\nFeature whose coefficient reaches zero first: "
        f"{first_feature} (alpha={first_alpha})"
    )
else:
    print(
        "\nNo Lasso coefficient is exactly zero at the tested alpha values."
    )


# ============================================================
# Q6. Wine Quality Binary Classification
# ============================================================

y_binary = (wine["quality"] >= 7).astype(int)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_wine,
    y_binary,
    test_size=0.2,
    random_state=42,
    stratify=y_binary
)

log_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42)
)

log_model.fit(X_train_bin, y_train_bin)
y_pred_bin = log_model.predict(X_test_bin)

accuracy = accuracy_score(y_test_bin, y_pred_bin)
precision = precision_score(y_test_bin, y_pred_bin, zero_division=0)
recall = recall_score(y_test_bin, y_pred_bin, zero_division=0)
f1 = f1_score(y_test_bin, y_pred_bin, zero_division=0)

q6_table = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
    "Value": [accuracy, precision, recall, f1]
})

print("\n================ Q6: LOGISTIC REGRESSION ================")
print(q6_table.to_string(index=False))


# ============================================================
# Q7. Iris - Multinomial Logistic Regression
# ============================================================

iris = load_iris()

# Petal length = column 2
# Petal width  = column 3
X_iris = iris.data[:, [2, 3]]
y_iris = iris.target

iris_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        multi_class="multinomial",
        max_iter=5000,
        random_state=42
    )
)

iris_model.fit(X_iris, y_iris)

# Create a dense grid for the decision boundary.
x_min, x_max = X_iris[:, 0].min() - 0.5, X_iris[:, 0].max() + 0.5
y_min, y_max = X_iris[:, 1].min() - 0.5, X_iris[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = iris_model.predict(grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(9, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.25,
    levels=np.arange(-0.5, 3.5, 1)
)

for class_value, class_name in enumerate(iris.target_names):
    mask = y_iris == class_value
    plt.scatter(
        X_iris[mask, 0],
        X_iris[mask, 1],
        label=class_name
    )

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Q7 - Multinomial Logistic Regression Decision Boundary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

