# Import all necessary libraries for data processing, modeling, evaluation, and visualization
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Scikit-Learn Modules
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure visualization styling
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.dpi'] = 120

print("All dependencies and libraries imported successfully.")

# Load the California Housing dataset
csv_path = 'housing.csv'
housing = pd.read_csv(csv_path)

print(f"Dataset Shape: {housing.shape[0]} rows, {housing.shape[1]} columns\n")
print("First 5 rows of the dataset:")
print(housing.head())

print("\nDataset Information:")
print(housing.info())

print("\nDescriptive Statistics of Numerical Attributes:")
print(housing.describe().T)

print("\nMissing Value Analysis:")
missing_df = pd.DataFrame({
    'Missing Values': housing.isnull().sum(),
    'Percentage (%)': (housing.isnull().sum() / len(housing)) * 100
})
print(missing_df[missing_df['Missing Values'] > 0])

print("\nCategorical Value Counts for 'ocean_proximity':")
print(housing['ocean_proximity'].value_counts().to_frame(name='Count'))
# Step 1: Create discrete income category column
housing['income_cat'] = pd.cut(
    housing['median_income'],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Step 2: Perform Stratified Shuffle Split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# Step 3: Tabulate counts and proportions
train_counts = strat_train_set['income_cat'].value_counts().sort_index()
test_counts = strat_test_set['income_cat'].value_counts().sort_index()
overall_counts = housing['income_cat'].value_counts().sort_index()

income_split_comparison = pd.DataFrame({
    'Train Count': train_counts,
    'Train Prop (%)': (train_counts / len(strat_train_set)) * 100,
    'Test Count': test_counts,
    'Test Prop (%)': (test_counts / len(strat_test_set)) * 100,
    'Overall Prop (%)': (overall_counts / len(housing)) * 100
})

print("=" * 75)
print("QUESTION 1: Stratified Sampling Verification Table")
print("=" * 75)
print(income_split_comparison)

# Visualization of Stratification Consistency
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(1, 6)
width = 0.25

ax.bar(x - width, income_split_comparison['Overall Prop (%)'], width, label='Overall Data', color='#4C72B0')
ax.bar(x, income_split_comparison['Train Prop (%)'], width, label='Train Set (80%)', color='#55A868')
ax.bar(x + width, income_split_comparison['Test Prop (%)'], width, label='Test Set (20%)', color='#C44E52')

ax.set_title('Income Category Proportions Across Overall, Train, and Test Sets', pad=12, fontweight='bold')
ax.set_xlabel('Income Category (1: Lowest to 5: Highest)')
ax.set_ylabel('Proportion (%)')
ax.set_xticks(x)
ax.legend(frameon=True)
plt.tight_layout()
plt.show()

# Separate features (X) and target (y), removing helper income_cat
X_train = strat_train_set.drop(['median_house_value', 'income_cat'], axis=1)
y_train = strat_train_set['median_house_value'].copy()

X_test = strat_test_set.drop(['median_house_value', 'income_cat'], axis=1)
y_test = strat_test_set['median_house_value'].copy()

print(f"Training Features Shape : {X_train.shape}")
print(f"Training Target Shape   : {y_train.shape}")
print(f"Test Features Shape     : {X_test.shape}")
print(f"Test Target Shape       : {y_test.shape}")
# Define numerical and categorical column lists
num_attribs = list(X_train.select_dtypes(include=[np.number]).columns)
cat_attribs = ['ocean_proximity']

# Construct the numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler()),
])

# Construct the complete full ColumnTransformer preprocessor
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_attribs),
])

# Fit-transform the training dataset
X_train_prepared = full_pipeline.fit_transform(X_train)

# Extract generated feature names
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
all_feature_names = num_attribs + cat_one_hot_attribs

print("=" * 70)
print("QUESTION 2: ColumnTransformer Pipeline Output")
print("=" * 70)
print(f"Shape of Prepared Training Array: {X_train_prepared.shape}")
print(f"Number of Rows (Instances)      : {X_train_prepared.shape[0]}")
print(f"Number of Columns (Features)    : {X_train_prepared.shape[1]}\n")

print("Transformed Feature Names (13 Total):")
for idx, name in enumerate(all_feature_names, 1):
    print(f"  {idx:2d}. {name}")

# Preview transformed data as DataFrame
df_prepared_preview = pd.DataFrame(X_train_prepared[:5], columns=all_feature_names)
print("\nFirst 5 Transformed Training Samples:")
print(df_prepared_preview)
# Instantiate the three baseline regression models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

cv_results = {}

print("=" * 80)
print("QUESTION 3: 5-Fold Cross-Validation Model Comparison")
print("=" * 80)

for name, model in models.items():
    # Execute 5-fold cross validation
    neg_rmse_scores = cross_val_score(
        model,
        X_train_prepared,
        y_train,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    rmse_scores = -neg_rmse_scores
    cv_results[name] = {
        'Fold Scores (RMSE)': [f"${s:,.2f}" for s in rmse_scores],
        'Mean RMSE ($)': rmse_scores.mean(),
        'Std Dev RMSE ($)': rmse_scores.std(),
        'Raw Scores': rmse_scores
    }
    print(f"{name:25s} -> Mean RMSE: ${rmse_scores.mean():,.2f}  (+/- ${rmse_scores.std():,.2f})")

# Tabulate the summary
summary_df = pd.DataFrame([
    {
        'Model': name,
        'Mean RMSE ($)': f"${data['Mean RMSE ($)']:,.2f}",
        'Std Dev ($)': f"${data['Std Dev RMSE ($)']:,.2f}",
        'Min Fold RMSE ($)': f"${data['Raw Scores'].min():,.2f}",
        'Max Fold RMSE ($)': f"${data['Raw Scores'].max():,.2f}"
    }
    for name, data in cv_results.items()
])

print("\nDetailed Summary Table:")
print(summary_df)

# Visual Comparison of 5-Fold Cross Validation Distributions
fig, ax = plt.subplots(figsize=(9, 4.5))
box_data = [cv_results[m]['Raw Scores'] for m in models.keys()]
bp = ax.boxplot(box_data, tick_labels=list(models.keys()), patch_artist=True,
                boxprops=dict(facecolor='#aec7e8', color='#1f77b4'),
                medianprops=dict(color='darkred', linewidth=2))

ax.set_title('5-Fold Cross-Validation RMSE Distribution Across Regressors', pad=12, fontweight='bold')
ax.set_ylabel('Root Mean Squared Error (USD $)')
plt.tight_layout()
plt.show()
# Define hyperparameter grid
param_grid = [
    {
        'n_estimators': [50, 100, 200],
        'max_features': [2, 4, 6, 8]
    }
]

forest_reg = RandomForestRegressor(random_state=42, n_jobs=-1)

# Configure GridSearchCV with 5-fold CV
grid_search = GridSearchCV(
    estimator=forest_reg,
    param_grid=param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    return_train_score=True,
    n_jobs=-1
)

# Execute grid search on prepared training data
print("Executing GridSearchCV on RandomForestRegressor (60 fits)...")
grid_search.fit(X_train_prepared, y_train)

best_params = grid_search.best_params_
best_cv_rmse = -grid_search.best_score_

print("=" * 75)
print("QUESTION 4: Best Hyperparameters & CV Score")
print("=" * 75)
print(f"Optimal Hyperparameters : {best_params}")
print(f"Best 5-Fold CV RMSE     : ${best_cv_rmse:,.2f}\n")

# Tabulate full grid search evaluation results
cvres = grid_search.cv_results_
grid_results_list = []
for mean_score, std_score, params in zip(cvres["mean_test_score"], cvres["std_test_score"], cvres["params"]):
    grid_results_list.append({
        'n_estimators': params['n_estimators'],
        'max_features': params['max_features'],
        'Mean CV RMSE ($)': -mean_score,
        'Std Dev ($)': std_score
    })

grid_df = pd.DataFrame(grid_results_list).sort_values(by='Mean CV RMSE ($)')
print(grid_df)

# Plot hyperparameter response curve
fig, ax = plt.subplots(figsize=(9, 5))
for n_est in [50, 100, 200]:
    subset = grid_df[grid_df['n_estimators'] == n_est].sort_values(by='max_features')
    ax.plot(subset['max_features'], subset['Mean CV RMSE ($)'], marker='o', linewidth=2, label=f'n_estimators = {n_est}')

ax.set_title('Random Forest Grid Search: Effect of max_features & n_estimators on CV RMSE', pad=12, fontweight='bold')
ax.set_xlabel('max_features')
ax.set_ylabel('5-Fold Mean CV RMSE ($)')
ax.legend(title='Ensemble Size', frameon=True)
plt.tight_layout()
plt.show()
# Extract best tuned model from grid search
final_model = grid_search.best_estimator_

# Preprocess test set using the fitted pipeline (WITHOUT fitting again)
X_test_prepared = full_pipeline.transform(X_test)

# Generate predictions on the held-out test set
final_predictions = final_model.predict(X_test_prepared)

# Compute performance metrics
test_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
test_mae = mean_absolute_error(y_test, final_predictions)
test_r2 = r2_score(y_test, final_predictions)

# Compute 95% confidence interval for test RMSE
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
ci_lower = np.sqrt(stats.t.interval(
    confidence,
    len(squared_errors) - 1,
    loc=squared_errors.mean(),
    scale=stats.sem(squared_errors)
)[0])
ci_upper = np.sqrt(stats.t.interval(
    confidence,
    len(squared_errors) - 1,
    loc=squared_errors.mean(),
    scale=stats.sem(squared_errors)
)[1])

print("=" * 70)
print("QUESTION 5: Final Held-Out Test Set Evaluation")
print("=" * 70)
print(f"Test Root Mean Squared Error (RMSE) : ${test_rmse:,.2f}")
print(f"Test Mean Absolute Error (MAE)      : ${test_mae:,.2f}")
print(f"Test Coefficient of Determination (R2): {test_r2:.4f} ({test_r2*100:.2f}% variance explained)")
print(f"95% Confidence Interval for RMSE   : [${ci_lower:,.2f}, ${ci_upper:,.2f}]")

# Tabulate metrics
test_metrics_df = pd.DataFrame({
    'Metric': ['Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)', 'R² Score (Variance Explained)', '95% CI (Lower)', '95% CI (Upper)'],
    'Value': [f"${test_rmse:,.2f}", f"${test_mae:,.2f}", f"{test_r2:.4f}", f"${ci_lower:,.2f}", f"${ci_upper:,.2f}"]
})
print(test_metrics_df)
# Extract the two specified features and target from the full dataset
features_subset = ['median_income', 'housing_median_age']
X_sub = housing[features_subset].values
y_sub = housing['median_house_value'].values

# Step 1: NumPy Normal Equation Implementation
# Add bias column (vector of 1s) to X
X_b = np.c_[np.ones((len(X_sub), 1)), X_sub]

# Closed-form Normal Equation: theta = (X^T * X)^(-1) * X^T * y
theta_numpy = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_sub)

# Step 2: Scikit-Learn LinearRegression implementation
lin_reg_2f = LinearRegression()
lin_reg_2f.fit(X_sub, y_sub)

theta_sklearn = np.array([
    lin_reg_2f.intercept_,
    lin_reg_2f.coef_[0],
    lin_reg_2f.coef_[1]
])

# Step 3: Compute absolute difference and percentage difference
abs_diff = np.abs(theta_numpy - theta_sklearn)
percent_diff = (abs_diff / np.abs(theta_sklearn)) * 100

# Step 4: Display side-by-side comparison table
param_names = ['Intercept (theta_0)', 'median_income (theta_1)', 'housing_median_age (theta_2)']
comparison_df = pd.DataFrame({
    'Parameter': param_names,
    'NumPy Normal Equation': theta_numpy,
    'Scikit-Learn LinearRegression': theta_sklearn,
    'Absolute Difference': abs_diff,
    'Percentage Difference (%)': [f"{p:.2e}%" for p in percent_diff]
})

print("=" * 85)
print("QUESTION 6: Normal Equation (NumPy) vs. Scikit-Learn Linear Regression")
print("=" * 85)
print(comparison_df)

print(f"\nLinear Model Equation:")
print(f"median_house_value = {theta_numpy[0]:,.4f} + ({theta_numpy[1]:,.4f} * median_income) + ({theta_numpy[2]:,.4f} * housing_median_age)")
print(f"\nConclusion: The percentage difference is on the order of 10^-13%, confirming exact mathematical equivalence within machine floating-point precision.")
fig, ax = plt.subplots(figsize=(11, 7.5))

scatter = ax.scatter(
    housing['longitude'],
    housing['latitude'],
    s=housing['population'] / 100,  # Point size proportional to population
    c=housing['median_house_value'], # Point color proportional to house value
    cmap=plt.get_cmap('jet'),
    alpha=0.4,
    label='District Population'
)

cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04)
cbar.set_label('Median House Value (USD $)', rotation=270, labelpad=20, fontsize=12)

ax.set_title('California Housing Prices & Population Density (1990 Census)', pad=14, fontweight='bold', fontsize=14)
ax.set_xlabel('Longitude (°W)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(10, 5.5))

sns.histplot(
    housing['median_house_value'],
    bins=50,
    kde=True,
    color='#2b5c8f',
    edgecolor='white',
    ax=ax
)

mean_val = housing['median_house_value'].mean()
median_val = housing['median_house_value'].median()

ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_val:,.0f}')
ax.axvline(median_val, color='orange', linestyle='-', linewidth=2, label=f'Median: ${median_val:,.0f}')
ax.axvline(500000, color='darkgreen', linestyle=':', linewidth=2, label='Capped Ceiling: $500,000')

ax.set_title('Frequency Distribution of Median House Value', pad=12, fontweight='bold', fontsize=14)
ax.set_xlabel('Median House Value (USD $)', fontsize=12)
ax.set_ylabel('Frequency (Districts)', fontsize=12)
ax.legend(frameon=True)
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(8.5, 7))

# Scatter plot of actual vs predicted
ax.scatter(
    y_test,
    final_predictions,
    alpha=0.3,
    color='#1f77b4',
    edgecolors='none',
    s=25,
    label='Test Set Districts'
)

# Reference diagonal line for perfect predictions: y = x
min_val = min(y_test.min(), final_predictions.min())
max_val = max(y_test.max(), final_predictions.max())
ax.plot([0, 550000], [0, 550000], 'r--', linewidth=2.2, label='Perfect Prediction (y = x)')

# Annotate summary performance metrics box
metrics_text = (
    f"Test RMSE: ${test_rmse:,.2f}\n"
    f"Test MAE : ${test_mae:,.2f}\n"
    f"Test R²  : {test_r2:.4f}"
)
ax.text(
    0.05, 0.88,
    metrics_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.9, edgecolor='gray')
)

ax.set_title('Tuned Random Forest: Predicted vs. Actual House Values (Test Set)', pad=12, fontweight='bold', fontsize=13)
ax.set_xlabel('Actual Median House Value (USD $)', fontsize=12)
ax.set_ylabel('Predicted Median House Value (USD $)', fontsize=12)
ax.set_xlim(0, 550000)
ax.set_ylim(0, 550000)
ax.legend(loc='lower right', frameon=True)
plt.tight_layout()
plt.show()
# Extract feature importances from the best Random Forest model
feature_importances = final_model.feature_importances_

# Combine with all 13 feature names
importance_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=True)

# Select top 10 features
top10_importance = importance_df.tail(10)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    top10_importance['Feature'],
    top10_importance['Importance'],
    color='#3470a3',
    edgecolor='black',
    alpha=0.85
)

# Annotate importance percentages on the bars
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f} ({width*100:.2f}%)",
        va='center',
        fontsize=10,
        fontweight='bold',
        color='#1c3d5a'
    )

ax.set_title('Top-10 Feature Importances from Tuned Random Forest Regressor', pad=12, fontweight='bold', fontsize=14)
ax.set_xlabel('Relative Feature Importance (Mean Decrease in Impurity)', fontsize=12)
ax.set_xlim(0, top10_importance['Importance'].max() * 1.22)
plt.tight_layout()
plt.show()

print("Top-10 Feature Importances Ranked Table:")
print(top10_importance.sort_values(by='Importance', ascending=False).reset_index(drop=True))
