
# EXPERIMENT 14: RECOMMENDATION SYSTEM DEVELOPMENT
# UBCF, IBCF, MATRIX FACTORIZATION AND SVD
# Includes all 5 exercises/assignments

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error

sns.set_theme(style="whitegrid")
np.random.seed(42)

print("=" * 85)
print("EXPERIMENT 14: RECOMMENDATION SYSTEM USING COLLABORATIVE FILTERING")
print("=" * 85)

# ---------------------------------------------------------
# 1. CREATE SAMPLE USER-ITEM RATING DATASET
# ---------------------------------------------------------

ratings_data = [
    ("U1", "Movie_A", 5),
    ("U1", "Movie_B", 4),
    ("U1", "Movie_C", 1),
    ("U1", "Movie_D", 2),
    ("U1", "Movie_E", 5),

    ("U2", "Movie_A", 4),
    ("U2", "Movie_B", 5),
    ("U2", "Movie_C", 1),
    ("U2", "Movie_D", 2),
    ("U2", "Movie_E", 4),

    ("U3", "Movie_A", 1),
    ("U3", "Movie_B", 2),
    ("U3", "Movie_C", 5),
    ("U3", "Movie_D", 4),
    ("U3", "Movie_E", 1),

    ("U4", "Movie_A", 2),
    ("U4", "Movie_B", 1),
    ("U4", "Movie_C", 4),
    ("U4", "Movie_D", 5),
    ("U4", "Movie_E", 2),

    ("U5", "Movie_A", 5),
    ("U5", "Movie_B", 4),
    ("U5", "Movie_C", 2),
    ("U5", "Movie_D", 1),
    ("U5", "Movie_E", 4),

    ("U6", "Movie_A", 3),
    ("U6", "Movie_B", 2),
    ("U6", "Movie_C", 4),
    ("U6", "Movie_D", 5),
    ("U6", "Movie_E", 1),

    ("U7", "Movie_A", 4),
    ("U7", "Movie_B", 5),
    ("U7", "Movie_C", 2),
    ("U7", "Movie_D", 1),
    ("U7", "Movie_E", 5),
]

ratings = pd.DataFrame(
    ratings_data, columns=["User", "Item", "Rating"]
)

print("\n1. ORIGINAL RATING DATASET")
print(ratings.to_string(index=False))
print("\nDataset loaded successfully.")
print("Total ratings:", len(ratings))
print("Number of users:", ratings["User"].nunique())
print("Number of movies:", ratings["Item"].nunique())

# ---------------------------------------------------------
# 2. CREATE USER-ITEM RATING MATRIX
# ---------------------------------------------------------

rating_matrix = ratings.pivot(
    index="User", columns="Item", values="Rating"
)

print("\n2. USER-ITEM RATING MATRIX")
print(rating_matrix.to_string())
print("\nUser-item matrix created successfully.")

# Fill missing values with zero for similarity calculations.
# Zero means unrated, not a genuine rating.
R = rating_matrix.fillna(0)

# ---------------------------------------------------------
# 3. USER-BASED COLLABORATIVE FILTERING (UBCF)
# ---------------------------------------------------------

user_similarity = pd.DataFrame(
    cosine_similarity(R),
    index=R.index,
    columns=R.index
)

print("\n3. USER SIMILARITY MATRIX")
print(user_similarity.round(2).to_string())

def recommend_ubcf(user, top_n=3):
    """Predict unrated movie scores using similar users."""
    similarities = user_similarity.loc[user].drop(user)
    predictions = {}

    for item in rating_matrix.columns:
        if pd.notna(rating_matrix.loc[user, item]):
            continue

        other_ratings = rating_matrix[item].dropna()
        common_users = similarities.index.intersection(
            other_ratings.index
        )

        weights = similarities.loc[common_users]
        values = other_ratings.loc[common_users]

        denominator = weights.abs().sum()

        if denominator > 0:
            predictions[item] = (
                (weights * values).sum() / denominator
            )

    return pd.Series(predictions, dtype=float).sort_values(
        ascending=False
    ).head(top_n)

print("\n4. USER-BASED COLLABORATIVE FILTERING")
print("Similar users to U1:")
print(user_similarity.loc["U1"].drop("U1").sort_values(
    ascending=False
).round(3).to_string())

print("\nUBCF recommendations for U1:")
ubcf_result = recommend_ubcf("U1")
if ubcf_result.empty:
    print("No unrated movies available for recommendation.")
else:
    print(ubcf_result.round(2).to_string())

# ---------------------------------------------------------
# 5. ITEM-BASED COLLABORATIVE FILTERING (IBCF)
# ---------------------------------------------------------

item_similarity = pd.DataFrame(
    cosine_similarity(R.T),
    index=R.columns,
    columns=R.columns
)

print("\n5. ITEM SIMILARITY MATRIX")
print(item_similarity.round(2).to_string())

def recommend_ibcf(user, top_n=3):
    """Predict scores using the user's ratings of similar movies."""
    predictions = {}

    for item in rating_matrix.columns:
        if pd.notna(rating_matrix.loc[user, item]):
            continue

        rated_items = rating_matrix.loc[user].dropna()
        similarities = item_similarity.loc[item, rated_items.index]

        denominator = similarities.abs().sum()

        if denominator > 0:
            predictions[item] = (
                similarities * rated_items
            ).sum() / denominator

    return pd.Series(predictions, dtype=float).sort_values(
        ascending=False
    ).head(top_n)

print("\n6. ITEM-BASED COLLABORATIVE FILTERING")
print("Movies similar to Movie_A:")
print(item_similarity["Movie_A"].drop("Movie_A").sort_values(
    ascending=False
).round(3).to_string())

print("\nIBCF recommendations for U1:")
ibcf_result = recommend_ibcf("U1")
if ibcf_result.empty:
    print("No unrated movies available for recommendation.")
else:
    print(ibcf_result.round(2).to_string())

# ---------------------------------------------------------
# 7. MATRIX FACTORIZATION AND TRUNCATED SVD
# ---------------------------------------------------------

# This sample is deliberately small and dense.
# SVD is applied to the zero-filled matrix for demonstration.
n_components = 3

svd = TruncatedSVD(
    n_components=n_components,
    random_state=42
)

user_factors = svd.fit_transform(R)
item_factors = svd.components_

reconstructed = user_factors @ item_factors

predicted_matrix = pd.DataFrame(
    np.clip(reconstructed, 1, 5),
    index=R.index,
    columns=R.columns
)

print("\n7. MATRIX FACTORIZATION USING SVD")
print("Number of SVD components:", n_components)

print("\nReduced user feature matrix:")
print(pd.DataFrame(
    user_factors,
    index=R.index,
    columns=[f"Feature_{i+1}" for i in range(n_components)]
).round(2).to_string())

print("\nItem latent feature matrix:")
print(pd.DataFrame(
    item_factors.T,
    index=R.columns,
    columns=[f"Feature_{i+1}" for i in range(n_components)]
).round(2).to_string())

print("\nSVD reconstructed and predicted rating matrix:")
print(predicted_matrix.round(2).to_string())

print("\nExplained variance ratio:",
      round(svd.explained_variance_ratio_.sum(), 4))

# SVD recommendations for a user
def recommend_svd(user, top_n=3, prediction_matrix=None):
    if prediction_matrix is None:
        prediction_matrix = predicted_matrix

    unrated = rating_matrix.loc[user][
        rating_matrix.loc[user].isna()
    ].index

    return prediction_matrix.loc[user, unrated].sort_values(
        ascending=False
    ).head(top_n)

# ---------------------------------------------------------
# 8. DISPLAY RECOMMENDATIONS FROM ALL METHODS
# ---------------------------------------------------------

print("\n8. RECOMMENDATION COMPARISON FOR U1")

print("\nUBCF:")
print(recommend_ubcf("U1").round(2).to_string())

print("\nIBCF:")
print(recommend_ibcf("U1").round(2).to_string())

print("\nSVD:")
print(recommend_svd("U1").round(2).to_string())

# ---------------------------------------------------------
# 9. VISUALIZATION 1: USER-ITEM RATING HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))
sns.heatmap(
    rating_matrix,
    annot=True,
    cmap="YlGnBu",
    vmin=1,
    vmax=5,
    linewidths=0.5
)
plt.title("User-Item Rating Matrix")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.tight_layout()
plt.savefig("DS14_rating_matrix.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 10. VISUALIZATION 2: USER SIMILARITY HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
sns.heatmap(
    user_similarity,
    annot=True,
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    linewidths=0.5
)
plt.title("User-Based Similarity Matrix")
plt.tight_layout()
plt.savefig("DS14_user_similarity.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 11. VISUALIZATION 3: ITEM SIMILARITY HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))
sns.heatmap(
    item_similarity,
    annot=True,
    cmap="viridis",
    vmin=0,
    vmax=1,
    linewidths=0.5
)
plt.title("Item-Based Similarity Matrix")
plt.tight_layout()
plt.savefig("DS14_item_similarity.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 12. VISUALIZATION 4: PREDICTED RATINGS HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))
sns.heatmap(
    predicted_matrix,
    annot=True,
    cmap="YlOrRd",
    vmin=1,
    vmax=5,
    linewidths=0.5
)
plt.title("SVD Predicted Rating Matrix")
plt.xlabel("Movies")
plt.ylabel("Users")
plt.tight_layout()
plt.savefig("DS14_predicted_ratings.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 13. VISUALIZATION 5: SVD EXPLAINED VARIANCE
# ---------------------------------------------------------

plt.figure(figsize=(7, 4))
components = np.arange(
    1, len(svd.explained_variance_ratio_) + 1
)

plt.bar(components, svd.explained_variance_ratio_)
plt.xlabel("SVD Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Explained Variance by SVD Components")
plt.xticks(components)
plt.tight_layout()
plt.savefig("DS14_svd_variance.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 14. EXERCISE 1: ADD A NEW USER AND RECOMMEND
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("EXERCISE 1: ADD A NEW USER AND GENERATE RECOMMENDATIONS")
print("=" * 85)

new_user_ratings = pd.Series({
    "Movie_A": 5,
    "Movie_B": 4,
    "Movie_C": np.nan,
    "Movie_D": np.nan,
    "Movie_E": 5
}, name="U8")

rating_matrix_extended = pd.concat([
    rating_matrix,
    new_user_ratings.to_frame().T
])

print("\nUpdated user-item matrix:")
print(rating_matrix_extended.to_string())

# Recalculate similarity including the new user.
R_extended = rating_matrix_extended.fillna(0)

user_similarity_extended = pd.DataFrame(
    cosine_similarity(R_extended),
    index=R_extended.index,
    columns=R_extended.index
)

print("\nUser similarity matrix with U8:")
print(user_similarity_extended.round(2).to_string())

# Generate UBCF recommendations for U8.
def recommend_new_user(user, top_n=3):
    similarities = user_similarity_extended.loc[user].drop(user)
    predictions = {}

    for item in rating_matrix_extended.columns:
        if pd.notna(rating_matrix_extended.loc[user, item]):
            continue

        other_ratings = rating_matrix_extended[item].dropna()
        common_users = similarities.index.intersection(
            other_ratings.index
        )

        weights = similarities.loc[common_users]
        values = other_ratings.loc[common_users]
        denominator = weights.abs().sum()

        if denominator > 0:
            predictions[item] = (
                (weights * values).sum() / denominator
            )

    return pd.Series(predictions, dtype=float).sort_values(
        ascending=False
    ).head(top_n)

print("\nRecommendations for new user U8:")
print(recommend_new_user("U8").round(2).to_string())


# ---------------------------------------------------------
# EXERCISE 2: COMPARE UBCF AND IBCF
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("EXERCISE 2: COMPARE UBCF AND IBCF")
print("=" * 85)

# Use U8, the new user who has unrated movies.
# UBCF and IBCF must use the same user-item matrix.

user = "U8"

# Create a temporary matrix including U8
matrix_ex2 = rating_matrix_extended.copy()
R_ex2 = matrix_ex2.fillna(0)

# Recalculate user similarity
user_sim_ex2 = pd.DataFrame(
    cosine_similarity(R_ex2),
    index=R_ex2.index,
    columns=R_ex2.index
)

# Recalculate item similarity
item_sim_ex2 = pd.DataFrame(
    cosine_similarity(R_ex2.T),
    index=R_ex2.columns,
    columns=R_ex2.columns
)

# UBCF recommendations
def ubcf_ex2(user, top_n=3):
    similarities = user_sim_ex2.loc[user].drop(user)
    predictions = {}

    for item in matrix_ex2.columns:
        if pd.notna(matrix_ex2.loc[user, item]):
            continue

        other_ratings = matrix_ex2[item].dropna()
        common_users = similarities.index.intersection(
            other_ratings.index
        )

        weights = similarities.loc[common_users]
        values = other_ratings.loc[common_users]
        denominator = weights.abs().sum()

        if denominator > 0:
            predictions[item] = (
                (weights * values).sum() / denominator
            )

    return pd.Series(predictions, dtype=float).sort_values(
        ascending=False
    ).head(top_n)


# IBCF recommendations
def ibcf_ex2(user, top_n=3):
    predictions = {}

    for item in matrix_ex2.columns:
        if pd.notna(matrix_ex2.loc[user, item]):
            continue

        rated_items = matrix_ex2.loc[user].dropna()
        similarities = item_sim_ex2.loc[
            item, rated_items.index
        ]

        denominator = similarities.abs().sum()

        if denominator > 0:
            predictions[item] = (
                similarities * rated_items
            ).sum() / denominator

    return pd.Series(predictions, dtype=float).sort_values(
        ascending=False
    ).head(top_n)


ubcf_result_ex2 = ubcf_ex2(user)
ibcf_result_ex2 = ibcf_ex2(user)

comparison = pd.concat(
    [
        ubcf_result_ex2.rename("UBCF"),
        ibcf_result_ex2.rename("IBCF")
    ],
    axis=1
)

print(f"\nRecommendation comparison for {user}:")
print(comparison.round(2).to_string())

# Plot only when recommendations are available
if not comparison.empty:
    comparison.plot(kind="bar", figsize=(8, 5))
    plt.title(f"UBCF vs IBCF Recommendation Scores for {user}")
    plt.ylabel("Predicted Rating")
    plt.xlabel("Movie")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("DS14_ubcf_ibcf_comparison.png", dpi=150)
    plt.show()
else:
    print("No recommendations available to plot.")

print("\nObservation:")
print("UBCF recommends items based on similar users.")
print("IBCF recommends items based on similar movies.")
print("Different similarity calculations can produce different scores.")


# ---------------------------------------------------------
# 16. EXERCISE 3: CHANGE NUMBER OF SVD COMPONENTS
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("EXERCISE 3: EFFECT OF DIFFERENT SVD COMPONENTS")
print("=" * 85)

svd_results = []
component_options = [1, 2, 3, 4]

for k in component_options:
    model = TruncatedSVD(
        n_components=k,
        random_state=42
    )

    factors = model.fit_transform(R)
    reconstruction = factors @ model.components_

    # Calculate reconstruction error on the complete,
    # zero-filled matrix (demonstration metric).
    rmse = np.sqrt(np.mean((R.values - reconstruction) ** 2))

    svd_results.append({
        "Components": k,
        "Explained Variance": model.explained_variance_ratio_.sum(),
        "Reconstruction RMSE": rmse
    })

svd_results_df = pd.DataFrame(svd_results)

print("\nSVD component comparison:")
print(svd_results_df.round(4).to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(
    svd_results_df["Components"],
    svd_results_df["Reconstruction RMSE"],
    marker="o"
)
plt.title("Effect of SVD Components on Reconstruction Error")
plt.xlabel("Number of Components")
plt.ylabel("Reconstruction RMSE")
plt.xticks(component_options)
plt.tight_layout()
plt.savefig("DS14_svd_components.png", dpi=150)
plt.show()

print("\nObservation:")
print("More components generally retain more information from the matrix.")
print("Reconstruction error generally decreases as components increase.")
print("Too many components increase model complexity.")

# ---------------------------------------------------------
# 17. EXERCISE 4: VISUALIZE PREDICTED RATINGS USING HEATMAP
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("EXERCISE 4: PREDICTED RATINGS HEATMAP")
print("=" * 85)

print("\nPredicted ratings:")
print(predicted_matrix.round(2).to_string())

print("\nHeatmap saved as DS14_predicted_ratings.png.")
print("The heatmap uses color intensity to represent predicted ratings.")

# ---------------------------------------------------------
# 18. EXERCISE 5: COLLABORATIVE VS CONTENT-BASED FILTERING
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("EXERCISE 5: COLLABORATIVE FILTERING VS CONTENT-BASED FILTERING")
print("=" * 85)

advantages = [
    ("1", "Uses collective user behavior",
     "Learns from ratings and interactions of many users."),
    ("2", "Discovers hidden preferences",
     "Can identify patterns not explicitly described by item features."),
    ("3", "No detailed item descriptions required",
     "Can work using user-item interaction data alone."),
    ("4", "Serendipitous recommendations",
     "Can suggest unexpected items that similar users enjoyed."),
    ("5", "Limitations",
     "Cold-start problems and sparse rating data can reduce performance.")
]

for number, heading, description in advantages:
    print(f"{number}. {heading}: {description}")

print("\nCONTENT-BASED FILTERING:")
print("Recommends items based on their features and the user's past preferences.")
print("It needs item attributes such as genre, description, or category.")

# ---------------------------------------------------------
# 19. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 85)
print("FINAL SUMMARY")
print("=" * 85)

print("UBCF: Recommendations based on similar users.")
print("IBCF: Recommendations based on similar items.")
print("Matrix Factorization/SVD: Represents users and items using latent features.")
print("Visualization: Rating, similarity, prediction, and SVD comparison plots.")
print("All five exercises completed.")
print("\nGenerated image files:")
print("1. DS14_rating_matrix.png")
print("2. DS14_user_similarity.png")
print("3. DS14_item_similarity.png")
print("4. DS14_predicted_ratings.png")
print("5. DS14_svd_variance.png")
print("6. DS14_ubcf_ibcf_comparison.png")
print("7. DS14_svd_components.png")

print("\nEXPERIMENT 14 COMPLETED SUCCESSFULLY.")
