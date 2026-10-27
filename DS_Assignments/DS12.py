
# Experiment 12: Distinct Element Estimation and Stream Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

print("=" * 65)
print("EXPERIMENT 12: DISTINCT ELEMENT ESTIMATION AND STREAM CLUSTERING")
print("=" * 65)

# Step 1: Generate a simulated data stream
np.random.seed(42)

stream = np.concatenate([
    np.random.normal(10, 2, 100),
    np.random.normal(25, 3, 100),
    np.random.normal(40, 2, 100)
])

# Convert values into a DataFrame
df = pd.DataFrame({"Value": stream})

print("\n1. ORIGINAL DATA STREAM")
print(df.head(10))
print("\nTotal elements in stream:", len(stream))

# Step 2: Estimate distinct elements using exact counting
distinct_elements = np.unique(stream)

print("\n2. DISTINCT ELEMENT ESTIMATION")
print("Total elements:", len(stream))
print("Number of distinct elements:", len(distinct_elements))
print("First 10 distinct values:", distinct_elements[:10])

# Step 3: Prepare data for clustering
X = stream.reshape(-1, 1)

# Step 4: Apply K-Means clustering with 3 clusters
kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels3 = kmeans3.fit_predict(X)

df["Cluster_3"] = labels3

print("\n3. K-MEANS CLUSTERING (3 CLUSTERS)")
print(df.head(10))
print("\nCluster centers:")
print(np.sort(kmeans3.cluster_centers_.flatten()))

print("\nNumber of elements in each cluster:")
print(df["Cluster_3"].value_counts().sort_index())

# Step 5: Visualize the clusters
plt.figure(figsize=(10, 5))

plt.scatter(
    np.arange(len(stream)),
    stream,
    c=labels3,
    cmap="viridis",
    s=25
)

plt.title("Stream Clustering Using K-Means (3 Clusters)")
plt.xlabel("Stream Element Index")
plt.ylabel("Value")
plt.colorbar(label="Cluster")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# EXERCISE 1
# Generate a larger stream and estimate distinct elements
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXERCISE 1: LARGER DATA STREAM")
print("=" * 65)

large_stream = np.random.randint(0, 10000, size=10000)

print("Total elements in larger stream:", len(large_stream))
print("Exact distinct elements:", len(np.unique(large_stream)))

# --------------------------------------------------
# EXERCISE 2
# Change number of clusters from 3 to 4 and compare
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXERCISE 2: COMPARE 3 AND 4 CLUSTERS")
print("=" * 65)

kmeans4 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels4 = kmeans4.fit_predict(X)

df["Cluster_4"] = labels4

print("Cluster centers for 3 clusters:")
print(np.sort(kmeans3.cluster_centers_.flatten()))

print("Cluster centers for 4 clusters:")
print(np.sort(kmeans4.cluster_centers_.flatten()))

print("\nCluster sizes for 3 clusters:")
print(df["Cluster_3"].value_counts().sort_index())

print("\nCluster sizes for 4 clusters:")
print(df["Cluster_4"].value_counts().sort_index())

# --------------------------------------------------
# EXERCISE 3
# Visualize clusters using different colors
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXERCISE 3: VISUALIZE 4 CLUSTERS")
print("=" * 65)

plt.figure(figsize=(10, 5))

colors = ["red", "blue", "green", "orange"]

for cluster in range(4):
    mask = labels4 == cluster

    plt.scatter(
        np.arange(len(stream))[mask],
        stream[mask],
        color=colors[cluster],
        label=f"Cluster {cluster}",
        s=25
    )

plt.title("Stream Clustering Using K-Means (4 Clusters)")
plt.xlabel("Stream Element Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# EXERCISE 4
# Importance of distinct element estimation
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXERCISE 4: IMPORTANCE OF DISTINCT ESTIMATION")
print("=" * 65)

print("""
1. Counts unique users, devices, or IP addresses.
2. Helps detect repeated and new events in data streams.
3. Supports network traffic monitoring.
4. Helps identify fraud and unusual activity.
5. Reduces the cost of analyzing very large datasets.
6. Approximate algorithms save memory for massive streams.
""")

# --------------------------------------------------
# EXERCISE 5
# Compare exact and approximate counting
# --------------------------------------------------

print("\n" + "=" * 65)
print("EXERCISE 5: EXACT VS APPROXIMATE COUNTING")
print("=" * 65)

exact_count = len(np.unique(large_stream))

# Demonstration of an approximate estimate using a sample.
# This is a simple sampling estimate, not HyperLogLog.
sample = large_stream[::10]
sample_distinct = len(np.unique(sample))
approx_count = sample_distinct * 10

print("Exact distinct count:", exact_count)
print("Sample-based approximate count:", approx_count)
print("Note: The sample estimate may overestimate or underestimate.")

comparison = pd.DataFrame({
    "Method": ["Exact Counting", "Approximate Sampling"],
    "Estimated Distinct Elements": [exact_count, approx_count],
    "Memory Requirement": ["Higher for large streams", "Lower for the sample"]
})

print("\nComparison table:")
print(comparison.to_string(index=False))

# Save results
df.to_csv("experiment12_clustered_data.csv", index=False)
comparison.to_csv("experiment12_counting_comparison.csv", index=False)

print("\nResults saved successfully.")
print("Files:")
print("1. experiment12_clustered_data.csv")
print("2. experiment12_counting_comparison.csv")

print("\nEXPERIMENT 12 COMPLETED SUCCESSFULLY.")
