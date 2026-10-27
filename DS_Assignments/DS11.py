
# ============================================================
# EXPERIMENT 11: STREAM DATA SAMPLING AND FILTERING USING PYTHON
# Mapped CO(s): CO3
# ============================================================

import random
import numpy as np
import pandas as pd

# Set seed for reproducible results
random.seed(42)


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("\n" + "=" * 65)
print("MAIN EXPERIMENT: STREAM DATA SAMPLING AND FILTERING")
print("=" * 65)

# Generate a simulated data stream
stream_size = 20
data_stream = [random.randint(1, 100) for _ in range(stream_size)]

print("\n1. ORIGINAL DATA STREAM")
print(data_stream)

# Random sampling
sample_size = 5
sampled_data = random.sample(data_stream, sample_size)

print("\n2. RANDOMLY SAMPLED DATA (5 values)")
print(sampled_data)

# Filter values greater than 50
filtered_data = [value for value in data_stream if value > 50]

print("\n3. FILTERED DATA (Values > 50)")
print(filtered_data)

# Analyze results
print("\n4. DATA ANALYSIS")
print("Original stream size:", len(data_stream))
print("Sample size:", len(sampled_data))
print("Number of filtered values:", len(filtered_data))
print("Original stream mean:", round(np.mean(data_stream), 2))
print("Sample mean:", round(np.mean(sampled_data), 2))
print("Filtered mean:",
      round(np.mean(filtered_data), 2) if filtered_data else "No values")

# Display a summary table
df = pd.DataFrame({
    "Original Stream": pd.Series(data_stream),
    "Sampled Data": pd.Series(sampled_data),
    "Filtered Data (>50)": pd.Series(filtered_data)
})

print("\n5. SUMMARY TABLE")
print(df.to_string(index=False))


# ============================================================
# EXERCISE 1: CHANGE THE SAMPLE SIZE
# ============================================================

print("\n" + "=" * 65)
print("EXERCISE 1: CHANGE THE SAMPLE SIZE")
print("=" * 65)

for sample_size in [3, 5, 10, 15]:
    sampled = random.sample(data_stream, sample_size)
    print(f"\nSample size = {sample_size}")
    print("Sampled values:", sampled)
    print("Number of sampled values:", len(sampled))


# ============================================================
# EXERCISE 2: FILTER VALUES GREATER THAN 80
# ============================================================

print("\n" + "=" * 65)
print("EXERCISE 2: FILTER VALUES GREATER THAN 80")
print("=" * 65)

filtered_above_80 = [value for value in data_stream if value > 80]

print("Original data stream:", data_stream)
print("Values greater than 80:", filtered_above_80)
print("Number of values greater than 80:", len(filtered_above_80))


# ============================================================
# EXERCISE 3: GENERATE A LARGER DATA STREAM
# ============================================================

print("\n" + "=" * 65)
print("EXERCISE 3: LARGER DATA STREAM")
print("=" * 65)

large_stream = [random.randint(1, 100) for _ in range(100)]

# Sample 20 values from the larger stream
large_sample = random.sample(large_stream, 20)

# Filter values greater than 50
large_filtered = [value for value in large_stream if value > 50]

print("\nLarge stream (100 values):")
print(large_stream)

print("\nSampled data (20 values):")
print(large_sample)

print("\nFiltered data (values > 50):")
print(large_filtered)

print("\nComparison:")
print("Original stream size:", len(data_stream))
print("Large stream size:", len(large_stream))
print("Original sample size: 5")
print("Large stream sample size:", len(large_sample))
print("Filtered values in original stream:", len(filtered_data))
print("Filtered values in large stream:", len(large_filtered))
print("Mean of large stream:", round(np.mean(large_stream), 2))


# ============================================================
# EXERCISE 4: SAMPLING VERSUS FILTERING
# ============================================================

print("\n" + "=" * 65)
print("EXERCISE 4: SAMPLING VERSUS FILTERING")
print("=" * 65)

comparison = pd.DataFrame({
    "Feature": [
        "Purpose",
        "Selection method",
        "Output size",
        "Example"
    ],
    "Sampling": [
        "Select a subset of data",
        "Random selection",
        "Usually a chosen sample size",
        "Select 5 random values"
    ],
    "Filtering": [
        "Select data satisfying a condition",
        "Apply a specified condition",
        "Depends on the data",
        "Select values greater than 50"
    ]
})

print(comparison.to_string(index=False))


# ============================================================
# EXERCISE 5: APPLY SAMPLING AND FILTERING TO TEMPERATURE DATA
# ============================================================

print("\n" + "=" * 65)
print("EXERCISE 5: TEMPERATURE SENSOR DATA")
print("=" * 65)

# Use a separate seed for reproducible temperature readings
temperature_rng = random.Random(10)

# Generate 20 simulated temperature readings
temperatures = [
    round(temperature_rng.uniform(20, 100), 2)
    for _ in range(20)
]

# Randomly sample 5 temperature readings
sampled_temperatures = temperature_rng.sample(temperatures, 5)

# Filter temperatures greater than 35
filtered_temperatures = [
    temp for temp in temperatures if temp > 35
]

print("\nOriginal temperature readings:")
print(temperatures)

print("\nSampled temperature readings:")
print(sampled_temperatures)

print("\nTemperatures greater than 35:")
print(filtered_temperatures)

print("\nTemperature data analysis:")
print("Total readings:", len(temperatures))
print("Sampled readings:", len(sampled_temperatures))
print("Filtered readings:", len(filtered_temperatures))
print("Average temperature:", round(np.mean(temperatures), 2))
print("Average sampled temperature:",
      round(np.mean(sampled_temperatures), 2))
print("Average filtered temperature:",
      round(np.mean(filtered_temperatures), 2)
      if filtered_temperatures else "No values")

# Display temperature data in a table
temperature_df = pd.DataFrame({
    "Original Temperature": pd.Series(temperatures),
    "Sampled Temperature": pd.Series(sampled_temperatures),
    "Filtered Temperature (>35)": pd.Series(filtered_temperatures)
})

print("\nTemperature Summary Table:")
print(temperature_df.to_string(index=False))


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT 11 COMPLETED SUCCESSFULLY")
print("=" * 65)
print("All five exercises have been executed.")
