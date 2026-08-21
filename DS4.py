# ============================================================
# Experiment 4: Statistical Hypothesis Testing Using Python
# Titanic Dataset
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats


# ============================================================
# MAIN EXPERIMENT
# Independent Sample t-test using AGE
# ============================================================

print("=" * 70)
print("EXPERIMENT 4: STATISTICAL HYPOTHESIS TESTING")
print("Titanic Dataset - Independent Sample t-test")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------

df = sns.load_dataset("titanic")

print("\n1. DATASET")
print("-" * 70)
print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# ------------------------------------------------------------
# 2. Select Required Variables
# ------------------------------------------------------------

data = df[["age", "survived"]]

# ------------------------------------------------------------
# 3. Remove Missing Values
# ------------------------------------------------------------

data = data.dropna()

print("\n2. DATA CLEANING")
print("-" * 70)
print("Missing values removed successfully")
print("Number of observations:", len(data))

# ------------------------------------------------------------
# 4. Separate Groups
# ------------------------------------------------------------

survived = data[data["survived"] == 1]["age"]
not_survived = data[data["survived"] == 0]["age"]

print("\n3. GROUPS")
print("-" * 70)
print("Number of survived passengers:", len(survived))
print("Number of non-survived passengers:", len(not_survived))

# ------------------------------------------------------------
# 5. Calculate Mean Age
# ------------------------------------------------------------

mean_survived = survived.mean()
mean_not_survived = not_survived.mean()

print("\nMean Age:")
print("Survived passengers     :", round(mean_survived, 2))
print("Non-survived passengers:", round(mean_not_survived, 2))

# ------------------------------------------------------------
# 6. Hypotheses
# ------------------------------------------------------------

print("\n4. HYPOTHESES")
print("-" * 70)
print("H0: Mean age of survived and non-survived passengers is equal.")
print("H1: Mean age of survived and non-survived passengers is different.")

# ------------------------------------------------------------
# 7. Independent Sample t-test
# ------------------------------------------------------------

t_stat, p_value = stats.ttest_ind(
    survived,
    not_survived,
    equal_var=False
)

# ------------------------------------------------------------
# 8. Display Results
# ------------------------------------------------------------

alpha = 0.05

print("\n5. INDEPENDENT SAMPLE T-TEST")
print("-" * 70)
print("t-statistic:", round(t_stat, 4))
print("p-value    :", round(p_value, 4))
print("Significance level:", alpha)

if p_value < alpha:
    print("Decision: Reject H0")
    print("Conclusion: There is a statistically significant")
    print("difference in the mean age of the two groups.")
else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: There is no statistically significant")
    print("difference in the mean age of the two groups.")


# ============================================================
# EXERCISE 1
# Perform t-test using FARE instead of AGE
# ============================================================

print("\n\n" + "=" * 70)
print("EXERCISE 1: T-TEST USING FARE")
print("=" * 70)

fare_data = df[["fare", "survived"]].dropna()

survived_fare = fare_data[fare_data["survived"] == 1]["fare"]
not_survived_fare = fare_data[fare_data["survived"] == 0]["fare"]

mean_survived_fare = survived_fare.mean()
mean_not_survived_fare = not_survived_fare.mean()

print("\nMean Fare:")
print("Survived passengers     :", round(mean_survived_fare, 2))
print("Non-survived passengers:", round(mean_not_survived_fare, 2))

print("\nHypotheses:")
print("H0: Mean fare of survived and non-survived passengers is equal.")
print("H1: Mean fare of survived and non-survived passengers is different.")

fare_t_stat, fare_p_value = stats.ttest_ind(
    survived_fare,
    not_survived_fare,
    equal_var=False
)

print("\nT-test Results:")
print("t-statistic:", round(fare_t_stat, 4))
print("p-value    :", round(fare_p_value, 4))

if fare_p_value < 0.05:
    print("Decision: Reject H0")
    print("Conclusion: There is a statistically significant")
    print("difference in the mean fare between the two groups.")
else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: There is no statistically significant")
    print("difference in the mean fare between the two groups.")


# ============================================================
# EXERCISE 2
# Change significance level from 0.05 to 0.01
# ============================================================

print("\n\n" + "=" * 70)
print("EXERCISE 2: SIGNIFICANCE LEVEL = 0.01")
print("=" * 70)

alpha_01 = 0.01

print("\nAge t-test:")
print("t-statistic:", round(t_stat, 4))
print("p-value    :", round(p_value, 4))
print("Significance level:", alpha_01)

if p_value < alpha_01:
    print("Decision: Reject H0")
    print("Conclusion: The difference is statistically significant")
    print("at the 0.01 significance level.")
else:
    print("Decision: Fail to Reject H0")
    print("Conclusion: The difference is NOT statistically significant")
    print("at the 0.01 significance level.")

print("\nComparison:")
print("At alpha = 0.05:", end=" ")

if p_value < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")

print("At alpha = 0.01:", end=" ")

if p_value < 0.01:
    print("Reject H0")
else:
    print("Fail to Reject H0")


# ============================================================
# EXERCISE 3
# Interpret the obtained p-value
# ============================================================

print("\n\n" + "=" * 70)
print("EXERCISE 3: INTERPRETATION OF P-VALUE")
print("=" * 70)

print("\nObtained p-value:", round(p_value, 4))

if p_value < 0.05:
    print("Since p-value < 0.05, reject H0.")
    print("The difference in mean age is statistically significant.")
else:
    print("Since p-value >= 0.05, fail to reject H0.")
    print("There is insufficient evidence of a significant difference")


# ============================================================
# EXERCISE 4
# Chi-Square Test
# Gender vs Survival
# ============================================================

print("\n\n" + "=" * 70)
print("EXERCISE 4: CHI-SQUARE TEST")
print("=" * 70)

chi_data = df[["sex", "survived"]].dropna()

# Create contingency table
contingency_table = pd.crosstab(
    chi_data["sex"],
    chi_data["survived"]
)

print("\nContingency Table:")
print(contingency_table)

print("\nHypotheses:")
print("H0: Gender and survival are independent.")
print("H1: Gender and survival are associated.")

# Perform Chi-Square test
chi2, chi_p_value, degrees_freedom, expected = stats.chi2_contingency(
    contingency_table
)

print("\nChi-Square Test Results:")
print("Chi-Square statistic:", round(chi2, 4))
print("p-value            :", round(chi_p_value, 4))
print("Degrees of freedom :", degrees_freedom)

if chi_p_value < 0.05:
    print("\nDecision: Reject H0")
    print("Conclusion: There is a statistically significant")
    print("association between gender and survival.")
else:
    print("\nDecision: Fail to Reject H0")
    print("Conclusion: There is no statistically significant")
    print("association between gender and survival.")


# ============================================================
# EXERCISE 5
# State H0 and H1 for another variable
# Example: Fare
# ============================================================

print("\n\n" + "=" * 70)
print("EXERCISE 5: HYPOTHESES FOR ANOTHER VARIABLE")
print("=" * 70)

print("\nVariable selected: Fare")

print("\nNull Hypothesis (H0):")
print("The mean fare of survived and non-survived passengers is equal.")

print("\nAlternative Hypothesis (H1):")
print("The mean fare of survived and non-survived passengers is different.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nMain Experiment - Age:")
print("t-statistic:", round(t_stat, 4))
print("p-value    :", round(p_value, 4))

print("\nExercise 1 - Fare:")
print("t-statistic:", round(fare_t_stat, 4))
print("p-value    :", round(fare_p_value, 4))

print("\nExercise 2 - Age at alpha = 0.01:")
if p_value < 0.01:
    print("Reject H0")
else:
    print("Fail to Reject H0")

print("\nExercise 4 - Chi-Square:")
print("Chi-Square statistic:", round(chi2, 4))
print("p-value:", round(chi_p_value, 4))

print("\n" + "=" * 70)
print("EXPERIMENT 4 COMPLETED SUCCESSFULLY")
print("=" * 70)