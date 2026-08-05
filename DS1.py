# ============================================
# Experiment 1
# Frequency Analysis and Probability
# Using Titanic Dataset
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# Load Titanic Dataset
# --------------------------------------------
df = sns.load_dataset("titanic")

print("First Five Records")
print(df.head())

print("\nDataset Shape:", df.shape)

# ============================================
# PART I : FREQUENCY ANALYSIS
# ============================================

print("\n========== FREQUENCY ANALYSIS ==========")

# Absolute Frequency
absolute_frequency = df['class'].value_counts()

# Relative Frequency
relative_frequency = df['class'].value_counts(normalize=True) * 100

# Cumulative Frequency
cumulative_frequency = absolute_frequency.cumsum()

# Frequency Table
frequency_table = pd.DataFrame({
    'Absolute Frequency': absolute_frequency,
    'Relative Frequency (%)': relative_frequency.round(2),
    'Cumulative Frequency': cumulative_frequency
})

print("\nFrequency Table of Passenger Class")
print(frequency_table)

# ============================================
# PART II : PROBABILITY ANALYSIS
# ============================================

print("\n========== PROBABILITY ANALYSIS ==========")

# Contingency Table
contingency_table = pd.crosstab(df['sex'], df['survived'])

print("\nContingency Table (Sex vs Survived)")
print(contingency_table)

# Joint Probability
joint_probability = contingency_table / len(df)

print("\nJoint Probability")
print(joint_probability)

# Marginal Probability
print("\nMarginal Probability")

print("\nProbability of Gender")
print(df['sex'].value_counts(normalize=True))

print("\nProbability of Survival")
print(df['survived'].value_counts(normalize=True))

# Conditional Probability

print("\nConditional Probability P(Survived | Sex)")

conditional_probability = contingency_table.div(
    contingency_table.sum(axis=1),
    axis=0
)

print(conditional_probability)

# ============================================
# PART III : CORRELATION ANALYSIS
# ============================================

print("\n========== CORRELATION ANALYSIS ==========")

# Select Numerical Columns
corr_data = df[['age', 'fare']].dropna()

# Pearson Correlation
correlation = corr_data.corr(method='pearson')

print("\nPearson Correlation Matrix")
print(correlation)

# ============================================
# HEATMAP
# ============================================

plt.figure(figsize=(5,4))
sns.heatmap(correlation,
            annot=True,
            cmap="coolwarm",
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# ============================================
# SCATTER PLOT
# ============================================

plt.figure(figsize=(7,5))
sns.scatterplot(data=corr_data,
                x='age',
                y='fare')

plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# ============================================
# STACKED BAR CHART
# ============================================

stacked = pd.crosstab(df['class'], df['survived'])

stacked.plot(kind='bar',
             stacked=True,
             figsize=(7,5))

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(["Did Not Survive", "Survived"])
plt.show()

# ============================================
# EXERCISES
# ============================================

print("\n========== EXERCISES ==========")

# Frequency Table of Embarked

embarked_frequency = pd.DataFrame({
    "Absolute": df['embarked'].value_counts(),
    "Relative (%)": (df['embarked'].value_counts(normalize=True)*100).round(2),
    "Cumulative": df['embarked'].value_counts().cumsum()
})

print("\nFrequency Table of Embarked")
print(embarked_frequency)

# Probability of Survival

prob_survival = (df['survived'] == 1).mean()

print("\nProbability that a Passenger Survived =", round(prob_survival,4))

# Pearson Correlation

pearson = df[['age','fare']].dropna().corr().iloc[0,1]

print("\nPearson Correlation (Age vs Fare) =", round(pearson,4))

# Correlation Interpretation

print("\nInterpretation:")

if pearson > 0:
    print("Positive correlation exists between Age and Fare.")
elif pearson < 0:
    print("Negative correlation exists between Age and Fare.")
else:
    print("No linear correlation exists.")