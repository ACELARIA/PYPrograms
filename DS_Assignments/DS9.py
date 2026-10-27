
# Experiment 9: Text Preprocessing and Feature Scaling Using Python

import re
import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize


# --------------------------------------------------
# STEP 1: DOWNLOAD REQUIRED NLTK RESOURCES
# --------------------------------------------------

print("Checking required NLTK resources...")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

print("NLTK resources ready.")


# --------------------------------------------------
# STEP 2: CREATE THE SAMPLE TEXT DATASET
# --------------------------------------------------

data = {
    "Text": [
        "Machine learning is very useful for data analysis!",
        "Python is a popular programming language.",
        "Data science uses machine learning techniques.",
        "Natural Language Processing analyzes text data.",
        "Python helps in building machine learning models."
    ]
}

df = pd.DataFrame(data)

print("\n" + "=" * 65)
print("ORIGINAL DATASET")
print("=" * 65)
print(df.to_string(index=False))


# --------------------------------------------------
# STEP 3: INITIALIZE PREPROCESSING TOOLS
# --------------------------------------------------

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


# --------------------------------------------------
# STEP 4: TEXT PREPROCESSING USING STEMMING
# --------------------------------------------------

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation, numbers and special characters
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    tokens = text.split()

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Apply stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Return processed text
    return " ".join(tokens)


df["Processed_Text"] = df["Text"].apply(preprocess_text)

print("\n" + "=" * 65)
print("PREPROCESSED DATASET USING STEMMING")
print("=" * 65)
print(df[["Text", "Processed_Text"]].to_string(index=False))


# --------------------------------------------------
# STEP 5: TF-IDF FEATURE EXTRACTION
# --------------------------------------------------

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(df["Processed_Text"])

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("\n" + "=" * 65)
print("TF-IDF FEATURE MATRIX")
print("=" * 65)
print(tfidf_df.round(3).to_string())


# --------------------------------------------------
# STEP 6: L2 NORMALIZATION OF TF-IDF FEATURES
# --------------------------------------------------

# TfidfVectorizer already applies L2 normalization by default.
# This step explicitly applies L2 normalization again.

normalized_matrix = normalize(tfidf_matrix, norm="l2")

normalized_df = pd.DataFrame(
    normalized_matrix.toarray(),
    columns=tfidf.get_feature_names_out()
)

print("\n" + "=" * 65)
print("L2 NORMALIZED TF-IDF MATRIX")
print("=" * 65)
print(normalized_df.round(3).to_string())

print("\nNote: TF-IDF and L2 normalized values are identical")
print("because TfidfVectorizer uses L2 normalization by default.")


# --------------------------------------------------
# STEP 7: MIN-MAX SCALING OF TF-IDF FEATURES
# --------------------------------------------------

minmax_scaler = MinMaxScaler()

minmax_data = minmax_scaler.fit_transform(
    tfidf_matrix.toarray()
)

minmax_df = pd.DataFrame(
    minmax_data,
    columns=tfidf.get_feature_names_out()
)

print("\n" + "=" * 65)
print("MIN-MAX SCALED TF-IDF MATRIX")
print("=" * 65)
print(minmax_df.round(3).to_string())


# --------------------------------------------------
# STEP 8: STANDARDIZATION OF TF-IDF FEATURES
# --------------------------------------------------

standard_scaler = StandardScaler()

standard_data = standard_scaler.fit_transform(
    tfidf_matrix.toarray()
)

standard_df = pd.DataFrame(
    standard_data,
    columns=tfidf.get_feature_names_out()
)

print("\n" + "=" * 65)
print("STANDARDIZED TF-IDF MATRIX")
print("=" * 65)
print(standard_df.round(3).to_string())


# ==================================================
# EXERCISE 1: APPLY COUNTVECTORIZER
# ==================================================

count_vectorizer = CountVectorizer()

count_matrix = count_vectorizer.fit_transform(
    df["Processed_Text"]
)

count_df = pd.DataFrame(
    count_matrix.toarray(),
    columns=count_vectorizer.get_feature_names_out()
)

print("\n" + "=" * 65)
print("EXERCISE 1: COUNTVECTORIZER OUTPUT")
print("=" * 65)
print(count_df.to_string())


# ==================================================
# EXERCISE 2: COMPARE TF-IDF AND COUNTVECTORIZER
# ==================================================

# Both vectorizers use the same preprocessed documents.
# Their vocabulary should therefore be the same.

print("\n" + "=" * 65)
print("EXERCISE 2: TF-IDF VS COUNTVECTORIZER")
print("=" * 65)

comparison_df = pd.DataFrame(
    {
        "Word": tfidf.get_feature_names_out(),
        "Count in Document 1": count_df.iloc[0].values,
        "TF-IDF in Document 1": tfidf_df.iloc[0].values
    }
)

print("\nComparison for the first sentence:")
print(comparison_df.round(3).to_string(index=False))

print("\nCountVectorizer records word occurrence counts.")
print("TF-IDF assigns weights based on word importance.")
print("CountVectorizer produces integer counts.")
print("TF-IDF produces decimal-valued feature weights.")


# ==================================================
# EXERCISE 3: LEMMATIZATION INSTEAD OF STEMMING
# ==================================================

def get_wordnet_pos(tag):
    # Convert NLTK POS tags into WordNet POS tags
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess_lemmatization(text):
    # Lowercase and remove unwanted characters
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    tokens = text.split()

    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    # Assign part-of-speech tags for better lemmatization
    tagged_tokens = nltk.pos_tag(tokens)

    # Convert words to their dictionary forms
    lemmas = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
    ]

    return " ".join(lemmas)


df["Lemmatized_Text"] = df["Text"].apply(
    preprocess_lemmatization
)

print("\n" + "=" * 65)
print("EXERCISE 3: LEMMATIZATION OUTPUT")
print("=" * 65)
print(
    df[["Text", "Lemmatized_Text"]].to_string(index=False)
)

print("\nStemming produces approximate root forms.")
print("Lemmatization produces dictionary forms using linguistic rules.")


# ==================================================
# EXERCISE 4: MIN-MAX SCALING OF ANOTHER DATASET
# ==================================================

numerical_data = np.array([
    [10],
    [20],
    [30],
    [40],
    [50]
])

numerical_minmax_scaler = MinMaxScaler()

numerical_minmax = numerical_minmax_scaler.fit_transform(
    numerical_data
)

exercise4_df = pd.DataFrame({
    "Original Value": numerical_data.flatten(),
    "Min-Max Scaled Value": numerical_minmax.flatten()
})

print("\n" + "=" * 65)
print("EXERCISE 4: MIN-MAX SCALING OF NUMERICAL DATA")
print("=" * 65)
print(exercise4_df.round(3).to_string(index=False))

print("\nMin-Max Scaling Formula:")
print("X_scaled = (X - X_min) / (X_max - X_min)")


# ==================================================
# EXERCISE 5: COMPARE NORMALIZATION AND STANDARDIZATION
# ==================================================

numerical_standard_scaler = StandardScaler()

numerical_standard = numerical_standard_scaler.fit_transform(
    numerical_data
)

# L2 normalization is applied to each row independently.
numerical_l2 = normalize(numerical_data, norm="l2")

exercise5_df = pd.DataFrame({
    "Original": numerical_data.flatten(),
    "Min-Max": numerical_minmax.flatten(),
    "Standardized": numerical_standard.flatten(),
    "L2 Normalized": numerical_l2.flatten()
})

print("\n" + "=" * 65)
print("EXERCISE 5: SCALING TECHNIQUES COMPARISON")
print("=" * 65)
print(exercise5_df.round(3).to_string(index=False))

print("\nComparison:")
print("1. Min-Max Scaling converts values to a specified range,")
print("   usually between 0 and 1.")
print("2. Standardization gives the data a mean near 0 and")
print("   a standard deviation near 1.")
print("3. L2 Normalization scales each row to unit length.")
print("4. Standardization does not restrict values to 0 and 1.")


# --------------------------------------------------
# STEP 10: PLOT NUMERICAL SCALING COMPARISON
# --------------------------------------------------

plot_df = exercise5_df.set_index("Original")

plot_df.plot(kind="bar", figsize=(10, 6))

plt.title("Comparison of Feature Scaling Techniques")
plt.xlabel("Original Numerical Value")
plt.ylabel("Transformed Value")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()

print("\nDisplaying the feature scaling comparison graph...")
plt.show()


# --------------------------------------------------
# STEP 11: FINAL VERIFICATION
# --------------------------------------------------

print("\n" + "=" * 65)
print("FINAL VERIFICATION")
print("=" * 65)

print("Original text dataset loaded: YES")
print("Text preprocessing completed: YES")
print("Stop-word removal completed: YES")
print("Stemming completed: YES")
print("TF-IDF feature extraction completed: YES")
print("L2 normalization completed: YES")
print("Min-Max scaling completed: YES")
print("Standardization completed: YES")
print("Exercise 1 - CountVectorizer: YES")
print("Exercise 2 - TF-IDF comparison: YES")
print("Exercise 3 - Lemmatization: YES")
print("Exercise 4 - Numerical Min-Max scaling: YES")
print("Exercise 5 - Scaling comparison: YES")

print("\nExperiment 9 and all five exercises completed successfully.")
