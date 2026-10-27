# ============================================================
# EXPERIMENT 10
# Document Similarity Analysis Using Shingling and LSH
# Mapped CO: CO3
# ============================================================

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from datasketch import MinHash, MinHashLSH


# ============================================================
# 1. CREATE SAMPLE DOCUMENTS
# ============================================================

documents = {
    "D1": (
        "Machine learning is a branch of artificial intelligence "
        "that enables computers to learn from data."
    ),
    "D2": (
        "Machine learning enables computers to learn from data "
        "using artificial intelligence techniques."
    ),
    "D3": (
        "Data science uses statistics, visualization, and machine "
        "learning to analyze data."
    ),
    "D4": (
        "Natural language processing analyzes human language "
        "using computational techniques."
    ),
    "D5": (
        "Artificial intelligence includes machine learning and "
        "natural language processing."
    )
}

print("=" * 70)
print("EXPERIMENT 10: DOCUMENT SIMILARITY USING SHINGLING AND LSH")
print("=" * 70)

df = pd.DataFrame({
    "Document ID": list(documents.keys()),
    "Text": list(documents.values())
})

print("\n1. ORIGINAL DOCUMENTS")
print(df.to_string(index=False))


# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================

def preprocess(text):
    """Convert text to lowercase and extract words."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ============================================================
# 3. GENERATE WORD SHINGLES
# ============================================================

def generate_shingles(text, k=3):
    """
    Generate overlapping word shingles of size k.
    Example: 'machine learning uses data', k=2
    produces:
    {'machine learning', 'learning uses', 'uses data'}
    """
    words = preprocess(text)

    if len(words) < k:
        return set()

    return {
        " ".join(words[i:i + k])
        for i in range(len(words) - k + 1)
    }


K = 3

shingles = {
    doc_id: generate_shingles(text, K)
    for doc_id, text in documents.items()
}

print(f"\n2. SHINGLES GENERATED (k={K})")

for doc_id, shingle_set in shingles.items():
    print(f"\n{doc_id}: {len(shingle_set)} shingles")
    print(sorted(shingle_set))


# ============================================================
# 4. JACCARD SIMILARITY
# ============================================================

def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    union = set1 | set2

    if not union:
        return 1.0

    return len(set1 & set2) / len(union)


print("\n\n3. JACCARD SIMILARITY BETWEEN DOCUMENTS")

similarity_matrix = pd.DataFrame(
    0.0,
    index=documents.keys(),
    columns=documents.keys()
)

for doc_id in documents:
    similarity_matrix.loc[doc_id, doc_id] = 1.0

pair_results = []

for d1, d2 in combinations(documents.keys(), 2):
    score = jaccard_similarity(shingles[d1], shingles[d2])

    similarity_matrix.loc[d1, d2] = score
    similarity_matrix.loc[d2, d1] = score

    pair_results.append({
        "Document 1": d1,
        "Document 2": d2,
        "Jaccard Similarity": round(score, 4)
    })

pair_df = pd.DataFrame(pair_results)

print(pair_df.to_string(index=False))

print("\nJACCARD SIMILARITY MATRIX")
print(similarity_matrix.round(3).to_string())


# ============================================================
# 5. GENERATE MINHASH SIGNATURES
# ============================================================

NUM_PERM = 128

minhashes = {}

for doc_id, shingle_set in shingles.items():
    mh = MinHash(num_perm=NUM_PERM)

    for shingle in shingle_set:
        mh.update(shingle.encode("utf-8"))

    minhashes[doc_id] = mh

print("\n\n4. MINHASH SIGNATURES")

for doc_id, mh in minhashes.items():
    print(
        f"{doc_id}: Signature length = "
        f"{len(mh.hashvalues)}"
    )
    print("First 10 values:", mh.hashvalues[:10])


# ============================================================
# 6. BUILD LOCALITY SENSITIVE HASHING INDEX
# ============================================================

THRESHOLD = 0.30

lsh = MinHashLSH(
    threshold=THRESHOLD,
    num_perm=NUM_PERM
)

for doc_id, mh in minhashes.items():
    lsh.insert(doc_id, mh)

print("\n\n5. LSH INDEX")
print("LSH index created successfully.")
print("Configured similarity threshold:", THRESHOLD)


# ============================================================
# 7. QUERY SIMILAR DOCUMENTS USING LSH
# ============================================================

print("\n6. SIMILAR DOCUMENTS RETRIEVED USING LSH")

lsh_results = {}

for doc_id, mh in minhashes.items():
    candidates = sorted(lsh.query(mh))

    # Exclude the document itself from the displayed results.
    candidates = [
        candidate for candidate in candidates
        if candidate != doc_id
    ]

    lsh_results[doc_id] = candidates

    print(f"\nQuery document: {doc_id}")
    print("Similar document candidates:", candidates)

    if candidates:
        for candidate in candidates:
            estimated = minhashes[doc_id].jaccard(
                minhashes[candidate]
            )
            exact = jaccard_similarity(
                shingles[doc_id], shingles[candidate]
            )

            print(
                f"  {candidate}: "
                f"Estimated Jaccard = {estimated:.4f}, "
                f"Exact Jaccard = {exact:.4f}"
            )
    else:
        print("  No other candidates found at this threshold.")


# ============================================================
# 8. ASSIGNMENT 1: JACCARD SIMILARITY FOR TWO DOCUMENTS
# ============================================================

print("\n\n7. ASSIGNMENT 1: TWO-DOCUMENT COMPARISON")

doc_a = "D1"
doc_b = "D2"

score = jaccard_similarity(shingles[doc_a], shingles[doc_b])

print(f"Document A: {doc_a}")
print(f"Document B: {doc_b}")
print(f"Jaccard Similarity: {score:.4f}")
print(f"Jaccard Similarity (%): {score * 100:.2f}%")


# ============================================================
# 9. ASSIGNMENT 2: CHANGE SHINGLE SIZE
# ============================================================

print("\n\n8. ASSIGNMENT 2: EFFECT OF SHINGLE SIZE")

shingle_sizes = [1, 2, 3, 4, 5]
shingle_size_results = []

for k in shingle_sizes:
    set_a = generate_shingles(documents[doc_a], k)
    set_b = generate_shingles(documents[doc_b], k)

    score = jaccard_similarity(set_a, set_b)

    shingle_size_results.append({
        "Shingle Size (k)": k,
        "Document A Shingles": len(set_a),
        "Document B Shingles": len(set_b),
        "Jaccard Similarity": round(score, 4)
    })

shingle_size_df = pd.DataFrame(shingle_size_results)
print(shingle_size_df.to_string(index=False))


# ============================================================
# 10. ASSIGNMENT 3: ADD A NEW DOCUMENT AND QUERY LSH
# ============================================================

print("\n\n9. ASSIGNMENT 3: ADD A NEW DOCUMENT")

new_doc_id = "D6"

new_document = (
    "Machine learning and artificial intelligence "
    "are useful for analyzing data."
)

documents[new_doc_id] = new_document
new_shingles = generate_shingles(new_document, K)

new_minhash = MinHash(num_perm=NUM_PERM)

for shingle in new_shingles:
    new_minhash.update(shingle.encode("utf-8"))

# Insert the new document into the existing LSH index.
lsh.insert(new_doc_id, new_minhash)

print("New document added:", new_doc_id)
print("New document text:", new_document)
print("Number of shingles:", len(new_shingles))

new_candidates = sorted(lsh.query(new_minhash))
new_candidates = [
    candidate for candidate in new_candidates
    if candidate != new_doc_id
]

print("Similar documents retrieved by LSH:", new_candidates)

for candidate in new_candidates:
    exact = jaccard_similarity(
        new_shingles,
        generate_shingles(documents[candidate], K)
    )

    estimated = new_minhash.jaccard(minhashes[candidate])

    print(
        f"{new_doc_id} vs {candidate}: "
        f"Estimated = {estimated:.4f}, "
        f"Exact = {exact:.4f}"
    )


# ============================================================
# 11. ASSIGNMENT 4: COMPARE DIFFERENT LSH THRESHOLDS
# ============================================================

print("\n\n10. ASSIGNMENT 4: DIFFERENT LSH THRESHOLDS")

thresholds = [0.20, 0.40, 0.60, 0.80]
threshold_results = []

# Include all six documents in each threshold comparison.
all_minhashes = dict(minhashes)
all_minhashes[new_doc_id] = new_minhash

for threshold in thresholds:
    threshold_lsh = MinHashLSH(
        threshold=threshold,
        num_perm=NUM_PERM
    )

    for doc_id, mh in all_minhashes.items():
        threshold_lsh.insert(doc_id, mh)

    candidate_pairs = set()

    for doc_id, mh in all_minhashes.items():
        for candidate in threshold_lsh.query(mh):
            if candidate != doc_id:
                candidate_pairs.add(
                    tuple(sorted((doc_id, candidate)))
                )

    threshold_results.append({
        "Threshold": threshold,
        "Candidate Pairs Found": len(candidate_pairs),
        "Pairs": sorted(candidate_pairs)
    })

    print(f"\nThreshold = {threshold}")
    print("Candidate pairs found:", len(candidate_pairs))
    print("Pairs:", sorted(candidate_pairs))


# ============================================================
# 12. SAVE RESULTS TO CSV FILES
# ============================================================

# Save the original five-document similarity matrix.
similarity_matrix.to_csv("DS10_similarity_matrix.csv")

# Save pairwise Jaccard scores.
pair_df.to_csv("DS10_pairwise_similarity.csv", index=False)

# Save the shingle-size comparison.
shingle_size_df.to_csv("DS10_shingle_size_results.csv", index=False)

# Save LSH query results.
lsh_df = pd.DataFrame([
    {
        "Document ID": doc_id,
        "LSH Similar Documents": ", ".join(candidates)
    }
    for doc_id, candidates in lsh_results.items()
])

lsh_df.to_csv("DS10_lsh_results.csv", index=False)

# Save threshold comparison.
threshold_df = pd.DataFrame([
    {
        "Threshold": item["Threshold"],
        "Candidate Pairs Found": item["Candidate Pairs Found"],
        "Pairs": str(item["Pairs"])
    }
    for item in threshold_results
])

threshold_df.to_csv("DS10_threshold_results.csv", index=False)

print("\n\n11. RESULTS SAVED")
print("DS10_similarity_matrix.csv")
print("DS10_pairwise_similarity.csv")
print("DS10_shingle_size_results.csv")
print("DS10_lsh_results.csv")
print("DS10_threshold_results.csv")


# ============================================================
# 13. VISUALIZATION 1: JACCARD SIMILARITY HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    similarity_matrix.values,
    interpolation="nearest",
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(label="Jaccard Similarity")
plt.xticks(
    range(len(similarity_matrix.columns)),
    similarity_matrix.columns
)
plt.yticks(
    range(len(similarity_matrix.index)),
    similarity_matrix.index
)

plt.title("Document Similarity Heatmap (k=3)")
plt.xlabel("Documents")
plt.ylabel("Documents")

for i in range(len(similarity_matrix)):
    for j in range(len(similarity_matrix.columns)):
        plt.text(
            j, i,
            f"{similarity_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("DS10_similarity_heatmap.png", dpi=300)
plt.show()


# ============================================================
# 14. VISUALIZATION 2: SHINGLE SIZE VS SIMILARITY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    shingle_size_df["Shingle Size (k)"],
    shingle_size_df["Jaccard Similarity"],
    marker="o"
)

plt.title("Effect of Shingle Size on Jaccard Similarity")
plt.xlabel("Shingle Size (k)")
plt.ylabel("Jaccard Similarity")
plt.xticks(shingle_sizes)
plt.ylim(0, 1.05)
plt.grid(True)

plt.tight_layout()
plt.savefig("DS10_shingle_size_plot.png", dpi=300)
plt.show()


# ============================================================
# 15. VISUALIZATION 3: LSH THRESHOLD COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    threshold_df["Threshold"].astype(str),
    threshold_df["Candidate Pairs Found"]
)

plt.title("LSH Candidate Pairs at Different Thresholds")
plt.xlabel("LSH Similarity Threshold")
plt.ylabel("Number of Candidate Pairs")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("DS10_threshold_comparison.png", dpi=300)
plt.show()


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 10 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("Documents analyzed initially:", len(df))
print("Shingle size used for main experiment:", K)
print("MinHash permutations:", NUM_PERM)
print("LSH threshold:", THRESHOLD)
print("New document added:", new_doc_id)

print("\nGenerated visualizations:")
print("1. DS10_similarity_heatmap.png")
print("2. DS10_shingle_size_plot.png")
print("3. DS10_threshold_comparison.png")

print("\nAll main experiment steps and five assignments executed.")
