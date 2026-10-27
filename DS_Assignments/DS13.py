# ================================================================
# EXPERIMENT 13
# FREQUENT ITEMSET MINING AND ASSOCIATION RULE GENERATION
# Algorithm: Apriori
# Libraries: Pandas, Mlxtend, Matplotlib
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

print("=" * 75)
print("EXPERIMENT 13: FREQUENT ITEMSET MINING AND ASSOCIATION RULE GENERATION")
print("=" * 75)

# ------------------------------------------------
# 1. CREATE THE TRANSACTION DATASET
# ------------------------------------------------

transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

transaction_ids = ['T1', 'T2', 'T3', 'T4', 'T5']

print("\n1. ORIGINAL TRANSACTION DATASET")

for tid, items in zip(transaction_ids, transactions):
    print(f"{tid}: {', '.join(items)}")

# ------------------------------------------------
# 2. ONE-HOT ENCODING
# ------------------------------------------------

encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_,
    index=transaction_ids
)

print("\n2. ONE-HOT ENCODED DATASET")
print(encoded_df.astype(int))

# ------------------------------------------------
# 3. APPLY APRIORI ALGORITHM
# ------------------------------------------------

min_support = 0.4

frequent_itemsets = apriori(
    encoded_df,
    min_support=min_support,
    use_colnames=True
)

frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
    lambda items: ', '.join(sorted(items))
)

print(f"\n3. FREQUENT ITEMSETS (Minimum Support = {min_support})")
print(frequent_itemsets.to_string(index=False))

# Keep the original frozenset itemsets for rule generation
frequent_itemsets_raw = apriori(
    encoded_df,
    min_support=min_support,
    use_colnames=True
)

# ------------------------------------------------
# 4. GENERATE ASSOCIATION RULES
# ------------------------------------------------

rules = association_rules(
    frequent_itemsets_raw,
    metric='confidence',
    min_threshold=0.5
)

if not rules.empty:
    rules_display = rules[
        ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    ].copy()

    rules_display['antecedents'] = rules_display['antecedents'].apply(
        lambda x: ', '.join(sorted(x))
    )
    rules_display['consequents'] = rules_display['consequents'].apply(
        lambda x: ', '.join(sorted(x))
    )

    print("\n4. ASSOCIATION RULES")
    print(rules_display.round(3).to_string(index=False))
else:
    rules_display = pd.DataFrame()
    print("\n4. No association rules found.")

# ------------------------------------------------
# EXERCISE 1: CHANGE MINIMUM SUPPORT
# ------------------------------------------------

print("\n" + "=" * 75)
print("EXERCISE 1: EFFECT OF CHANGING MINIMUM SUPPORT")
print("=" * 75)

support_values = [0.2, 0.4, 0.6]

for support in support_values:
    result = apriori(
        encoded_df,
        min_support=support,
        use_colnames=True
    )

    result['itemsets'] = result['itemsets'].apply(
        lambda x: ', '.join(sorted(x))
    )

    print(f"\nMinimum support = {support}")
    print(result.to_string(index=False))

# ------------------------------------------------
# EXERCISE 2: HIGHER CONFIDENCE THRESHOLD
# ------------------------------------------------

print("\n" + "=" * 75)
print("EXERCISE 2: ASSOCIATION RULES WITH HIGHER CONFIDENCE")
print("=" * 75)

high_conf_rules = association_rules(
    frequent_itemsets_raw,
    metric='confidence',
    min_threshold=0.7
)

if not high_conf_rules.empty:
    high_conf_display = high_conf_rules[
        ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    ].copy()

    high_conf_display['antecedents'] = (
        high_conf_display['antecedents'].apply(
            lambda x: ', '.join(sorted(x))
        )
    )
    high_conf_display['consequents'] = (
        high_conf_display['consequents'].apply(
            lambda x: ', '.join(sorted(x))
        )
    )

    print(high_conf_display.round(3).to_string(index=False))
else:
    print("No rules satisfy the higher confidence threshold.")

# ------------------------------------------------
# EXERCISE 3: ADD NEW TRANSACTIONS
# ------------------------------------------------

print("\n" + "=" * 75)
print("EXERCISE 3: ADD NEW TRANSACTIONS")
print("=" * 75)

new_transactions = transactions + [
    ['Milk', 'Bread'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter']
]

new_encoder = TransactionEncoder()
new_encoded_array = new_encoder.fit(new_transactions).transform(
    new_transactions
)

new_encoded_df = pd.DataFrame(
    new_encoded_array,
    columns=new_encoder.columns_
)

new_frequent_itemsets = apriori(
    new_encoded_df,
    min_support=0.4,
    use_colnames=True
)

new_frequent_itemsets['itemsets'] = (
    new_frequent_itemsets['itemsets'].apply(
        lambda x: ', '.join(sorted(x))
    )
)

print("\nDataset after adding 3 transactions:")
print("Original transactions:", len(transactions))
print("New total transactions:", len(new_transactions))

print("\nNew frequent itemsets:")
print(new_frequent_itemsets.to_string(index=False))

# ------------------------------------------------
# EXERCISE 4: RULE WITH THE HIGHEST LIFT
# ------------------------------------------------

print("\n" + "=" * 75)
print("EXERCISE 4: RULE WITH HIGHEST LIFT")
print("=" * 75)

if not rules.empty:
    best_rule = rules.loc[rules['lift'].idxmax()]

    antecedent = ', '.join(sorted(best_rule['antecedents']))
    consequent = ', '.join(sorted(best_rule['consequents']))

    print(f"Rule: {antecedent} -> {consequent}")
    print(f"Support: {best_rule['support']:.3f}")
    print(f"Confidence: {best_rule['confidence']:.3f}")
    print(f"Lift: {best_rule['lift']:.3f}")
else:
    print("No rules available to compare lift values.")

# ------------------------------------------------
# EXERCISE 5: EXPLAIN SUPPORT, CONFIDENCE AND LIFT
# ------------------------------------------------

print("\n" + "=" * 75)
print("EXERCISE 5: SIGNIFICANCE OF EVALUATION MEASURES")
print("=" * 75)

print("""
1. SUPPORT:
   The proportion of all transactions containing an itemset.
   Support(A) = Transactions containing A / Total transactions

2. CONFIDENCE:
   The proportion of transactions containing A that also contain B.
   Confidence(A -> B) = Support(A union B) / Support(A)

3. LIFT:
   Measures the strength of association between A and B.
   Lift(A -> B) = Confidence(A -> B) / Support(B)

   Lift > 1 : Positive association
   Lift = 1 : No association beyond independence
   Lift < 1 : Negative association
""")

# ------------------------------------------------
# GRAPH 1: SUPPORT OF FREQUENT ITEMSETS
# ------------------------------------------------

original_itemsets = apriori(
    encoded_df,
    min_support=0.2,
    use_colnames=True
)

original_itemsets['label'] = original_itemsets['itemsets'].apply(
    lambda x: ', '.join(sorted(x))
)

plt.figure(figsize=(10, 6))
plt.bar(
    original_itemsets['label'],
    original_itemsets['support']
)
plt.title('Experiment 13: Support of Frequent Itemsets')
plt.xlabel('Itemsets')
plt.ylabel('Support')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('DS13_graph1_support.png', dpi=300)
plt.show()

# ------------------------------------------------
# GRAPH 2: CONFIDENCE AND LIFT OF ASSOCIATION RULES
# ------------------------------------------------

if not rules.empty:
    graph_rules = rules.copy()

    graph_rules['rule'] = graph_rules.apply(
        lambda row:
        ', '.join(sorted(row['antecedents'])) + ' -> ' +
        ', '.join(sorted(row['consequents'])),
        axis=1
    )

    plt.figure(figsize=(11, 6))
    x = range(len(graph_rules))
    width = 0.35

    plt.bar(
        [i - width / 2 for i in x],
        graph_rules['confidence'],
        width=width,
        label='Confidence'
    )

    plt.bar(
        [i + width / 2 for i in x],
        graph_rules['lift'],
        width=width,
        label='Lift'
    )

    plt.title('Experiment 13: Confidence and Lift of Association Rules')
    plt.xlabel('Association Rules')
    plt.ylabel('Metric Value')
    plt.xticks(list(x), graph_rules['rule'], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('DS13_graph2_rules.png', dpi=300)
    plt.show()

# ------------------------------------------------
# SAVE OUTPUT TABLES
# ------------------------------------------------

frequent_itemsets_raw.to_csv(
    'DS13_frequent_itemsets.csv', index=False
)

if not rules.empty:
    rules_display.to_csv('DS13_association_rules.csv', index=False)

print("\n" + "=" * 75)
print("EXPERIMENT 13 COMPLETED SUCCESSFULLY")
print("Saved files:")
print("1. DS13_graph1_support.png")
print("2. DS13_graph2_rules.png")
print("3. DS13_frequent_itemsets.csv")
print("4. DS13_association_rules.csv (if rules were generated)")
print("=" * 75)
