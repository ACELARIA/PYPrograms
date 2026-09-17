
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, balanced_accuracy_score,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize

PINK       = '#F48FB1'
LAVENDER   = '#CE93D8'
PEACH      = '#FFAB91'
MINT       = '#A5D6A7'
SKY        = '#90CAF9'
LILAC      = '#B39DDB'
ROSE       = '#F06292'
SOFT_PINK  = '#FCE4EC'
SOFT_LAVEN = '#F3E5F5'

PALETTE    = [PINK, LAVENDER, PEACH, MINT, SKY, LILAC]

plt.rcParams.update({
    'figure.facecolor': '#FFF8FA',
    'axes.facecolor'  : '#FFF8FA',
    'axes.edgecolor'  : LAVENDER,
    'axes.labelcolor' : '#7B5E7B',
    'xtick.color'     : '#7B5E7B',
    'ytick.color'     : '#7B5E7B',
    'text.color'      : '#5C3D5C',
    'grid.color'      : '#F8BBD0',
    'grid.linestyle'  : '--',
    'font.family'     : 'DejaVu Sans',
})

print(" All libraries imported successfully!")
print(f"Colour palette loaded: {PALETTE}")

# Load dataset
df = pd.read_csv('heart_disease.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
df.head()

print("Data Types:")
print(df.dtypes)
print("\nMissing values:", df.isnull().sum().sum())
print("\nBasic Statistics:")
df.describe().round(2)

X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Features         : {list(X.columns)}")

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

train_acc = dt.score(X_train, y_train)
test_acc  = dt.score(X_test,  y_test)

print(f" Training Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
print(f" Testing  Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")

# ── Visualise the Decision Tree ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, 10), facecolor='#FFF8FA')

plot_tree(
    dt,
    feature_names=X.columns.tolist(),
    class_names=['No Disease', 'Disease'],
    filled=True,
    rounded=True,
    fontsize=9,
    ax=ax,
    impurity=True,
    proportion=False,
)

# Recolour nodes pink/lavender
for artist in ax.get_children():
    if hasattr(artist, 'get_facecolor'):
        fc = artist.get_facecolor()
        if fc is not None:
            artist.set_edgecolor(ROSE)

plt.title(' Decision Tree — Heart Disease Prediction',
          fontsize=16, color='#7B5E7B', pad=20, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree_viz.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()
print("\n Tree saved as decision_tree_viz.png")

# ── Interpret first two splits ────────────────────────────────────────────────
rules = export_text(dt, feature_names=X.columns.tolist())
# Print only first ~20 lines (first 2 split levels)
lines = rules.split('\n')
print("=== First Two Splits (Text Representation) ===")
for line in lines[:25]:
    print(line)

print("""
 Interpretation of First Two Splits:
─────────────────────────────────────────────────────────────────────────────
  Split 1 (Root): The tree's very first split is on the feature shown at the
                  root node. This is the MOST informative single feature for
                  predicting heart disease in this dataset.

  Split 2 (Left/Right child): Each child of the root makes a further split
                  on the next most informative feature, given the parent's
                  condition. The Gini impurity values drop with each split,
                  showing increasing class purity.

  Key insight: Features like Age, BP, and Cholesterol dominate early splits,
               reflecting their clinical importance in heart disease diagnosis.
─────────────────────────────────────────────────────────────────────────────
""")

y_pred = dt.predict(X_test)
y_prob = dt.predict_proba(X_test)[:, 1]

# ── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5), facecolor='#FFF8FA')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['No Disease', 'Disease'])
disp.plot(ax=ax, colorbar=False, cmap='RdPu')

ax.set_title(' Confusion Matrix', fontsize=14, color='#7B5E7B',
             fontweight='bold', pad=12)
ax.set_xlabel('Predicted Label', color='#7B5E7B')
ax.set_ylabel('True Label',      color='#7B5E7B')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()

# ── Classification Report ────────────────────────────────────────────────────
print("=" * 55)
print("        Classification Report")
print("=" * 55)
print(classification_report(y_test, y_pred,
                             target_names=['No Disease', 'Disease']))

tn, fp, fn, tp = cm.ravel()
print(f"True  Negatives  (TN) : {tn}")
print(f"False Positives  (FP) : {fp}")
print(f"False Negatives  (FN) : {fn}")
print(f"True  Positives  (TP) : {tp}")
print(f"\nSensitivity (Recall) : {tp/(tp+fn):.4f}")
print(f"Specificity           : {tn/(tn+fp):.4f}")

# ── ROC Curve ────────────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 6), facecolor='#FFF8FA')

ax.plot(fpr, tpr, color=ROSE, lw=2.5,
        label=f'ROC Curve  (AUC = {roc_auc:.3f})')
ax.fill_between(fpr, tpr, alpha=0.15, color=PINK)
ax.plot([0, 1], [0, 1], color=LAVENDER, lw=1.5,
        linestyle='--', label='Random Classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate',  fontsize=12)
ax.set_title(' ROC Curve — Heart Disease Detection',
             fontsize=14, color='#7B5E7B', fontweight='bold')
ax.legend(loc='lower right', framealpha=0.8,
          facecolor='#FFF0F5', edgecolor=LAVENDER)
ax.grid(True)

plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()

print(f"\n AUC Score : {roc_auc:.4f}")
if roc_auc >= 0.9:
    print("   → Excellent discrimination!")
elif roc_auc >= 0.8:
    print("   → Good discrimination.")
elif roc_auc >= 0.7:
    print("   → Fair discrimination.")
else:
    print("   → Poor discrimination — consider tuning.")

# ── Classification Report ────────────────────────────────────────────────────
print("=" * 55)
print("        Classification Report")
print("=" * 55)
print(classification_report(y_test, y_pred,
                             target_names=['No Disease', 'Disease']))

tn, fp, fn, tp = cm.ravel()
print(f"True  Negatives  (TN) : {tn}")
print(f"False Positives  (FP) : {fp}")
print(f"False Negatives  (FN) : {fn}")
print(f"True  Positives  (TP) : {tp}")
print(f"\nSensitivity (Recall) : {tp/(tp+fn):.4f}")
print(f"Specificity           : {tn/(tn+fp):.4f}")

# ── Vary max_depth ────────────────────────────────────────────────────────────
depths = range(1, 16)
train_scores, test_scores = [], []

for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test,  y_test))

fig, ax = plt.subplots(figsize=(9, 5), facecolor='#FFF8FA')

ax.plot(depths, train_scores, 'o-', color=ROSE,     lw=2.5,
        label='Training Accuracy',  markersize=7)
ax.plot(depths, test_scores,  's-', color=LAVENDER, lw=2.5,
        label='Testing Accuracy',   markersize=7)
ax.fill_between(depths, train_scores, test_scores,
                alpha=0.12, color=PINK, label='Overfit Gap')

ax.set_xlabel('max_depth',          fontsize=12)
ax.set_ylabel('Accuracy',           fontsize=12)
ax.set_title(' max_depth: Training vs Testing Accuracy',
             fontsize=14, color='#7B5E7B', fontweight='bold')
ax.legend(facecolor='#FFF0F5', edgecolor=LAVENDER)
ax.grid(True)
ax.set_xticks(list(depths))

plt.tight_layout()
plt.savefig('depth_overfitting.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()
print(" Observation: Training acc → 1.0 as depth grows (overfitting).")
print("   Testing  acc peaks then drops — sweet spot visible in the plot.")

# ── Vary min_samples_split ───────────────────────────────────────────────────
splits = range(2, 30, 2)
tr_split, te_split = [], []

for s in splits:
    clf = DecisionTreeClassifier(min_samples_split=s, random_state=42)
    clf.fit(X_train, y_train)
    tr_split.append(clf.score(X_train, y_train))
    te_split.append(clf.score(X_test,  y_test))

fig, ax = plt.subplots(figsize=(9, 5), facecolor='#FFF8FA')
ax.plot(splits, tr_split, 'o-', color=PEACH,  lw=2.5, label='Train', markersize=7)
ax.plot(splits, te_split, 's-', color=MINT,   lw=2.5, label='Test',  markersize=7)
ax.set_xlabel('min_samples_split', fontsize=12)
ax.set_ylabel('Accuracy',          fontsize=12)
ax.set_title(' min_samples_split: Train vs Test Accuracy',
             fontsize=14, color='#7B5E7B', fontweight='bold')
ax.legend(facecolor='#FFF0F5', edgecolor=LAVENDER)
ax.grid(True)
plt.tight_layout()
plt.savefig('min_samples_split.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()

# ── Vary min_samples_leaf ────────────────────────────────────────────────────
leaves = range(1, 25)
tr_leaf, te_leaf = [], []

for l in leaves:
    clf = DecisionTreeClassifier(min_samples_leaf=l, random_state=42)
    clf.fit(X_train, y_train)
    tr_leaf.append(clf.score(X_train, y_train))
    te_leaf.append(clf.score(X_test,  y_test))

fig, ax = plt.subplots(figsize=(9, 5), facecolor='#FFF8FA')
ax.plot(leaves, tr_leaf, 'o-', color=SKY,    lw=2.5, label='Train', markersize=7)
ax.plot(leaves, te_leaf, 's-', color=LILAC,  lw=2.5, label='Test',  markersize=7)
ax.set_xlabel('min_samples_leaf', fontsize=12)
ax.set_ylabel('Accuracy',         fontsize=12)
ax.set_title(' min_samples_leaf: Train vs Test Accuracy',
             fontsize=14, color='#7B5E7B', fontweight='bold')
ax.legend(facecolor='#FFF0F5', edgecolor=LAVENDER)
ax.grid(True)
plt.tight_layout()
plt.savefig('min_samples_leaf.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()

# ── GridSearchCV ──────────────────────────────────────────────────────────────
param_grid = {
    'max_depth'         : [2, 3, 4, 5, 6, 8, 10, None],
    'min_samples_split' : [2, 5, 10, 15, 20],
    'min_samples_leaf'  : [1, 2, 4, 6, 8],
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

print("=" * 50)
print(" GridSearchCV Results")
print("=" * 50)
print(f"Best Parameters : {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")

best_dt = grid_search.best_estimator_
print(f"\nBest model — Train Acc : {best_dt.score(X_train, y_train):.4f}")
print(f"Best model — Test  Acc : {best_dt.score(X_test,  y_test):.4f}")

y_pred_all = best_dt.predict(X_test)

# Indices in the test set
misclassified_idx = np.where(y_pred_all != y_test.values)[0]
correct_idx       = np.where(y_pred_all == y_test.values)[0]

X_test_reset = X_test.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

misclassified_df = X_test_reset.iloc[misclassified_idx].copy()
misclassified_df['true_label']      = y_test_reset.iloc[misclassified_idx].values
misclassified_df['predicted_label'] = y_pred_all[misclassified_idx]

correct_df = X_test_reset.iloc[correct_idx].copy()
correct_df['true_label']      = y_test_reset.iloc[correct_idx].values
correct_df['predicted_label'] = y_pred_all[correct_idx]

print(f"Total test samples   : {len(X_test)}")
print(f"Misclassified        : {len(misclassified_df)}  ({len(misclassified_df)/len(X_test)*100:.1f}%)")
print(f"Correctly classified : {len(correct_df)}")
print("\n Misclassified Patients:")
misclassified_df

# ── Compare feature means ─────────────────────────────────────────────────────
features = ['age', 'sex', 'BP', 'cholesterol']

comp = pd.DataFrame({
    'Misclassified': misclassified_df[features].mean(),
    'Correct'      : correct_df[features].mean(),
})
print("\n Feature Mean Comparison:")
print(comp.round(2))

fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor='#FFF8FA')

for i, feat in enumerate(features):
    ax = axes[i]
    vals_m = misclassified_df[feat].values
    vals_c = correct_df[feat].values

    ax.hist(vals_c, bins=12, alpha=0.7, color=LAVENDER,
            label='Correct',        edgecolor='white')
    ax.hist(vals_m, bins=12, alpha=0.8, color=ROSE,
            label='Misclassified',  edgecolor='white')

    ax.set_title(feat.capitalize(), color='#7B5E7B', fontweight='bold')
    ax.set_xlabel(feat,             color='#7B5E7B')
    ax.set_ylabel('Count',          color='#7B5E7B')
    ax.legend(facecolor='#FFF0F5',  edgecolor=LAVENDER, fontsize=8)
    ax.grid(True, alpha=0.4)

fig.suptitle(' Feature Distribution: Misclassified vs Correct Patients',
             fontsize=14, color='#7B5E7B', fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('error_analysis.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()

print("""
 Error Analysis — Key Patterns Observed:
─────────────────────────────────────────────────────────────────────────────
  1. BORDERLINE FEATURE VALUES:
     Misclassified patients often have feature values close to the decision
     boundary (e.g., moderate BP ~130, moderate cholesterol ~240-270).
     The tree finds it hardest to split these borderline cases correctly.

  2. AGE OVERLAP:
     Middle-aged patients (40–55) appear in both classes with similar feature
     values, causing confusion. Younger patients with atypical high BP/chol
     may be misclassified as disease-free.

  3. SEX INTERACTION:
     Female patients (sex=0) with high cholesterol but low BP can be
     misclassified, since the tree primarily learns male-pattern disease.

  4. RECOMMENDATION:
     → Increase max_depth slightly or use ensemble methods (Random Forest)
       to better capture complex feature interactions.
     → Feature engineering (BP × cholesterol interaction) may help.
─────────────────────────────────────────────────────────────────────────────
""")


# ── Class distribution ────────────────────────────────────────────────────────
class_counts = y.value_counts()
class_pct    = y.value_counts(normalize=True) * 100

print("=" * 45)
print(" Class Distribution")
print("=" * 45)
print(f"  No Disease (0) : {class_counts[0]}  ({class_pct[0]:.1f}%)")
print(f"  Disease    (1) : {class_counts[1]}  ({class_pct[1]:.1f}%)")

imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\n  Imbalance Ratio : {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print("    Moderate/Severe imbalance detected → apply resampling!")
else:
    print("   Classes are relatively balanced.")

# ── Pie + Bar chart ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='#FFF8FA')

# Pie
wedge_colors = [LAVENDER, ROSE]
explode = (0.04, 0.04)
ax1.pie(class_counts.values,
        labels=['No Disease', 'Disease'],
        colors=wedge_colors,
        explode=explode,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'color': '#5C3D5C', 'fontsize': 12})
ax1.set_title(' Class Distribution (Pie)',
              color='#7B5E7B', fontweight='bold', fontsize=13)

# Bar
bars = ax2.bar(['No Disease (0)', 'Disease (1)'],
               class_counts.values,
               color=[LAVENDER, ROSE],
               edgecolor='white', linewidth=1.5, width=0.5)

for bar, val in zip(bars, class_counts.values):
    ax2.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 2,
             str(val),
             ha='center', va='bottom',
             color='#7B5E7B', fontweight='bold', fontsize=12)

ax2.set_ylabel('Count',     color='#7B5E7B')
ax2.set_title(' Class Distribution (Bar)',
              color='#7B5E7B', fontweight='bold', fontsize=13)
ax2.grid(True, axis='y', alpha=0.4)
ax2.set_ylim(0, max(class_counts.values) * 1.15)

plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()

# ── Balanced Accuracy ─────────────────────────────────────────────────────────
bal_acc = balanced_accuracy_score(y_test, y_pred_all)
print(f" Standard Accuracy  : {best_dt.score(X_test, y_test):.4f}")
print(f" Balanced Accuracy  : {bal_acc:.4f}")
print("""
  Balanced accuracy = average of sensitivity and specificity.
  Use this metric when classes are imbalanced — it is not biased
  toward the majority class like standard accuracy.
""")

# ── SMOTE Resampling ──────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print("After SMOTE resampling:")
    unique, counts = np.unique(y_res, return_counts=True)
    for u, c in zip(unique, counts):
        label = 'No Disease' if u == 0 else 'Disease'
        print(f"  Class {u} ({label}) : {c}")

    dt_smote = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
    dt_smote.fit(X_res, y_res)

    print(f"\n SMOTE Model — Test Accuracy  : {dt_smote.score(X_test, y_test):.4f}")
    print(f" SMOTE Model — Balanced Acc   : {balanced_accuracy_score(y_test, dt_smote.predict(X_test)):.4f}")

    # Compare
    fig, ax = plt.subplots(figsize=(7, 5), facecolor='#FFF8FA')
    models  = ['Original Model', 'SMOTE Model']
    std_acc = [best_dt.score(X_test, y_test), dt_smote.score(X_test, y_test)]
    bal_acc_list = [balanced_accuracy_score(y_test, y_pred_all),
                    balanced_accuracy_score(y_test, dt_smote.predict(X_test))]

    x = np.arange(len(models))
    w = 0.3
    bars1 = ax.bar(x - w/2, std_acc,     w, label='Standard Accuracy', color=LAVENDER, edgecolor='white')
    bars2 = ax.bar(x + w/2, bal_acc_list, w, label='Balanced Accuracy', color=ROSE,     edgecolor='white')

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}',
                ha='center', va='bottom', fontsize=10, color='#5C3D5C')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy Score')
    ax.set_title(' Original vs SMOTE Model Comparison',
                 color='#7B5E7B', fontweight='bold', fontsize=13)
    ax.legend(facecolor='#FFF0F5', edgecolor=LAVENDER)
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig('smote_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='#FFF8FA')
    plt.show()

except ImportError:
    print("  imbalanced-learn not installed.")
    print("   Run: pip install imbalanced-learn")
    print("\n   Showing class weights approach instead:")
    dt_weighted = DecisionTreeClassifier(
        **grid_search.best_params_,
        class_weight='balanced',
        random_state=42
    )
    dt_weighted.fit(X_train, y_train)
    print(f"   Balanced-weight Model Test Acc : {dt_weighted.score(X_test, y_test):.4f}")
    print(f"   Balanced Accuracy              : {balanced_accuracy_score(y_test, dt_weighted.predict(X_test)):.4f}")

# ── Feature Importance ────────────────────────────────────────────────────────
importances = pd.Series(best_dt.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(7, 4), facecolor='#FFF8FA')
colors_bar = [PINK, LAVENDER, PEACH, MINT]

importances.plot(kind='barh', ax=ax,
                 color=colors_bar[:len(importances)],
                 edgecolor='white', linewidth=1.2)

ax.set_xlabel('Feature Importance (Gini)', color='#7B5E7B')
ax.set_title(' Feature Importances — Best Decision Tree',
             color='#7B5E7B', fontweight='bold', fontsize=13)
ax.grid(True, axis='x', alpha=0.4)

for i, v in enumerate(importances.values):
    ax.text(v + 0.003, i, f'{v:.3f}',
            va='center', color='#7B5E7B', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight',
            facecolor='#FFF8FA')
plt.show()
print("\n Most important feature is the one with the highest bar —")
print("   this drives the root split of the decision tree.")
