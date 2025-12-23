import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ==================================================================
# CONFIGURATION
# ==================================================================
ROOT = Path(r"D:\New_ITC_Reformatted\Features20Full")
RESULTS = Path(r"D:\Codes\LabProj\Telegram_New\Result\Features20Full")
RESULTS.mkdir(parents=True, exist_ok=True)


# ==================================================================
# DATA LOADING
# ==================================================================
def load_dataset(ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        path = ROOT / ds_id / f"{cat}.json"
        with open(path, "r") as f:
            j = json.load(f)
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))
    return np.array(xs), np.array(ys)


print("Loading Data...")
X_train, y_train = load_dataset("1")
X_test, y_test = load_dataset("2")

# Encode Labels (Non-Telegram=0, Telegram=1)
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

print(f"Class Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ==================================================================
# MODEL DEFINITIONS
# ==================================================================
models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

# ==================================================================
# TRAINING & EVALUATION LOOP
# ==================================================================
print("\n" + "=" * 80)
print(f"{'Algorithm':<15} | {'Acc':<8} | {'Prec':<8} | {'Recall':<8} | {'TP':<6} | {'TN':<6} | {'FP':<6} | {'FN':<6}")
print("=" * 80)

results = []

for name, clf in models.items():
    # Train
    clf.fit(X_train, y_train_enc)

    # Predict
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test_enc, y_pred)
    prec = precision_score(y_test_enc, y_pred, average='binary')
    rec = recall_score(y_test_enc, y_pred, average='binary')

    # Confusion Matrix
    cm = confusion_matrix(y_test_enc, y_pred)
    # Binary classification returns [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()

    # Print Row
    print(f"{name:<15} | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {tp:<6} | {tn:<6} | {fp:<6} | {fn:<6}")

    # Store results
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    })

print("=" * 80)

# ==================================================================
# SAVE RESULTS
# ==================================================================
df = pd.DataFrame(results)
out_csv = RESULTS / "ml_benchmark_full_features_detailed.csv"
df.to_csv(out_csv, index=False)
print(f"\n[✓] Detailed report saved to: {out_csv}")