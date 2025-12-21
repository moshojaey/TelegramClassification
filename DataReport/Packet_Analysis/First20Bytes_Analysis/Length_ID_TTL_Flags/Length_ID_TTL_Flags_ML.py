import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

ROOT = Path(r"D:\New_ITC_Reformatted\Length_ID_TTL_Flags")
RESULTS = Path(r"D:\Codes\LabProj\Telegram_New\Result\Length_ID_TTL_Flags")

def load_dataset(ds_id):
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        path = ROOT / ds_id / f"{cat}.json"
        with open(path, "r") as f:
            j = json.load(f)
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))
    return np.array(xs), np.array(ys)

X_train, y_train = load_dataset("1")
X_test, y_test   = load_dataset("2")

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "KNN (k=5)":     KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":   GaussianNB()
}

print("\n--- ML Results (4 Features: Len, ID, TTL, Flags) ---")
results = []
for name, clf in models.items():
    clf.fit(X_train, y_train_enc)
    acc = accuracy_score(y_test_enc, clf.predict(X_test))
    print(f"{name:<20} | {acc:.4f}")
    results.append({"Model": name, "Accuracy": acc})

pd.DataFrame(results).to_csv(RESULTS / "ml_benchmark.csv", index=False)