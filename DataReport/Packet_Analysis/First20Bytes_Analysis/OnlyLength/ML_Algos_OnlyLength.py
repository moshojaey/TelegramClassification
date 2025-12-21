import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ==================================================================
# CONFIGURATION
# ==================================================================
ROOT = Path(r"D:\New_ITC_Reformatted\OnlyLength")
RESULTS = Path(r"D:\Codes\LabProj\Telegram_New\Result\ML_OnlyLength")
RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_DS = "1"
TEST_DS = "2"


# ==================================================================
# DATA LOADER
# ==================================================================
def load_dataset(ds_id):
    print(f"Loading Dataset {ds_id} ...")
    xs, ys = [], []
    for cat in ["Telegram", "Non-Telegram"]:
        path = ROOT / ds_id / f"{cat}.json"
        if not path.exists():
            print(f"[!] Missing: {path}")
            continue

        with open(path, "r") as f:
            j = json.load(f)

        # Sklearn expects shape (N_samples, N_features)
        # Your data is [[0.5], [0.1], ...], which is already correct for 1 feature
        xs.extend(j["features"])
        ys.extend([cat.lower()] * len(j["features"]))

    return np.array(xs), np.array(ys)


# ==================================================================
# MAIN
# ==================================================================
def main():
    # 1. Load Data
    X_train, y_train = load_dataset(TRAIN_DS)
    X_test, y_test = load_dataset(TEST_DS)

    print(f"\nTrain Shape: {X_train.shape} (Rows, Features)")
    print(f"Test Shape:  {X_test.shape}")

    # 2. Encode Labels (Telegram=1, Non-Telegram=0)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # 3. Define Models to Test
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Logistic Reg": LogisticRegression(random_state=42),
        # SVM can be slow on large datasets, uncomment if dataset < 50k
        # "SVM (RBF)":     SVC(kernel='rbf', max_iter=10000)
    }

    results = []

    print("\n" + "=" * 40)
    print(f"{'Algorithm':<20} | {'Accuracy':<10}")
    print("=" * 40)

    # 4. Train and Evaluate Loop
    for name, clf in models.items():
        # Train
        clf.fit(X_train, y_train_enc)

        # Predict
        y_pred = clf.predict(X_test)

        # Score
        acc = accuracy_score(y_test_enc, y_pred)

        print(f"{name:<20} | {acc:.4f}")

        results.append({
            "Algorithm": name,
            "Accuracy": acc,
            "Train_DS": TRAIN_DS,
            "Test_DS": TEST_DS
        })

    # 5. Save Summary
    df = pd.DataFrame(results)
    out_path = RESULTS / "ml_benchmark_length_only.csv"
    df.to_csv(out_path, index=False)
    print("=" * 40)
    print(f"\n[✓] Results saved to: {out_path}")

    # 6. (Optional) Show Detailed Report for Best Model
    best_model_name = df.loc[df['Accuracy'].idxmax()]['Algorithm']
    print(f"\nBest Model: {best_model_name}")

    # Re-run best model to show report
    best_clf = models[best_model_name]
    y_pred_best = best_clf.predict(X_test)
    print(classification_report(y_test_enc, y_pred_best, target_names=le.classes_))


if __name__ == "__main__":
    main()