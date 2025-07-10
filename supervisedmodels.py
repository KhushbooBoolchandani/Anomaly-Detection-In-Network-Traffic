import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from joblib import dump

# ---------------------------
# Load Dataset
# ---------------------------
df = pd.read_csv("kddcup_clean.csv")
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ---------------------------
# Model Configurations
# ---------------------------
models = {
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

metrics_table = []

# ---------------------------
# Train, Evaluate, Save
# ---------------------------
for name, model in models.items():
    print(f"\n🔷 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Save model
    dump(model, f"models/{name.lower()}_model.joblib")

    # Evaluation Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    metrics_table.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": auc
    })

    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"static/{name.lower()}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f"{name} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"static/{name.lower()}_roc_curve.png")
    plt.close()

    print(f"✅ Saved model & plots for {name}")

# ---------------------------
# Show Final Comparison
# ---------------------------
results_df = pd.DataFrame(metrics_table)
print("\n📊 Model Performance Comparison:\n")
print(results_df.sort_values(by="F1 Score", ascending=False).round(4))

# ---------------------------
# Best Model Announcement
# ---------------------------
best_model = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]
print(f"\n🏆 Best Performing Model: {best_model['Model']}")
print(f"   ➤ Accuracy : {best_model['Accuracy']:.4f}")
print(f"   ➤ Precision: {best_model['Precision']:.4f}")
print(f"   ➤ Recall   : {best_model['Recall']:.4f}")
print(f"   ➤ F1 Score : {best_model['F1 Score']:.4f}")
print(f"   ➤ AUC      : {best_model['AUC']:.4f}")

