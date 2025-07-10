import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from joblib import load
from keras.models import load_model
from keras.losses import MeanSquaredError

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("kddcup_clean.csv")
X = df.drop(columns=["label"])
y_true = df["label"]

# -------------------------------
# Load Models
# -------------------------------
models = {
    "CatBoost": load("models/catboost_model.joblib"),
    "LightGBM": load("models/lightgbm_model.joblib"),
    "ExtraTrees": load("models/extratrees_model.joblib"),
    "GradientBoosting": load("models/gradientboosting_model.joblib"),
    "RandomForest": load("models/randomforest_model.joblib"),
    "IsolationForest": load("models/isolation_forest_model.joblib"),
}

autoencoder = load_model("models/autoencoder_model.h5", compile=False)
autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

# -------------------------------
# Evaluate Models
# -------------------------------
results = []

# Supervised Models
for name, model in models.items():
    if name != "IsolationForest":
        y_pred = model.predict(X)
    else:
        y_pred_raw = model.predict(X)
        y_pred = np.where(y_pred_raw == -1, 1, 0)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_pred)
    })

# Autoencoder (unsupervised)
reconstructions = autoencoder.predict(X)
mse = np.mean(np.square(X - reconstructions), axis=1)
threshold = np.percentile(mse[y_true == 0], 95)
y_pred = (mse > threshold).astype(int)

results.append({
    "Model": "Autoencoder",
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1 Score": f1_score(y_true, y_pred),
    "ROC AUC": roc_auc_score(y_true, y_pred)
})

# -------------------------------
# Final Comparison Table
# -------------------------------
df_results = pd.DataFrame(results)
df_sorted = df_results.sort_values(by="F1 Score", ascending=False)

print("\n📊 Model Performance Comparison:")
print(df_sorted.round(4))

# -------------------------------
# Best Performing Model
# -------------------------------
best = df_sorted.iloc[0]
print(f"\n🏆 Best Model Overall: {best['Model']}")
print(f"   ➤ Accuracy : {best['Accuracy']:.4f}")
print(f"   ➤ Precision: {best['Precision']:.4f}")
print(f"   ➤ Recall   : {best['Recall']:.4f}")
print(f"   ➤ F1 Score : {best['F1 Score']:.4f}")
print(f"   ➤ ROC AUC  : {best['ROC AUC']:.4f}")
