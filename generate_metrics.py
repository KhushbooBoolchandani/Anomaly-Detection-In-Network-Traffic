import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from tensorflow.keras.models import load_model

# Load dataset
df = pd.read_csv("kddcup_clean.csv")
X = df.drop(columns=["label"])
y = df["label"]

# ------------------------- Supervised Models -------------------------
model_names = ["catboost", "lightgbm", "extratrees", "gradientboosting", "randomforest"]
metrics = []

for name in model_names:
    model = joblib.load(f"models/{name}_model.joblib")
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_pred)

    metrics.append({
        "Model": name.capitalize(),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": auc
    })

# ------------------------- Isolation Forest -------------------------
iso_model = joblib.load("models/isolation_forest_model.joblib")
y_pred_iso = iso_model.predict(X)
y_pred_iso = np.where(y_pred_iso == -1, 1, 0)

metrics.append({
    "Model": "Isolation Forest",
    "Accuracy": accuracy_score(y, y_pred_iso),
    "Precision": precision_score(y, y_pred_iso),
    "Recall": recall_score(y, y_pred_iso),
    "F1 Score": f1_score(y, y_pred_iso),
    "AUC": roc_auc_score(y, y_pred_iso)
})

# ------------------------- Autoencoder -------------------------
auto_model = load_model("models/autoencoder_model.h5")
recons = auto_model.predict(X)
mse = np.mean(np.square(X - recons), axis=1)
threshold = np.percentile(mse[y == 0], 95)
y_pred_auto = (mse > threshold).astype(int)

metrics.append({
    "Model": "Autoencoder",
    "Accuracy": accuracy_score(y, y_pred_auto),
    "Precision": precision_score(y, y_pred_auto),
    "Recall": recall_score(y, y_pred_auto),
    "F1 Score": f1_score(y, y_pred_auto),
    "AUC": roc_auc_score(y, y_pred_auto)
})

# ------------------------- Save to CSV -------------------------
metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("models/metrics.csv", index=False)

print("\n✅ metrics.csv saved with all model performance including unsupervised models.")
print(metrics_df.round(4))
