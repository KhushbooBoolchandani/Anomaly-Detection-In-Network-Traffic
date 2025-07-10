import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from joblib import dump

# Load preprocessed dataset
df = pd.read_csv("kddcup_clean.csv")
X = df.drop(columns=["label"])
y_true = df["label"]

# Train Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# Predict: IsolationForest outputs -1 for anomaly, 1 for normal
y_pred_raw = model.predict(X)
y_pred = np.where(y_pred_raw == -1, 1, 0)  # Convert to 1=anomaly, 0=normal

import os
os.makedirs("models", exist_ok=True)

# Save model
dump(model, "models/isolationforest_model.joblib")

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.title("Isolation Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

import os
os.makedirs("static", exist_ok=True)

plt.savefig("static/isolation_forest_confusion_matrix.png")
plt.show()
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve - Isolation Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("static/isolation_forest_roc_curve.png")
plt.show()
plt.close()

print(f"\nROC AUC Score: {roc_auc:.4f}")
print("Model and visualizations saved successfully.")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Additional Evaluation Metrics ---
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n--- Additional Evaluation Metrics ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {roc_auc:.4f}")
