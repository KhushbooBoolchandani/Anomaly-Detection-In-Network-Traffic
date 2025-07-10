import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from joblib import dump

# ------------------- Load Preprocessed Data -------------------
df = pd.read_csv("kddcup_clean.csv")
X = df.drop(columns=["label"])
y_true = df["label"]

# Train on normal data only (unsupervised)
X_train = X[y_true == 0]
X_test = X
y_test = y_true

# ------------------- Autoencoder Model -------------------
input_dim = X_train.shape[1]
encoding_dim = 20

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation="relu")(input_layer)
decoded = Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
from tensorflow.keras.losses import MeanSquaredError
autoencoder.compile(optimizer='adam', loss=MeanSquaredError())

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = autoencoder.fit(
    X_train, X_train,
    epochs=100,
    batch_size=256,
    shuffle=True,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ------------------- Save Model -------------------
os.makedirs("models", exist_ok=True)
autoencoder.save("models/autoencoder_model.h5")

# ------------------- Prediction & Evaluation -------------------
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.square(X_test - reconstructions), axis=1)

threshold = np.percentile(mse[y_test == 0], 95)
y_pred = (mse > threshold).astype(int)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# ------------------- Confusion Matrix -------------------
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"])
plt.title("Autoencoder Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("static/autoencoder_confusion_matrix.png")
plt.show()
plt.close()

# ------------------- ROC Curve -------------------
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve - Autoencoder")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("static/autoencoder_roc_curve.png")
plt.show()
plt.close()

print(f"\nROC AUC Score: {roc_auc:.4f}")
print("Autoencoder model and visualizations saved successfully.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Additional Evaluation Metrics ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n--- Additional Evaluation Metrics ---")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC      : {roc_auc:.4f}")