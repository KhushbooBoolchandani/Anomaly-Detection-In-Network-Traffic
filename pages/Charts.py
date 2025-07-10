import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from tensorflow.keras.models import load_model

# Define the main function to show charts
def show():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("🔐 Access denied. Please login to continue.")
        st.stop()
    st.title("📊 Visual Analytics & Model Insights")

    # Load uploaded dataset from session
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset first using the 'Upload' page.")
        return

    df = st.session_state['uploaded_data']
    if df is None or df.empty:
        st.warning("⚠️ Uploaded dataset is empty.")
        return

    # Sidebar chart selection
    st.subheader("📌 Select a Chart Type")
    chart_type = st.selectbox(
        "Choose the type of chart you want to view:",
        [
            "📉 Feature Importance (Supervised Models)",
            "📊 Anomaly Class Distribution",
            "📈 Autoencoder MSE Distribution"
        ]
    )

    st.markdown("---")

    # -------------------------
    # 1. Feature Importance
    # -------------------------
    if chart_type == "📉 Feature Importance (Supervised Models)":
        model_choice = st.selectbox("Choose a model:", ["CatBoost", "LightGBM", "ExtraTrees"])
        try:
            model_path = f"models/{model_choice.lower()}_model.joblib"
            model = load(model_path)
            X = df.drop(columns=["label"], errors="ignore")
            importances = model.feature_importances_

            feature_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).head(15)

            st.markdown(f"### 🔍 Top 15 Feature Importances — {model_choice}")
            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_df, palette="viridis")
            plt.title(f"{model_choice} - Top Features")
            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")

    # -------------------------
    # 2. Anomaly Class Distribution
    # -------------------------
    elif chart_type == "📊 Anomaly Class Distribution":
        if "label" not in df.columns:
            st.error("❌ No 'label' column found in the dataset.")
        else:
            st.markdown("### 🧮 Anomaly vs Normal Distribution")

            counts = df['label'].value_counts().rename(index={0: 'Normal', 1: 'Anomaly'})
            col1, col2 = st.columns(2)

            with col1:
                plt.figure(figsize=(5, 4))
                sns.barplot(x=counts.index, y=counts.values, palette="pastel")
                plt.title("Class Count")
                plt.ylabel("Samples")
                st.pyplot(plt.gcf())
                plt.clf()

            with col2:
                plt.figure(figsize=(5, 4))
                plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
                plt.title("Class Distribution")
                st.pyplot(plt.gcf())
                plt.clf()

    # -------------------------
    # 3. Autoencoder MSE Distribution
    # -------------------------
    elif chart_type == "📈 Autoencoder MSE Distribution":
        try:
            st.markdown("### ⚙️ MSE (Reconstruction Error) Distribution — Autoencoder")

            model = load_model("models/autoencoder_model.h5")
            X = df.drop(columns=["label"], errors="ignore")

            reconstructions = model.predict(X)
            mse = np.mean(np.square(X - reconstructions), axis=1)
            threshold = np.percentile(mse, 95)

            plt.figure(figsize=(10, 5))
            sns.histplot(mse, bins=50, kde=True, color="skyblue")
            plt.axvline(x=threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
            plt.title("MSE Distribution for Autoencoder")
            plt.xlabel("Reconstruction Error (MSE)")
            plt.ylabel("Frequency")
            plt.legend()
            st.pyplot(plt.gcf())
            plt.clf()

        except Exception as e:
            st.error(f"❌ Autoencoder chart failed: {e}")
