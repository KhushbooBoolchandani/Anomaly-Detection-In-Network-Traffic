'''import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import load
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

def show():
    st.title("🔍 Prediction - Anomaly Detection")
    st.markdown("Use the trained models to predict anomalies on new data. Upload your CSV and select a model.")

    # File Upload
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload a preprocessed CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("❌ Uploaded file is empty.")
            return

        st.success("✅ File uploaded successfully.")
        st.dataframe(df.head(15), use_container_width=True)

        # Split if label exists
        if 'label' in df.columns:
            X_test = df.drop(columns=['label'])
            y_true = df['label']
            has_label = True
        else:
            X_test = df
            y_true = None
            has_label = False

        st.markdown("---")
        st.subheader("🧠 Select a Model")

        model_choice = st.selectbox("Choose a trained model to make predictions", (
            "Isolation Forest", "Autoencoder",
            "CatBoost", "LightGBM", "GradientBoosting", "ExtraTrees", "RandomForest"
        ))

        if st.button("🚀 Run Prediction"):
            try:
                # Load and Predict
                if model_choice == "Autoencoder":
                    model = load_model("models/autoencoder_model.h5")
                    reconstructions = model.predict(X_test)
                    mse = np.mean(np.square(X_test - reconstructions), axis=1)
                    threshold = np.percentile(mse, 95)
                    y_pred = (mse > threshold).astype(int)
                else:
                    model = load(f"models/{model_choice.lower()}_model.joblib")
                    y_pred = model.predict(X_test)

                # Display results
                st.markdown("### 📊 Prediction Results")
                result_df = X_test.copy()
                result_df["Prediction"] = y_pred
                if has_label:
                    result_df["Actual"] = y_true

                st.dataframe(result_df.head(20), use_container_width=True)

                # Metrics if label exists
                if has_label:
                    st.markdown("### 📈 Classification Report")
                    report = classification_report(y_true, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose().round(4)
                    st.dataframe(report_df)

                    # Confusion Matrix
                    st.markdown("### 🔍 Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                                xticklabels=["Normal", "Anomaly"], 
                                yticklabels=["Normal", "Anomaly"])
                    plt.xlabel("Predicted")
                    plt.ylabel("Actual")
                    plt.title(f"{model_choice} - Confusion Matrix")
                    st.pyplot(fig)

                st.success("✅ Prediction completed successfully!")

            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

def show():
    st.markdown("<h2 class='page-title'>🔎 Real-Time Prediction</h2>", unsafe_allow_html=True)

    # Check if uploaded data exists
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ Please upload your dataset first from the 'Upload' page.")
        return

    df = st.session_state['uploaded_data']
    st.markdown("Use the form below to simulate a real-time anomaly prediction:")

    # Select a row to predict
    row_index = st.selectbox("🔢 Select a record to predict from uploaded data:", range(len(df)))

    sample = df.iloc[[row_index]].copy()
    st.write("📋 **Selected Input Sample:**")
    st.dataframe(sample, use_container_width=True)

    # Choose model
    model_choice = st.selectbox("🤖 Choose a trained model for prediction:", [
        "Autoencoder", "Isolation_Forest", "CatBoost", "LightGBM", "Extra_Trees", "GradientBoosting", "RandomForest"
    ])

    if st.button("🚀 Predict"):
        try:
            X_test = sample.values
            prediction = None

            if model_choice == "Autoencoder":
                model = load_model("models/autoencoder_model.h5")
                reconstructions = model.predict(X_test)
                mse = np.mean(np.square(X_test - reconstructions), axis=1)
                threshold = np.percentile(mse, 95)
                prediction = (mse > threshold).astype(int)[0]

            elif model_choice == "Isolation_Forest":
                model = joblib.load("models/isolation_forest_model.joblib")
                prediction = model.predict(X_test)[0]
                prediction = 1 if prediction == -1 else 0  # Convert -1 to 1 for anomaly

            else:
                model_file = f"models/{model_choice.lower()}_model.joblib"
                if not os.path.exists(model_file):
                    st.error(f"❌ Model file not found: {model_file}")
                    return
                model = joblib.load(model_file)
                prediction = model.predict(X_test)[0]

            # Display Result Table
            st.markdown("### 🧾 Prediction Result")
            result_table = sample.copy()
            result_table["Model Used"] = model_choice
            result_table["Prediction Result"] = ["🔴 Anomaly" if prediction == 1 else "🟢 Normal"]

            st.dataframe(result_table.T, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
'''
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import load
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def show():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("🔐 Access denied. Please login to continue.")
        st.stop()
    st.title("🔍 Prediction - Anomaly Detection")
    st.markdown("Use the trained models to predict anomalies on uploaded data.")

    # Check if data is already uploaded
    if 'uploaded_data' not in st.session_state:
        st.warning("⚠️ Please upload a dataset from the Upload tab first.")
        return

    df = st.session_state['uploaded_data']

    if df.empty:
        st.error("❌ Uploaded dataset is empty.")
        return

    # Display the uploaded data
    st.markdown("### 📄 Preview of Uploaded Dataset")
    st.dataframe(df.head(15), use_container_width=True)

    # Handle label presence
    has_label = 'label' in df.columns
    if has_label:
        X_test = df.drop(columns=['label'])
        y_true = df['label']
    else:
        X_test = df
        y_true = None

    st.markdown("---")
    st.subheader("🧠 Select a Model")

    model_choice = st.selectbox("Choose a model for prediction", (
        "Isolation Forest", "Autoencoder",
        "CatBoost", "LightGBM", "GradientBoosting", "ExtraTrees", "RandomForest"
    ))

    if st.button("🚀 Run Prediction"):
        try:
            # Load and predict
            if model_choice == "Autoencoder":
                model = load_model("models/autoencoder_model.h5")
                reconstructions = model.predict(X_test)
                mse = np.mean(np.square(X_test - reconstructions), axis=1)
                threshold = np.percentile(mse, 95)
                y_pred = (mse > threshold).astype(int)
            else:
                model_filename = model_choice.lower().replace(" ", "") + "_model.joblib"
                model = load(f"models/{model_filename}")
                y_pred = model.predict(X_test)

            # Display results
            st.markdown("### 📊 Prediction Results")
            result_df = X_test.copy()
            result_df["Prediction"] = y_pred
            if has_label:
                result_df["Actual"] = y_true

            st.dataframe(result_df.head(20), use_container_width=True)

            # Show metrics if ground truth available
            if has_label:
                st.markdown("### 📈 Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose().round(4)
                st.dataframe(report_df)

                st.markdown("### 🔍 Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Normal", "Anomaly"],
                            yticklabels=["Normal", "Anomaly"])
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title(f"{model_choice} - Confusion Matrix")
                st.pyplot(fig)

                acc = accuracy_score(y_true, y_pred)
                st.markdown(f"### ✅ Accuracy: `{acc:.4f}`")

            st.success("✅ Prediction completed successfully!")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
