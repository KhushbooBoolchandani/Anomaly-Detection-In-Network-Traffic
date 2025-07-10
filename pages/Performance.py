import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def show():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("🔐 Access denied. Please login to continue.")
        st.stop()

    st.title("📊 Model Performance Comparison")
    st.markdown("Detailed view of all trained models' performance using multiple metrics.")

    try:
        df = pd.read_csv("models/metrics.csv")

        if df.empty:
            st.warning("⚠️ Metrics CSV is empty.")
            return

        st.markdown("### 📋 Metrics Table")
        st.dataframe(df, use_container_width=True)

        # Best model summary
        st.markdown("### 🏆 Best Model Summary")
        best = df.sort_values(by='F1 Score', ascending=False).iloc[0]
        st.success(f"Top Model: **{best['Model']}** with F1 Score: **{best['F1 Score']:.4f}**")

        # Bar charts with fixed size and rotated x-labels
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

        for i, metric in enumerate(metrics_to_plot):
            st.markdown(f"### 📊 {metric} Comparison")
            fig, ax = plt.subplots(figsize=(7, 4))  # Smaller and consistent size
            sns.barplot(x='Model', y=metric, data=df, palette=[colors[i]], ax=ax)

            ax.set_title(f"{metric} Comparison Across Models", fontsize=14)
            ax.set_xlabel("Model", fontsize=12)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xticklabels(df['Model'], rotation=20, ha='right', fontsize=10)  # Rotate for readability
            ax.set_ylim(0, 1)  # Ensure consistent scale for all charts

            st.pyplot(fig)

    except FileNotFoundError:
        st.error("❌ File 'models/metrics.csv' not found.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
