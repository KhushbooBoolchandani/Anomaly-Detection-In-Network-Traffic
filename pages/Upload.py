import streamlit as st
import pandas as pd

def show():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("🔐 Access denied. Please login to continue.")
        st.stop()

    st.markdown("<h2 style='text-align:center;'>📁 Upload Network Traffic CSV</h2>", unsafe_allow_html=True)

    st.write("""
Upload your **preprocessed CSV dataset** for anomaly detection analysis.

🔹 The file should follow the structure of the cleaned KDD Cup 1999 dataset  
🔹 Example file name: `kddcup_clean.csv`  
🔹 Ensure the dataset has no missing values and is label-encoded if needed  
    """)

    # Optional helpful caption
    st.caption("ℹ️ You can find sample format details in the project documentation")

    # File upload widget
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read uploaded file
            df = pd.read_csv(uploaded_file)
            st.session_state['uploaded_data'] = df
            st.success("✅ File uploaded successfully!")

            # Preview first 10 rows
            st.write("### 🧾 Preview of Uploaded Dataset (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)

            # Optional full view
            if st.checkbox("🔍 Show full dataset"):
                st.write("### 📊 Full Uploaded Dataset")
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
