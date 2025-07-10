
import streamlit as st

def show():
    st.markdown(
        "<h1 style='text-align:center; color:black; font-weight:800;'>🔐 Anomaly Detection System for Network Security</h1>", 
        unsafe_allow_html=True
    )

    st.markdown("""
    <div style='text-align:center; font-size:16px;'>
        A professional-grade application that uses advanced <b>Supervised</b> and <b>Unsupervised</b> Machine Learning<br>
        techniques to detect anomalies in network traffic using models such as <b>Isolation Forest</b>, 
        <b>Autoencoder</b>, <b>CatBoost</b>, and more.
    </div><br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#333;'>✨ Core Features</h4>", unsafe_allow_html=True)
        st.markdown("""
        - 📂 Upload & analyze network traffic data  
        - 🧠 Multiple ML models:
            - Isolation Forest  
            - Autoencoder  
            - CatBoost, LightGBM, ExtraTrees  
        - 🔁 Workflow: Upload → Predict → Evaluate → Visualize  
        - 📤 Export results & performance metrics  
        """)

    with col2:
        st.markdown("<div class='section-box'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#333;'>📊 Insights & Analysis</h4>", unsafe_allow_html=True)
        st.markdown("""
        - ✅ Confusion Matrix, ROC Curve  
        - 📈 Accuracy, Precision, Recall, F1-Score  
        - 🔐 Secure login & session  
        - 🖥️ Clean UI with Streamlit  
        """)

    with st.expander("🧭 How to Use the App"):
        st.markdown("""
        <ol>
            <li>🔑 <b>Login</b> first to access the system.</li>
            <li>📤 Upload a preprocessed CSV dataset.</li>
            <li>🚀 Go to <b>Prediction</b> to detect anomalies.</li>
            <li>📋 Check <b>Performance</b> for model accuracy & metrics.</li>
            <li>📊 View <b>Charts</b> for ROC curves, distributions, etc.</li>
            <li>🚪 <b>Logout</b> when done.</li>
        </ol>
        """, unsafe_allow_html=True)

    st.success("✅ Built using Streamlit and advanced Machine Learning techniques | " 
    "Designed for real-world deployment with a focus on performance, visual clarity, and anomaly detection excellence.")
