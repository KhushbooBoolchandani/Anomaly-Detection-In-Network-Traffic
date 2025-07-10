import streamlit as st

def show():
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    st.markdown("<h2 style='text-align:center;'>🔐 Login to Anomaly Detection System</h2>", unsafe_allow_html=True)

    # Already logged in
    if st.session_state.logged_in:
        st.success("✅ You are already logged in.")
        return

    # Login form message
    st.caption("Use your credentials to continue. If you’re a new user, refer to the documentation")

    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.success("✅ Login successful!")
        else:
            st.error("❌ Invalid username or password")
