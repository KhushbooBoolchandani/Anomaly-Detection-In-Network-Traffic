import streamlit as st
from streamlit_option_menu import option_menu

# Page config
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
)

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize login session
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Login", "Upload", "Prediction", "Performance", "Charts", "Logout"],
        icons=["house", "person", "cloud-upload", "activity", "bar-chart-line", "pie-chart", "box-arrow-right"],
        menu_icon="clipboard-data",
        default_index=0,
    )

# -------------------------
# Page Routing
# -------------------------

# Home page (no login required)
if selected == "Home":
    import pages.Home as Home
    Home.show()

# Login page
elif selected == "Login":
    import pages.Login as Login
    Login.show()

# Logout
elif selected == "Logout":
    st.session_state.logged_in = False
    st.success("✅ You have successfully logged out.")

# Protected pages
elif selected in ["Upload", "Prediction", "Performance", "Charts"]:
    if not st.session_state.logged_in:
        st.warning("🔐 Access denied. Please login to continue.")
        st.info("➡️ Use the **Login** option from the sidebar.")
    else:
        if selected == "Upload":
            import pages.Upload as Upload
            Upload.show()

        elif selected == "Prediction":
            import pages.Prediction as Prediction
            Prediction.show()

        elif selected == "Performance":
            import pages.Performance as Performance
            Performance.show()

        elif selected == "Charts":
            import pages.Charts as Charts
            Charts.show()

