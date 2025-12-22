import streamlit as st
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Public Speaking Coach",
    layout="wide"
)

# ---------------- LOAD CSS ----------------
def load_css():
    css_path = Path(__file__).parent / "styles" / "style.css"
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Login"

# ================= NAVBAR =================
st.markdown("""
<div class="navbar">
    <div class="nav-inner">
        <div class="nav-logo"> AI Public Speaking Coach</div>
        <div class="nav-links">
            <button onclick="window.location.reload()">Home</button>
            <button onclick="window.location.reload()">Register</button>
            <button onclick="window.location.reload()">Login</button>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= HERO + LOGIN =================
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

left, right = st.columns([7, 5])


with left:
    st.markdown("""
    <div class="hero">
        <h1>Speak with Confidence</h1>
        <p>
            AI-powered public speaking coach using attention-based deep learning
            to analyze voice, facial expressions, and delivery.
        </p>
        <ul>
            <li>🎯 Personalized AI feedback</li>
            <li>📊 Performance tracking</li>
            <li>🎥 Video & audio analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)

    st.markdown("### Login to your account")

    st.text_input("Username", key="login_user")
    st.text_input("Password", type="password", key="login_pass")

    if st.button("Login", key="login_btn"):
        if st.session_state.login_user and st.session_state.login_pass:
            st.switch_page("pages/dashboard.py")
        else:
            st.error("Please enter username and password")


    st.markdown("<p class='login-footer'>New user? Register above</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ================= FOOTER =================
st.markdown("""
<div class="footer">
    © 2025 AI Public Speaking Coach • Final Year Project
</div>
""", unsafe_allow_html=True)
