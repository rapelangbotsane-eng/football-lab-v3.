"""Football Lab V3.3 — mobile-first weekly football research assistant."""
import streamlit as st
from mobile_dashboard import render_home

st.set_page_config(
    page_title="Football Lab · Weekly Scout",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)
render_home()
