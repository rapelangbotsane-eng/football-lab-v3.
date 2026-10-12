"""Football Lab V3.5 — verified three-by-two mobile navigation build."""
import streamlit as st
from mobile_dashboard import render_home

st.set_page_config(
    page_title="Football Lab V3.5 · Weekly Scout",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
render_home()
