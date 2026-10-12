"""Football Lab V3.7 — client-side, instant six-tile navigation."""
import streamlit as st
from mobile_dashboard import render_home

st.set_page_config(
    page_title="Football Lab V3.7 · Fast Navigation",
    page_icon=":material/analytics:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
render_home()
