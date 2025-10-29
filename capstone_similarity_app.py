import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------
st.set_page_config(page_title="IBU Capstone Project Portal", layout="centered")
st.title("🎓 IBU Capstone Project Portal")

# ---------------------------------------------
# SIDEBAR: GOOGLE SHEET CONNECTION
# ---------------------------------------------
st.sidebar.header("📄 Google Sheet (Live Data)")
sheet_url = st.sidebar.text_input(
    "Paste your Google Sheet CSV link here:",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQQAoO_eJz3idWJSu4PVCzgBgEw_NDFwFgNiAOAGoQSvkvTMdZyxwVHiHSuPseZEvpoH6Z8SKDF077b/pub?output=csv"
)

try:
    df_titles = pd.read_csv(sheet_url)
    df_titles = df_titles.rename(columns=lambda x: x.strip())
    st.sidebar.success("✅ Loaded data from Google Sheet successfully!")
except Exception as e:
    st.sidebar.warning("⚠️ Could not load Google Sheet. Using default dataset instead.")
    df_titles = pd.DataFrame({
        "Student Name": [],
        "Program": [],
        "Year": [],
        "Supervisor": [],
        "Project Title": []
    })

# ---------------------------------------------
# MAIN TABS
# ---------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Similarity Checker", "📊 Faculty Dashboard", "📝 Submit New Capstone"])

# ==============================================================  
# TAB 1 — SIMILARITY CHECKER  
# ==============================================================  
with tab1:
    st.subheader("🔍 Check Your Capstone Title Similarity")
    st.capt
