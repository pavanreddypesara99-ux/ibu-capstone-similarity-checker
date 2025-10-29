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
st.caption(f"Loaded {len(df_titles)} past titles.")
if len(df_titles) > 0:
st.write("Example titles:", df_titles["Project Title"].head(5).tolist())

new_title = st.text_input("Enter your Capstone Title:")
top_k = st.slider("How many similar titles to show?", 1, 10, 3)

if st.button("Check Similarity"):
if not new_title.strip():
st.warning("Please enter a title first.")
elif len(df_titles) == 0:
st.error("No titles available to compare. Please check your data source.")
else:
past_titles = df_titles["Project Title"].fillna("").astype(str).tolist()
vectorizer = TfidfVectorizer(stop_words="english")
all_titles = past_titles + [new_title]
tfidf_matrix = vectorizer.fit_transform(all_titles)
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
order = np.argsort(similarity_scores)[::-1][:top_k]

st.subheader("📊 Top Similar Titles")
for rank, idx in enumerate(order, start=1):
row = df_titles.iloc[idx]
project = str(row.get("Project Title", "N/A"))
score = similarity_scores[idx] * 100

st.markdown(f"**{rank}. {project}** — {score:.2f}% similarity")
st.caption(
f"👩‍🎓 **Student:** {row.get('Student Name', 'N/A')}  \n"
f"🎓 **Program:** {row.get('Program', 'N/A')} | 📅 **Year:** {row.get('Year', 'N/A')}  \n"
f"👩‍🏫 **Supervisor:** {row.get('Supervisor', 'N/A')}"
)
st.write("---")

best = similarity_scores[order[0]] * 100 if len(order) else 0
if best > 80:
st.error("⚠️ High overlap! Please consider modifying your topic (best match > 80%).")
elif best > 50:
st.warning("⚠️ Medium overlap detected (best match between 50–80%). Review your title focus.")
else:
st.success("✅ Low overlap (best match < 50%). Your topic seems unique!")

# ==============================================================
# TAB 2 — FACULTY DASHBOARD
# ==============================================================
with tab2:
st.subheader("📊 Capstone Insights Dashboard")

if len(df_titles) > 0:
df_titles["Year"] = pd.to_numeric(df_titles["Year"], errors="coerce")

col1, col2 = st.columns(2)
with col1:
st.metric("Total Projects", len(df_titles))
with col2:
unique_supervisors = df_titles["Supervisor"].nunique() if "Supervisor" in df_titles.columns else 0
st.metric("Total Supervisors", unique_supervisors)

if "Program" in df_titles.columns:
st.markdown("### 🎓 Projects per Program")
st.bar_chart(df_titles["Program"].value_counts())

if "Year" in df_titles.columns:
st.markdown("### 📅 Projects per Year")
st.bar_chart(df_titles["Year"].value_counts().sort_index())

if "Supervisor" in df_titles.columns:
st.markdown("### 👩‍🏫 Top Supervisors by Project Count")
st.bar_chart(df_titles["Supervisor"].value_counts().head(10))
else:
st.warning("⚠️ No data found to display dashboard insights.")

# ==============================================================
# TAB 3 — NEW PROJECT SUBMISSION
# ==============================================================
with tab3:
st.subheader("📝 Submit New Capstone Project")
st.write("Fill out the form below to add a new project to the IBU Capstone database.")

with st.form("submission_form"):
student_name = st.text_input("👩‍🎓 Student Name")
program = st.text_input("🎓 Program")
year = st.number_input("📅 Year", min_value=2020, max_value=2030, value=2025)
supervisor = st.text_input("👩‍🏫 Supervisor Name")
project_title = st.text_input("💡 Project Title")

submitted = st.form_submit_button("Submit Project")

if submitted:
if not all([student_name, program, year, supervisor, project_title]):
st.warning("⚠️ Please fill in all fields.")
else:
# ✅ Use your deployed Google Apps Script URL
script_url = "https://script.google.com/macros/s/AKfycbyk0hNT0WSx6xETPZGJ84QTq6-NjCQQJ-UgUQ9kNGH3xCuNvzjOkhCXDmuNQChVvfXD3A/exec"

payload = {
"student_name": student_name,
"program": program,
"year": year,
"supervisor": supervisor,
"project_title": project_title,
}

try:
response = requests.post(script_url, json=payload)
if response.status_code == 200:
st.success("✅ Project added successfully!")
st.balloons()
else:
st.error(f"❌ Error submitting data (status {response.status_code}).")
except Exception as e:
st.error(f"⚠️ Failed to connect to Google Script: {e}")

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown("---")
st.caption("Prototype — IBU Capstone Simi
st.caption("Prototype — IBU Capstone Similarity & Insights Dashboard | Built by Shakksha")
