import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="IBU Capstone Similarity & Dashboard", layout="centered")
st.title("🎓 IBU Capstone Project Portal")

# -------------------------------
# SIDEBAR: GOOGLE SHEET CONNECTION
# -------------------------------
st.sidebar.header("📄 Google Sheet (Live Data)")

# Default / fallback dataset
default_titles = [
    "AI and Blockchain in Supply Chain Management",
    "Machine Learning Applications in Healthcare",
    "Digital Transformation in Banking Sector",
    "Sustainability Practices in Retail Industry",
    "Customer Data Analytics using Python",
    "Impact of Social Media on Consumer Behavior",
    "Smart City Development using IoT and AI",
    "E-commerce Strategies for Small Businesses",
    "Cybersecurity Challenges in Cloud Computing",
    "Automation and Robotics in Manufacturing"
]

# Google Sheet link (replace with yours)
sheet_url = st.sidebar.text_input(
    "Paste your Google Sheet CSV link here:",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQQAoO_eJz3idWJSu4PVCzgBgEw_NDFwFgNiAOAGoQSvkvTMdZyxwVHiHSuPseZEvpoH6Z8SKDF077b/pub?output=csv"
)

# Load data from Google Sheet
try:
    df_titles = pd.read_csv(sheet_url)
    df_titles = df_titles.rename(columns=lambda x: x.strip())
    st.sidebar.success("✅ Loaded data from Google Sheet successfully!")
except Exception as e:
    st.sidebar.warning("⚠️ Could not load Google Sheet. Using default dataset instead.")
    df_titles = pd.DataFrame({"Project Title": default_titles})

# Ensure structure
if "Project Title" not in df_titles.columns:
    st.error("❌ Your Google Sheet must include a column named 'Project Title'.")
    st.stop()

# -------------------------------
# TABS: APP SECTIONS
# -------------------------------
tab1, tab2 = st.tabs(["🔍 Similarity Checker", "📊 Faculty Dashboard"])

# ==============================================================
# TAB 1: SIMILARITY CHECKER
# ==============================================================
with tab1:
    st.subheader("🔍 Check Your Capstone Title Similarity")
    st.caption(f"Loaded {len(df_titles)} past titles.")
    if len(df_titles) > 0:
        st.write("Example titles:", df_titles["Project Title"].head(5).tolist())

    new_title = st.text_input("Enter your Capstone Title:")
    top_k = st.slider("How many similar titles to show?", min_value=1, max_value=10, value=3)

    if st.button("Check Similarity"):
        if not new_title.strip():
            st.warning("Please enter a title first.")
        elif len(df_titles) == 0:
            st.error("No titles available to compare. Please check your data source.")
        else:
            past_titles = df_titles["Project Title"].fillna("").astype(str).tolist()

            # TF-IDF + Cosine Similarity
            vectorizer = TfidfVectorizer(stop_words="english")
            all_titles = past_titles + [new_title]
            tfidf_matrix = vectorizer.fit_transform(all_titles)
            similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
            order = np.argsort(similarity_scores)[::-1][:top_k]

            # Display results
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

            # Summary of overlap
            best = similarity_scores[order[0]] * 100 if len(order) else 0
            if best > 80:
                st.error("⚠️ High overlap! Please consider modifying your topic (best match > 80%).")
            elif best > 50:
                st.warning("⚠️ Medium overlap detected (best match between 50–80%). Review your title focus.")
            else:
                st.success("✅ Low overlap (best match < 50%). Your topic seems unique!")

# ==============================================================
# TAB 2: FACULTY DASHBOARD
# ==============================================================
with tab2:
    st.subheader("📊 Capstone Insights Dashboard")

    if len(df_titles) > 0:
        # Clean Year column
        df_titles["Year"] = pd.to_numeric(df_titles["Year"], errors="coerce")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Projects", len(df_titles))
        with col2:
            unique_supervisors = df_titles["Supervisor"].nunique() if "Supervisor" in df_titles.columns else 0
            st.metric("Total Supervisors", unique_supervisors)

        st.markdown("### 🎓 Projects per Program")
        if "Program" in df_titles.columns:
            program_counts = df_titles["Program"].value_counts()
            st.bar_chart(program_counts)
        else:
            st.info("No 'Program' column found in data.")

        st.markdown("### 📅 Projects per Year")
        if "Year" in df_titles.columns:
            year_counts = df_titles["Year"].value_counts().sort_index()
            st.bar_chart(year_counts)
        else:
            st.info("No 'Year' column found in data.")

        st.markdown("### 👩‍🏫 Top Supervisors by Project Count")
        if "Supervisor" in df_titles.columns:
            supervisor_counts = df_titles["Supervisor"].value_counts().head(10)
            st.bar_chart(supervisor_counts)
        else:
            st.info("No 'Supervisor' column found in data.")
    else:
        st.warning("⚠️ No data found to display dashboard insights.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Prototype — IBU Capstone Similarity & Insights Dashboard | Built by Shakksha")
