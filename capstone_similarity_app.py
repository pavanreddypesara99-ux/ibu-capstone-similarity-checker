import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="IBU Capstone Similarity Checker", layout="centered")
st.title("🎓 IBU Capstone Similarity Checker")
st.write("Type your capstone title and check how similar it is to past projects.")

# -------------------------------
# SIDEBAR: GOOGLE SHEET CONNECTION
# -------------------------------
st.sidebar.header("📄 Google Sheet (Live Data)")

# Fallback dataset (in case Google Sheet isn't reachable)
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

# Google Sheet link (replace with your published CSV link)
sheet_url = st.sidebar.text_input(
    "Paste your Google Sheet CSV link here:",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQQAoO_eJz3idWJSu4PVCzgBgEw_NDFwFgNiAOAGoQSvkvTMdZyxwVHiHSuPseZEvpoH6Z8SKDF077b/pub?output=csv"
)

# Try loading the Google Sheet
try:
    df_titles = pd.read_csv(sheet_url)
    df_titles = df_titles.rename(columns=lambda x: x.strip())
    st.sidebar.success("✅ Loaded data from Google Sheet successfully!")
except Exception as e:
    st.sidebar.warning("⚠️ Could not load Google Sheet. Using default dataset instead.")
    df_titles = pd.DataFrame({"Project Title": default_titles})

# Check data structure
if "Project Title" not in df_titles.columns:
    st.error("❌ Your data must have a column named 'Project Title'. Please fix it in the Google Sheet.")
    st.stop()

st.caption(f"Loaded {len(df_titles)} past titles.")
if len(df_titles) > 0:
    st.write("Example titles:", df_titles["Project Title"].head(5).tolist())

# -------------------------------
# INPUT SECTION
# -------------------------------
new_title = st.text_input("Enter your Capstone Title:")
top_k = st.slider("How many similar titles to show?", min_value=1, max_value=10, value=3)

# -------------------------------
# SIMILARITY CALCULATION
# -------------------------------
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

        # -------------------------------
        # DISPLAY RESULTS
        # -------------------------------
        st.subheader("📊 Top Similar Titles")
        for rank, idx in enumerate(order, start=1):
            project = past_titles[idx]
            score = similarity_scores[idx] * 100
            st.write(f"{rank}. **{project}** — {score:.2f}%")
            
            # Optional: show details if your sheet has columns like Program, Year, etc.
            if "Program" in df_titles.columns:
                details = df_titles.iloc[idx]
                st.caption(f"📘 Program: {details.get('Program', 'N/A')} | 👩‍🏫 Supervisor: {details.get('Supervisor', 'N/A')} | 📅 Year: {details.get('Year', 'N/A')}")

        best = similarity_scores[order[0]] * 100 if len(order) else 0
        if best > 80:
            st.error("⚠️ High overlap! Please consider modifying your topic (best match > 80%).")
        elif best > 50:
            st.warning("⚠️ Medium overlap detected (best match between 50–80%). Review your title focus.")
        else:
            st.success("✅ Low overlap (best match < 50%). Your topic seems unique!")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Prototype — IBU Capstone Similarity Checker | Built by Shakksha")
