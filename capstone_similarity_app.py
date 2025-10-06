import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

st.set_page_config(page_title="IBU Capstone Similarity Checker", layout="centered")
st.title("🎓 IBU Capstone Similarity Checker")
st.write("Type your capstone title and check how similar it is to past projects.")

# Sidebar – upload CSV option
st.sidebar.header("📁 Data Source")
uploaded = st.sidebar.file_uploader("Upload CSV of past capstone titles", type=["csv"])
st.sidebar.caption("CSV should have a column named 'Project Title' or 'title'.")

# Default titles
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

def load_titles():
    if uploaded is None:
        return pd.DataFrame({"Project Title": default_titles})
    try:
        df = pd.read_csv(uploaded)
        cols = [c for c in df.columns if c.strip().lower() in ["project title", "title", "project_title", "project"]]
        if not cols:
            st.error("CSV must include a 'Project Title' or 'title' column.")
            return pd.DataFrame({"Project Title": default_titles})
        df = df.rename(columns={cols[0]: "Project Title"})
        df = df.dropna(subset=["Project Title"])
        df["Project Title"] = df["Project Title"].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return pd.DataFrame({"Project Title": default_titles})

df_titles = load_titles()
st.caption(f"Loaded {len(df_titles)} past titles.")
if len(df_titles) > 0:
    st.write("Example titles:", df_titles["Project Title"].head(5).tolist())

new_title = st.text_input("Enter your Capstone Title:")
top_k = st.slider("How many similar titles to show?", min_value=1, max_value=10, value=3)

if st.button("Check Similarity"):
    if not new_title.strip():
        st.warning("Please enter a title first.")
    elif len(df_titles) == 0:
        st.error("No titles available to compare. Please upload a valid CSV.")
    else:
        past_titles = df_titles["Project Title"].fillna("").astype(str).tolist()
        vectorizer = TfidfVectorizer(stop_words="english")
        all_titles = past_titles + [new_title]
        tfidf_matrix = vectorizer.fit_transform(all_titles)
        similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])[0]
        order = np.argsort(similarity_scores)[::-1][:top_k]

        st.subheader("📊 Top Similar Titles")
        for rank, idx in enumerate(order, start=1):
            st.write(f"{rank}. **{past_titles[idx]}** — {similarity_scores[idx]*100:.2f}%")

        best = similarity_scores[order[0]] * 100 if len(order) else 0
        if best > 80:
            st.error("⚠️ High overlap! Please consider modifying your topic (best match > 80%).")
        elif best > 50:
            st.warning("⚠️ Medium overlap detected (best match between 50–80%). Review your title focus.")
        else:
            st.success("✅ Low overlap (best match < 50%). Your topic seems unique!")

st.markdown("---")
st.caption("Prototype — IBU Capstone Similarity Checker | Built by Shakksha")
