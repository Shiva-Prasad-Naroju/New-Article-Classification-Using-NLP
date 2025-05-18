import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from collections import Counter
import re

# --- Load model and vectorizer ---
model = joblib.load("log_reg_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Label map ---
label_map = {
    1: ("🌎 World", "World news and international events."),
    2: ("🏅 Sports", "Sports news, events, and results."),
    3: ("💼 Business", "Business, finance, and economic news."),
    4: ("🔬 Sci/Tech", "Science and technology updates.")
}

# --- Sidebar: Project Info & Sample Input ---
st.sidebar.title("📰 News Topic Classifier")
st.sidebar.markdown("""
A powerful ML app to classify news articles into topics and analyze your text.

**How to use:**
- Enter a news headline and description.
- Click 'Classify' to see the predicted topic, model confidence, and text analytics.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Sample Input:**")
st.sidebar.write("Title: NASA’s Perseverance Rover Discovers Signs of Ancient Life on Mars")
st.sidebar.write("Description: Scientists are excited after the rover’s instruments detected organic molecules in Martian rock samples, suggesting the possibility of past microbial life.")

# --- Main Page ---
st.title("📰 News Topic Classifier & Text Analyzer")

# --- Highlighted Input Section ---
with st.container():
    st.markdown("## ✍️ Enter News Article Details")
    st.info("Please provide a news headline and a short description below, then click **Classify**.")
    title = st.text_input("News Title", value="", placeholder="e.g. NASA’s Perseverance Rover Discovers Signs of Ancient Life on Mars")
    description = st.text_area("News Description", value="", placeholder="e.g. Scientists are excited after the rover’s instruments detected organic molecules in Martian rock samples, suggesting the possibility of past microbial life.")

def preprocess_text(text):
    # Simple preprocessing: lowercase, remove special chars
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

if st.button("Classify"):
    text = title + " " + description
    clean_text = preprocess_text(text)
    X = vectorizer.transform([clean_text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    topic, topic_desc = label_map.get(pred, ("Unknown", "No description available."))

    # --- Detailed Output ---
    st.markdown(f"## 🎯 Predicted Topic: {topic}")
    st.markdown(f"**{topic_desc}**")
    st.markdown(f"**Confidence:** `{proba[pred-1]*100:.2f}%`")

    # --- Interactive Probability Bar Chart ---
    class_names = [label_map[i][0] for i in range(1, 5)]
    colors = ['#636EFA' if i == pred-1 else '#B6E880' for i in range(4)]
    fig = go.Figure(go.Bar(
        x=proba,
        y=class_names,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.2f}%" for p in proba],
        textposition='auto'
    ))
    fig.update_layout(
        title="Model Confidence for Each Topic",
        xaxis_title="Probability",
        yaxis_title="Topic",
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Text Analytics ---
    st.markdown("### 📝 Text Analytics")
    tokens = clean_text.split()
    word_count = len(tokens)
    unique_words = len(set(tokens))
    top_keywords = [word for word, count in Counter(tokens).most_common(5)]
    avg_word_length = np.mean([len(word) for word in tokens]) if tokens else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Words", word_count)
    col2.metric("Unique Words", unique_words)
    col3.metric("Avg Word Length", f"{avg_word_length:.1f}")
    col4.metric("Top Keywords", ", ".join(top_keywords))

    with st.expander("🔍 See Preprocessed Text"):
        st.code(clean_text, language="text")
    
    st.info("Try more news samples or modify your input for different results!")

st.markdown("---")
st.markdown("Made with using Streamlit & Plotly")
