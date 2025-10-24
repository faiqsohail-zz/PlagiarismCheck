import streamlit as st
import pdfplumber
import re
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import numpy as np

# -------------------------------
# Helper functions
# -------------------------------
def clean_text(text):
    """Lowercase and remove non-alphabetic characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def jaccard_similarity(text1, text2):
    """Compute Jaccard similarity between two texts."""
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform([text1, text2])
    return jaccard_score(X.toarray()[0], X.toarray()[1])

# -------------------------------
# Load reference data
# -------------------------------
with open("reference_texts.pkl", "rb") as f:
    reference_texts = pickle.load(f)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("📄 PDF Similarity Checker (Using Pickle References)")
st.markdown("Upload a PDF to check its similarity with 4 fixed reference PDFs.")

uploaded_file = st.file_uploader("📤 Upload PDF", type="pdf")

# -------------------------------
# Main logic
# -------------------------------
if uploaded_file:
    with st.spinner("Extracting and comparing..."):
        # Extract text using pdfplumber (pure Python, no compile)
        uploaded_text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    uploaded_text += text

        uploaded_text = clean_text(uploaded_text)

        similarities = []
        for name, ref_text in reference_texts.items():
            sim = jaccard_similarity(uploaded_text, ref_text)
            similarities.append((name, sim))

        st.subheader("📊 Similarity Results")
        for name, sim in similarities:
            st.write(f"**{name}:** {sim*100:.2f}%")

        st.bar_chart(np.array([sim for _, sim in similarities]))
else:
    st.info("⬆️ Please upload a PDF file to start comparison.")
