import streamlit as st
import joblib
from preprocess import clean_text

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection")
st.title("📰 Fake News Detection System")

text = st.text_area("Paste news article or headline here")

if st.button("Check News"):
    if not text.strip():
        st.warning("Please enter text")
    else:
        if len(text.split()) < 15:
            st.warning("⚠️ Short text may reduce accuracy")

        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        proba = model.predict_proba(vec)[0]
        confidence = max(proba) * 100
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success(f"✅ Real News ({confidence:.2f}% confidence)")
        else:
            st.error(f"🚨 Fake News ({confidence:.2f}% confidence)")

        st.info(
            "This system detects linguistic patterns, not factual correctness."
        )
