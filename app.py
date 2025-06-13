import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Ticket Prioritization", layout="centered")

st.title("🚨 Ticket Prioritization App")
st.write("Enter your support ticket text below to predict its priority level (P1, P2, P3).")

user_input = st.text_area("📝 Enter Support Ticket Description", height=200)

if st.button("🔍 Predict Priority"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input and predict
        X_input = vectorizer.transform([user_input])
        pred_class = model.predict(X_input)[0]

        # Display result
        st.success(f"✅ Predicted Priority: **{pred_class}**")
