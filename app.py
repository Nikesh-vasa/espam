import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.set_page_config(page_title="Email Spam Detector")
st.title(" Enron Email Spam Detector")
st.write("Paste your email content below to check if it's spam or not.")

# Text input
email_text = st.text_area("Enter Email Content:", height=300)

# Predict button
if st.button("Check Email"):
    if not email_text.strip():
        st.warning(" Please enter some email content.")
    else:
        # Vectorize and predict
        vectorized_input = vectorizer.transform([email_text])
        prediction = model.predict(vectorized_input)

        # Display result
        if prediction[0] == 1:
            st.error(" This email is **SPAM**.")
        else:
            st.success(" This email is **NOT spam**.")
