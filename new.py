import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st

# Import the preprocess module or define your preprocessing functions here
# import preprocess

st.title("Phishing Email Detector")

user_input = st.text_input("Enter the email you want to check", "")
process_button = st.button("Process Data")

# Load your TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')  
model = joblib.load('phishing_model.pkl')  

if process_button:
    email_text = user_input

    preprocessed_text = tfidf_vectorizer.transform([email_text])

    # Make a prediction for the sample email
    phishing_status = model.predict(preprocessed_text)

    if phishing_status[0] == 1:
        st.write("The sample email is predicted to be phishing.")
    else:
        st.write("The sample email is predicted to be legitimate.")
