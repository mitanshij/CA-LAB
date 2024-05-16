import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the CSV file
data = pd.read_csv('BBC News Text.csv')

# Assuming 'text' is the column with the document text and 'label' is the column with the category
documents = data['text'].values
labels = data['category'].values

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Streamlit App
st.title("Text Classification")

# Text area for user input
user_input = st.text_area("Enter the text for classification:", "")

# Predict and display result
if st.button("Classify"):
    # Vectorize user input
    user_input_vectorized = vectorizer.transform([user_input])

    # Make prediction
    prediction = clf.predict(user_input_vectorized)

    # Display prediction
    st.write("Predicted Category:", prediction[0])
