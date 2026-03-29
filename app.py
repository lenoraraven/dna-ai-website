import streamlit as st
import joblib
import numpy as np

# 1. Load the Brain AND the Dictionary
model = joblib.load('dna_model_kmer.pkl')
cv = joblib.load('vectorizer.pkl')

st.title("🧬 Super-Smarter DNA Detector")
st.write("This version uses **K-mer Counting** to find promoters anywhere in the sequence!")

# 2. Function to chop user input into words
def get_kmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

user_seq = st.text_input("Enter DNA Sequence:", "atgc...")

if st.button("Run AI Analysis"):
    # Preprocess the same way we trained
    words = ' '.join(get_kmers(user_seq))
    vectorized_data = cv.transform([words]).toarray()
    
    # Predict
    prediction = model.predict(vectorized_data)
    prob = model.predict_proba(vectorized_data)[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ PROMOTER DETECTED ({prob:.1f}% confidence)")
        st.balloons()
    else:
        st.warning(f"❌ NOT A PROMOTER ({100-prob:.1f}% confidence)")
