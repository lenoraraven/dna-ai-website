import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the "Brain" you uploaded
model = joblib.load('dna_model.pkl')

st.title("🧬 Global DNA Promoter Detector")
st.write("Enter a 57-nucleotide sequence to check for a Promoter switch.")

# 2. The Translator Function
def preprocess_input(sequence):
    chars = list(sequence.lower().replace(" ", ""))
    mapping = {'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1]}
    encoded = []
    # Take only the first 57 characters
    for char in chars[:57]:
        encoded.extend(mapping.get(char, [0,0,0,0]))
    return np.array(encoded).reshape(1, -1)

# 3. The Website Interface
user_seq = st.text_input("DNA Sequence (A, T, G, C):", "tactagcaatacgctgcgc...")

if st.button("Run AI Analysis"):
    if len(user_seq.strip()) < 57:
        st.error(f"Sequence too short! You provided {len(user_seq)} chars, but we need 57.")
    else:
        processed = preprocess_input(user_seq)
        prediction = model.predict(processed)
        
        if prediction[0] == 1:
            st.success("✅ RESULT: This sequence is a PROMOTER (ON-Switch)")
            st.balloons()
        else:
            st.warning("❌ RESULT: This is NOT a Promoter")
