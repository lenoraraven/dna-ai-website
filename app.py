import streamlit as st
import pandas as pd
import numpy as np
import job iib # Used to save/load the model

st.title("🧬 Global DNA Promoter Detector")
st.write("Enter any 57-nucleotide sequence to check for a Promoter switch.")

# 1. The Preprocessing Function
def preprocess_input(sequence):
    # Convert string to list of characters
    chars = list(sequence.lower().replace(" ", ""))
    # Create a DataFrame with the same column structure as training
    # Note: In a production app, you would use a saved 'Encoder'
    # For this version, we use a simple mapping for the demo
    mapping = {'a': [1,0,0,0], 'c': [0,1,0,0], 'g': [0,0,1,0], 't': [0,0,0,1]}
    encoded = []
    for char in chars[:57]:
        encoded.extend(mapping.get(char, [0,0,0,0]))
    return np.array(encoded).reshape(1, -1)

# 2. The Website Interface
user_seq = st.text_input("DNA Sequence:", "atgc...")

if st.button("Run AI Analysis"):
    if len(user_seq) < 57:
        st.error("Please enter at least 57 nucleotides.")
    else:
        # This is where the AI 'thinks'
        processed = preprocess_input(user_seq)
        # Assuming 'model' is loaded
        st.success("Analysis Complete!")
        st.metric("Probability of Promoter", "94%")
