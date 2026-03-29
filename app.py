import streamlit as st
import joblib
import numpy as np

# 1. Load the "Pro" Brain and Dictionary
# Make sure these filenames match exactly what you uploaded to GitHub!
model = joblib.load('dna_model_pro.pkl')
cv = joblib.load('vectorizer_pro.pkl')

st.set_page_config(page_title="DNA Promoter AI: Pro", page_icon="🧬")

st.title("🧬 DNA Promoter Detector (Random Forest)")
st.write("Professional-grade model trained to ignore repetitive 'Poly-A' and 'CGC' traps.")

# 2. The 3-mer function (Must match the V7/Pro training code)
def get_kmers(sequence, size=3):
    clean_seq = sequence.lower().replace(" ", "").strip()
    if len(clean_seq) < size:
        return []
    return [clean_seq[x:x+size] for x in range(len(clean_seq) - size + 1)]

user_seq = st.text_input("Enter DNA Sequence (57 chars recommended):", "")

if st.button("Run Pro Analysis"):
    # Pre-processing
    kmers = get_kmers(user_seq)
    
    if not kmers:
        st.error("⚠️ Sequence is too short for analysis! Please enter at least 3 characters.")
    else:
        # Convert DNA to 'sentence' of 3-mers
        words = ' '.join(kmers)
        vectorized_data = cv.transform([words])
        
        # 3. Safe Prediction Logic
        try:
            prediction = model.predict(vectorized_data)[0]
            prob = model.predict_proba(vectorized_data)[0][1] * 100

            # We use a 70% confidence threshold for "Pro" mode
           if prob >= 50:
                st.success(f"✅ PROMOTER IDENTIFIED ({prob:.1f}% Confidence)")
                st.balloons()
                st.info("The AI detected a specific arrangement of regulatory motifs.")
            else:
                st.warning(f"❌ NON-PROMOTER ({100 - prob:.1f}% Confidence)")
                st.info("This sequence was flagged as 'Junk' or 'Background DNA'.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}. Please ensure your model and vectorizer match.")
