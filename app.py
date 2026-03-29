import streamlit as st
import joblib

# Load the V10 files
model = joblib.load('dna_v10_model.pkl')
cv = joblib.load('dna_v10_vocab.pkl')

st.title("🧬 DNA Promoter AI: V10")

user_input = st.text_input("Paste DNA sequence:", "")

if st.button("Analyze"):
    if len(user_input.strip()) < 20:
        st.error("Sequence too short for a reliable match.")
    else:
        # Clean the input
        clean_seq = user_input.lower().replace(" ", "").strip()
        
        # The vectorizer now handles the 'char' analysis automatically
        vec = cv.transform([clean_seq]).toarray()
        
        # Get Probability
        prob = model.predict_proba(vec)[0][1] * 100
        
        # 50% Threshold for a balanced majority vote
        if prob >= 50:
            st.success(f"✅ PROMOTER ({prob:.1f}% Match)")
            st.balloons()
        else:
            st.warning(f"❌ NON-PROMOTER ({100 - prob:.1f}% Match)")
