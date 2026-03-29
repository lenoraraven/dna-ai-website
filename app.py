import streamlit as st
import joblib

# Load the FINAL files
model = joblib.load('dna_final_model.pkl')
cv = joblib.load('dna_final_vocab.pkl')

st.title("🧬 Professional DNA Promoter AI")

def get_kmers(sequence, size=3):
    clean = sequence.lower().replace(" ", "").strip()
    return [clean[x:x+size] for x in range(len(clean) - size + 1)]

user_input = st.text_input("Paste DNA sequence here:", "")

if st.button("Analyze Sequence"):
    if len(user_input) < 20:
        st.error("Sequence too short!")
    else:
        words = ' '.join(get_kmers(user_input))
        vec = cv.transform([words]).toarray()
        
        # Get Probability
        prob = model.predict_proba(vec)[0][1] * 100
        
        # The 'Bio-Standard' 35% Threshold
        if prob >= 35:
            st.success(f"✅ PROMOTER DETECTED ({prob:.1f}% Signal)")
            st.balloons()
        else:
            st.warning(f"❌ NON-PROMOTER ({100 - prob:.1f}% Match)")
