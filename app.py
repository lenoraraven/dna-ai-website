import streamlit as st
import joblib

# Load the V8 files
model = joblib.load('dna_brain_v8.pkl')
cv = joblib.load('vocab_v8.pkl')

st.title("🧬 DNA Promoter Detector (V8)")
st.write("Calibrated for the Lac Operon and biological switches.")

def get_kmers(sequence, size=3):
    clean = sequence.lower().replace(" ", "").strip()
    return [clean[x:x+size] for x in range(len(clean) - size + 1)]

user_input = st.text_input("Enter DNA Sequence:", "")

if st.button("Analyze"):
    if len(user_input) < 10:
        st.error("Sequence too short!")
    else:
        words = ' '.join(get_kmers(user_input))
        vec = cv.transform([words]).toarray()
        
        # Get Probability
        prob = model.predict_proba(vec)[0][1] * 100
        
        if prob >= 40:
            st.success(f"✅ PROMOTER ({prob:.1f}% Match)")
            st.balloons()
        else:
            st.warning(f"❌ NON-PROMOTER ({100 - prob:.1f}% Match)")
