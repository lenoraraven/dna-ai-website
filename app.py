import streamlit as st
import joblib

# Load the NEW stable files
model = joblib.load('stable_model.pkl')
cv = joblib.load('stable_vectorizer.pkl')

st.title("🧬 DNA Promoter Detector (V7-Stable)")

def get_kmers(sequence, size=3):
    clean = sequence.lower().replace(" ", "").strip()
    return [clean[x:x+size] for x in range(len(clean) - size + 1)]

user_seq = st.text_input("Enter 57-nucleotide sequence:", "")

if st.button("Analyze"):
    if len(user_seq.strip()) < 50:
        st.error("Please enter a full sequence.")
    else:
        words = ' '.join(get_kmers(user_seq))
        vec = cv.transform([words]).toarray()
        
        # Get probability
        prob = model.predict_proba(vec)[0] # [Prob_Negative, Prob_Positive]
pos_score = prob[1] * 100

# NEW BALANCED THRESHOLD (50%)
if pos_score >= 50:
    st.success(f"✅ PROMOTER IDENTIFIED ({pos_score:.1f}% Match)")
    st.balloons()
else:
    st.warning(f"❌ NON-PROMOTER ({100 - pos_score:.1f}% Match)")
