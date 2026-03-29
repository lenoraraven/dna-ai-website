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
    if len(user_seq.strip()) < 10:
        st.error("Please enter a longer DNA sequence.")
    else:
        words = ' '.join(get_kmers(user_seq))
        vec = cv.transform([words]).toarray()
        
        # 1. Get the Hard Prediction (0 or 1)
        prediction = model.predict(vec)[0]
        
        # 2. Try to get Probability, otherwise use a fallback
        try:
            prob = model.predict_proba(vec)[0]
            pos_score = prob[1] * 100
        except:
            # Fallback if probability isn't available
            pos_score = 100.0 if prediction == 1 else 0.0

        # 3. Final Result Logic
        if prediction == 1:
            st.success(f"✅ PROMOTER IDENTIFIED ({pos_score:.1f}% Confidence)")
            st.balloons()
        else:
            st.warning(f"❌ NON-PROMOTER ({100 - pos_score:.1f}% Confidence)")
