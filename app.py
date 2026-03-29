import streamlit as st
import joblib

# Load the Level 3 Brain and Dictionary
model = joblib.load('dna_model_6mer.pkl')
cv = joblib.load('vectorizer_6.pkl')

st.title("🧬 Level 3: DNA Hexamer Detector")
st.write("This version looks for 6-nucleotide biological signatures.")

def get_kmers(sequence, size=6):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

user_seq = st.text_input("Enter DNA Sequence (57 chars):", "")

if st.button("Deep AI Analysis"):
    if len(user_seq) < 6:
        st.error("Sequence too short for 6-mer analysis.")
    else:
        words = ' '.join(get_kmers(user_seq))
        vectorized_data = cv.transform([words]).toarray()
        
        prediction = model.predict(vectorized_data)
        prob = model.predict_proba(vectorized_data)[0][1] * 100

        if prediction[0] == 1:
            st.success(f"✅ PROMOTER IDENTIFIED ({prob:.1f}% Match)")
            st.balloons()
        else:
            st.warning(f"❌ NON-PROMOTER ({100-prob:.1f}% Match)")
