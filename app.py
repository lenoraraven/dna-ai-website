import streamlit as st
import joblib

# 1. Load the "V6" Balanced Brain and Dictionary
model = joblib.load('dna_model_6mer.pkl')
cv = joblib.load('vectorizer_6.pkl')

st.set_page_config(page_title="DNA Promoter AI", page_icon="🧬")

st.title("🧬 Final: Goldilocks DNA Detector")
st.write("Balanced version: Optimized to find real signals while ignoring junk.")

# 2. The 4-mer function (Crucial: Must be size=4 now!)
def get_kmers(sequence, size=4):
    clean_seq = sequence.lower().replace(" ", "").strip()
    if len(clean_seq) < size:
        return []
    return [clean_seq[x:x+size] for x in range(len(clean_seq) - size + 1)]

user_seq = st.text_input("Enter DNA Sequence (57 chars):", "")

if st.button("Deep AI Analysis"):
    # Pre-processing
    kmers = get_kmers(user_seq)
    
    if not kmers:
        st.error("Sequence too short!")
    else:
        words = ' '.join(kmers)
        vectorized_data = cv.transform([words]).toarray()
        
        # Prediction logic
        prediction = model.predict(vectorized_data)
        prob = model.predict_proba(vectorized_data)[0][1] * 100

        if prediction[0] == 1:
            st.success(f"✅ PROMOTER IDENTIFIED ({prob:.1f}% Match)")
            st.balloons()
        else:
            st.warning(f"❌ NON-PROMOTER ({100-prob:.1f}% Match)")
