import streamlit as st
import joblib

# 1. Load the "Strict" Final Brain and Dictionary
# Make sure these filenames match exactly what you uploaded to GitHub!
model = joblib.load('dna_model_v5.pkl')
cv = joblib.load('vectorizer_v5.pkl')

st.set_page_config(page_title="DNA Promoter AI", page_icon="🧬")

st.title("🧬 Level 4: Strict DNA Detector")
st.write("This version is trained to ignore 'Junk DNA' like repeated sequences.")

# 2. The 6-mer function (Must match the training code)
def get_kmers(sequence, size=6):
    # .lower() ensures it isn't case-sensitive
    # .replace(" ", "") removes accidental spaces
    clean_seq = sequence.lower().replace(" ", "").strip()
    return [clean_seq[x:x+size] for x in range(len(clean_seq) - size + 1)]

user_seq = st.text_input("Enter DNA Sequence (e.g., A, T, G, C):", "")

if st.button("Deep AI Analysis"):
    if len(user_seq.strip()) < 6:
        st.error("Sequence too short. Please provide at least 6 nucleotides.")
    else:
        # Transform the input using the NEW dictionary
        words = ' '.join(get_kmers(user_seq))
        vectorized_data = cv.transform([words]).toarray()
        
        # Run the Prediction
        prediction = model.predict(vectorized_data)
        prob = model.predict_proba(vectorized_data)[0][1] * 100

        if prediction[0] == 1:
            st.success(f"✅ PROMOTER IDENTIFIED ({prob:.1f}% Match)")
            st.balloons()
        else:
            st.warning(f"❌ NON-PROMOTER ({100-prob:.1f}% Match)")
            st.info("The AI determined this sequence lacks the necessary biological 'words' to be a switch.")
