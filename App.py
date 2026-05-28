import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("Logistic_txt.pkl")
tfidf = joblib.load("tfidf.pkl")

emotion_map = {
    0: "Sadness 😔",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}

# Page Config
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# CSS
st.markdown("""
<style>
.stApp {
background: linear-gradient(135deg,#667eea,#764ba2);
color: white;
}

.result-box {
padding: 25px;
border-radius: 20px;
background: rgba(255,255,255,0.15);
backdrop-filter: blur(10px);
text-align: center;
font-size: 28px;
font-weight: bold;
margin-top:20px;
}

textarea {
border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    "<h1 style='text-align:center;'>😊 Emotion Detection App</h1>",
    unsafe_allow_html=True
)

st.write("Type your sentence below and detect emotion instantly")

# Input
text = st.text_area(
    "Enter your text",
    placeholder="Example: I am feeling happy today..."
)

col1, col2 = st.columns(2)

# Predict
with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# Clear
with col2:
    clear_btn = st.button("🗑 Clear", use_container_width=True)

if clear_btn:
    st.session_state.history = []
    st.rerun()

if predict_btn:

    if text.strip():

        vector = tfidf.transform([text])

        pred = model.predict(vector)[0]
        emotion = emotion_map[pred]

        confidence = model.predict_proba(vector).max() * 100

        # save history
        st.session_state.history.append({
            "Text": text,
            "Emotion": emotion
        })

        st.balloons()

        st.markdown(
            f"""
            <div class='result-box'>
                {emotion}<br><br>
                Confidence: {confidence:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(int(confidence))

        # Download
        st.download_button(
            "📥 Download Result",
            f"Text: {text}\nPredicted Emotion: {emotion}",
            file_name="emotion_result.txt"
        )

    else:
        st.warning("Please enter text first")

# History
if st.session_state.history:
    st.subheader("📜 Prediction History")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)