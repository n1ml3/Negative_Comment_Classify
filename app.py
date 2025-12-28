import streamlit as st
import pickle
import os
import sys

# Ensure preprocessing import works
sys.path.append(os.getcwd())
try:
    from preprocessing import prepare_data
except ImportError:
    # Fallback if running inside subdirectory
    sys.path.append(os.path.join(os.getcwd(), 'StreamlitApp'))
    from preprocessing import prepare_data

# Page Configuration
st.set_page_config(
    page_title="English Comment Classifier",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .result-card {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .toxic {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #c62828;
    }
    .non-toxic {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #2e7d32;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Vectorizer
@st.cache_resource
def load_resources():
    try:
        # Load Vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load Model (SVM Only)
        with open('models/svm_linearsvc.pkl', 'rb') as f:
            model = pickle.load(f)
            
        return vectorizer, model
    except Exception as e:
        return None, None

# Load resources
vectorizer, model = load_resources()

if not model:
    st.error("Failed to load SVM model. Please ensure 'models/svm_linearsvc.pkl' exists.")
    st.stop()

# --- MAIN PAGE ---
st.title("🛡️ English Comment Classification (SVM)")
st.markdown("### Detect Toxic Comments using Machine Learning")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("#### 📝 Enter Comment")
    user_input = st.text_area("Type your english comment here...", height=150)
    
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Processing..."):
                # 1. Preprocess
                processed_text = prepare_data(user_input)
                
                # 2. Vectorize
                input_vector = vectorizer.transform([processed_text])
                
                # 3. Predict
                prediction = model.predict(input_vector)[0]
                
                # 4. Probabilities (if supported)
                confidence = None
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(input_vector)[0]
                        confidence = probs[prediction]
                    except:
                        pass
                elif hasattr(model, "decision_function"):
                    # LinearSVC doesn't have predict_proba by default, but we can use decision_function for a score if needed
                    # For now, we'll skip confidence for LinearSVC unless calibrated
                    pass

                # 5. Display Result
                st.markdown("---")
                if prediction == 1:
                    conf_str = f"({confidence:.1%} confidence)" if confidence else ""
                    st.markdown(f"""
                        <div class="result-card toxic">
                            <h2>⚠️ TOXIC DETECTED</h2>
                            <p>This comment is classified as toxic/offensive. {conf_str}</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    conf_str = f"({confidence:.1%} confidence)" if confidence else ""
                    st.markdown(f"""
                        <div class="result-card non-toxic">
                            <h2>✅ NON-TOXIC</h2>
                            <p>This comment is clean. {conf_str}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with st.expander("See Processed Text (Internal)"):
                    st.code(processed_text)

with col2:
    st.markdown("#### ℹ️ Model Information")
    st.info("Using **Support Vector Machine (LinearSVC)** for high-speed text classification.")
