import streamlit as st
import pandas as pd
import pickle
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load Metrics
@st.cache_data
def load_metrics():
    try:
        with open('models/metrics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Load Model & Vectorizer
@st.cache_resource
def load_resources(model_filename):
    try:
        # Load Vectorizer
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Load Model
        with open(f'models/{model_filename}', 'rb') as f:
            model = pickle.load(f)
            
        return vectorizer, model
    except Exception as e:
        return None, None

# --- SIDEBAR ---
st.sidebar.title("🤖 Model Selector")
metrics_data = load_metrics()

# Filter out Random Forest
metrics_data = [m for m in metrics_data if m['Model'] != 'Random Forest']

if not metrics_data:
    st.error("Metrics file not found. Please run training script first.")
    st.stop()

# Create dictionary for mapping display name to filename
model_map = {item['Model']: item['Filename'] for item in metrics_data}
model_names = list(model_map.keys())

selected_model_name = st.sidebar.selectbox(
    "Choose a model for prediction:",
    model_names
)

selected_filename = model_map[selected_model_name]

# Load selected resources
vectorizer, model = load_resources(selected_filename)

if model:
    st.sidebar.success(f"Loaded: {selected_model_name}")
else:
    st.sidebar.error("Failed to load model files.")
    st.stop()

# --- MAIN PAGE ---
st.title("🛡️ English Comment Classification")
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
                    # LinearSVC doesn't have predict_proba by default
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
    st.markdown("#### 📊 Current Model Performance")
    # Find metrics for selected model
    current_metrics = next((item for item in metrics_data if item['Model'] == selected_model_name), None)
    if current_metrics:
        st.metric("Accuracy", f"{current_metrics['Accuracy']:.2%}")
        st.metric("F1-Score", f"{current_metrics['F1-Score']:.2%}")
        st.metric("Training Time", f"{current_metrics['Time (s)']} s")

# --- COMPARISON SECTION ---
st.markdown("---")
st.header("📈 Model Comparison")

df_metrics = pd.DataFrame(metrics_data)

# Table
st.dataframe(df_metrics[['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'Time (s)']].style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score'], color='#d1e7dd'), use_container_width=True)

# Chart
st.subheader("Accuracy Comparison")
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=df_metrics, x='Accuracy', y='Model', palette='viridis', ax=ax)
plt.xlim(0.8, 1.0) # Zoom in to see differences
for i, v in enumerate(df_metrics['Accuracy']):
    ax.text(v, i, f" {v:.2%}", va='center')
st.pyplot(fig)
