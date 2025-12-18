import streamlit as st
import pickle
import os
import sys

# Thêm thư mục hiện tại vào path để import preprocessing
sys.path.append(os.getcwd())
from preprocessing import prepare_data

# Cấu hình trang
st.set_page_config(
    page_title="Toxic Tweet Detector",
    page_icon="🚫",
    layout="centered"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        background-color: #ffffff;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
    }
    .toxic {
        background-color: #ffcccc;
        color: #d8000c;
        border: 2px solid #d8000c;
    }
    .normal {
        background-color: #dff0d8;
        color: #3c763d;
        border: 2px solid #3c763d;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề
st.title("🚫 Toxic Tweet Classification")
st.write("Nhập nội dung tweet tiếng Anh bên dưới để kiểm tra xem nó có độc hại (Toxic) hay bình thường (Normal).")

# Load model & vectorizer
@st.cache_resource
def load_artifacts():
    model_path = os.path.join(os.path.dirname(__file__), 'models/rf_model.pkl')
    vect_path = os.path.join(os.path.dirname(__file__), 'models/tfidf_vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vect_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Không tìm thấy file model. Vui lòng chạy script huấn luyện trước!")
    st.stop()

# Input
user_input = st.text_area("Nội dung Tweet:", height=150, placeholder="Type something here...")

if st.button("Phân tích"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung!")
    else:
        with st.spinner("Đang xử lý..."):
            # 1. Preprocessing
            processed_text = prepare_data(user_input)
            
            # Hiển thị text sau khi xử lý (để debug/minh họa)
            with st.expander("Xem dữ liệu sau khi tiền xử lý"):
                st.write(f"`{processed_text}`")

            # 2. Vectorization
            if processed_text.strip() == "":
                st.warning("Văn bản sau khi xử lý bị rỗng (do chỉ chứa ký tự đặc biệt hoặc stopword).")
            else:
                input_vector = vectorizer.transform([processed_text])

                # 3. Prediction
                prediction = model.predict(input_vector)[0]
                probability = model.predict_proba(input_vector)[0]

                # 4. Display Result
                if prediction == 1:
                    st.markdown(f'<div class="result-box toxic">TOXIC (Độc hại) <br> Độ tin cậy: {probability[1]:.2%}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-box normal">NORMAL (Bình thường) <br> Độ tin cậy: {probability[0]:.2%}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Developed with Random Forest & Streamlit")
