import streamlit as st
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Thêm thư mục hiện tại vào path để import preprocessing
sys.path.append(os.getcwd())
try:
    from preprocessing import prepare_data
except ImportError:
    # Fallback nếu chạy trực tiếp trong thư mục con
    sys.path.append(os.path.join(os.getcwd(), 'StreamlitApp'))
    from preprocessing import prepare_data

# Cấu hình trang
st.set_page_config(
    page_title="Toxic Tweet Detector",
    page_icon="🚫",
    layout="wide"
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
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Functions ---

@st.cache_data
def load_data():
    """Load dataset from csv"""
    try:
        # Thử các đường dẫn có thể
        possible_paths = [
            'data/Data_finish.csv',
            'StreamlitApp/data/Data_finish.csv',
            os.path.join(os.path.dirname(__file__), 'data/Data_finish.csv')
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                break
        
        if df is None:
            st.error("Không tìm thấy file dữ liệu (Data_finish.csv).")
            return None
            
        # Loại bỏ giá trị null
        df = df.dropna(subset=['tweet_ok'])
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu: {e}")
        return None

def train_model(df, model_type, params):
    """Huấn luyện mô hình dựa trên lựa chọn"""
    
    X = df['tweet_ok']
    y = df['class']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorization
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    model = None
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif model_type == "LightGBM":
        model = lgb.LGBMClassifier(
            n_estimators=params['n_estimators'],
            num_leaves=params['num_leaves'],
            learning_rate=params['learning_rate'],
            random_state=42
        )
        
    if model:
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return model, tfidf, acc, report
    return None, None, 0, None

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Model Configuration")

model_option = st.sidebar.selectbox(
    "Chọn mô hình:",
    ("XGBoost", "LightGBM")
)

params = {}

if model_option == "XGBoost":
    st.sidebar.subheader("Tham số XGBoost")
    params['n_estimators'] = st.sidebar.slider("Number of Estimators", 50, 500, 100, 50)
    params['max_depth'] = st.sidebar.slider("Max Depth", 3, 20, 6)
    params['learning_rate'] = st.sidebar.number_input("Learning Rate", 0.01, 0.5, 0.1, 0.01)

elif model_option == "LightGBM":
    st.sidebar.subheader("Tham số LightGBM")
    params['n_estimators'] = st.sidebar.slider("Number of Estimators", 50, 500, 100, 50)
    params['num_leaves'] = st.sidebar.slider("Num Leaves", 10, 100, 31)
    params['learning_rate'] = st.sidebar.number_input("Learning Rate", 0.01, 0.5, 0.1, 0.01)

if st.sidebar.button("🚀 Huấn luyện mô hình"):
    with st.spinner("Đang tải dữ liệu và huấn luyện..."):
        df = load_data()
        if df is not None:
            model, vectorizer, acc, report = train_model(df, model_option, params)
            
            # Lưu vào session state
            st.session_state['model'] = model
            st.session_state['vectorizer'] = vectorizer
            st.session_state['accuracy'] = acc
            st.session_state['report'] = report
            st.session_state['model_name'] = model_option
            
            st.sidebar.success(f"Huấn luyện xong! Accuracy: {acc:.4f}")

# --- Main Page ---
st.title("🚫 Toxic Tweet Classification")
st.write("Hệ thống phân loại tweet độc hại sử dụng Machine Learning.")

# Hiển thị thông tin mô hình hiện tại
if 'model' in st.session_state:
    st.markdown("---")
    st.subheader(f"Mô hình đang sử dụng: **{st.session_state['model_name']}**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Độ chính xác (Accuracy)</h3>
            <h2>{st.session_state['accuracy']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        report = st.session_state['report']
        macro_f1 = report['macro avg']['f1-score']
        st.markdown(f"""
        <div class="metric-card">
            <h3>F1-Score (Macro)</h3>
            <h2>{macro_f1:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📝 Kiểm tra Tweet")
    
    user_input = st.text_area("Nhập nội dung tweet tiếng Anh:", height=100, placeholder="Type something here...")

    if st.button("Phân tích"):
        if user_input.strip() == "":
            st.warning("Vui lòng nhập nội dung!")
        else:
            with st.spinner("Đang xử lý..."):
                # Preprocessing
                processed_text = prepare_data(user_input)
                
                # Show processed text
                with st.expander("Xem dữ liệu sau khi tiền xử lý"):
                    st.write(f"`{processed_text}`")

                if processed_text.strip() == "":
                    st.warning("Văn bản sau khi xử lý bị rỗng.")
                else:
                    # Prediction
                    input_vector = st.session_state['vectorizer'].transform([processed_text])
                    prediction = st.session_state['model'].predict(input_vector)[0]
                    
                    try:
                        probability = st.session_state['model'].predict_proba(input_vector)[0]
                        confidence = probability[prediction]
                    except:
                        confidence = 0.0 # Một số model cấu hình đặc biệt có thể không có predict_proba ngay
                        
                    # Display
                    if prediction == 1: # Giả sử 1 là Toxic
                        st.markdown(f'<div class="result-box toxic">TOXIC (Độc hại) <br> Độ tin cậy: {confidence:.2%}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="result-box normal">NORMAL (Bình thường) <br> Độ tin cậy: {confidence:.2%}</div>', unsafe_allow_html=True)

else:
    st.info("👈 Vui lòng chọn mô hình và nhấn **'Huấn luyện mô hình'** ở thanh bên trái để bắt đầu.")
    st.markdown("""
    ### Hướng dẫn:
    1. Chọn loại mô hình (XGBoost hoặc LightGBM).
    2. Điều chỉnh các tham số (Hyperparameters) nếu muốn.
    3. Nhấn nút Huấn luyện.
    4. Nhập văn bản để kiểm tra kết quả.
    """)