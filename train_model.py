import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Tạo thư mục models nếu chưa có
if not os.path.exists('StreamlitApp/models'):
    os.makedirs('StreamlitApp/models')

print("--- Đang đọc dữ liệu ---")
# Đọc file Data_finish.csv (đã tiền xử lý tweet_ok)
# Điều chỉnh đường dẫn tương đối tùy thuộc vào nơi bạn chạy script
try:
    df = pd.read_csv('Project/data/Data_finish.csv')
except FileNotFoundError:
    # Thử đường dẫn thay thế nếu chạy từ root
    df = pd.read_csv('../Project/data/Data_finish.csv')

# Loại bỏ giá trị null
df = df.dropna(subset=['tweet_ok'])

X = df['tweet_ok']
y = df['class']

print(f"Tổng số mẫu: {len(df)}")

# Chia tập dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vector hóa
print("--- Đang Vector hóa (TF-IDF) ---")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Huấn luyện Random Forest (Tham số đã tối ưu)
print("--- Đang huấn luyện Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_tfidf, y_train)

# Đánh giá
y_pred = rf_model.predict(X_test_tfidf)
print(f"Độ chính xác: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Lưu model
print("--- Đang lưu mô hình ---")
with open('StreamlitApp/models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('StreamlitApp/models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Hoàn tất! Model đã được lưu tại StreamlitApp/models/")
