# 🚫 Toxic Tweet Detector

Ứng dụng web sử dụng học máy (Machine Learning) để nhận diện và phân loại các Tweet tiếng Anh có nội dung độc hại (Toxic) hoặc bình thường (Normal).

## 📌 Mục đích
Dự án này là một phần của môn học **Khai phá dữ liệu (Data Mining)**. Mục tiêu chính là xây dựng một hệ thống phân loại văn bản tự động dựa trên thuật toán **Random Forest** và kỹ thuật vector hóa **TF-IDF**. Ứng dụng giúp người dùng nhanh chóng đánh giá mức độ độc hại của một nội dung văn bản trước khi đăng tải hoặc kiểm duyệt nội dung.

## 🛠 Công nghệ sử dụng
- **Ngôn ngữ:** Python 3.x
- **Thư viện Machine Learning:** Scikit-learn
- **Xử lý ngôn ngữ tự nhiên (NLP):** NLTK (Stopwords, Lemmatization)
- **Giao diện người dùng:** Streamlit
- **Lưu trữ mô hình:** Pickle

## ⚙️ Cài đặt

### 1. Yêu cầu hệ thống
Đảm bảo bạn đã cài đặt Python (phiên bản 3.10 trở lên).

### 2. Cài đặt thư viện
Mở terminal tại thư mục dự án và chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

### 3. Dữ liệu NLTK
Ứng dụng sẽ tự động tải các gói dữ liệu cần thiết (`stopwords`, `wordnet`) trong lần chạy đầu tiên.

## 🚀 Sử dụng

### Chạy ứng dụng trên môi trường Local
Để khởi động giao diện web, chạy lệnh sau:
```bash
streamlit run app.py
```
Giao diện sẽ tự động mở tại địa chỉ: `http://localhost:8501`

### Cách sử dụng
1. Nhập hoặc dán nội dung Tweet tiếng Anh vào ô văn bản.
2. Nhấn nút **"Phân tích"**.
3. Hệ thống sẽ hiển thị kết quả phân loại (Toxic/Normal) cùng với độ tin cậy của mô hình.

## 📁 Cấu trúc thư mục
- `app.py`: Mã nguồn chính của ứng dụng Streamlit.
- `preprocessing.py`: Các hàm tiền xử lý văn bản dùng chung.
- `models/`: Chứa mô hình đã huấn luyện (`.pkl`).
- `requirements.txt`: Danh sách các thư viện cần cài đặt.

## 🤝 Đóng góp
Các đóng góp cho dự án luôn được hoan nghênh. Bạn có thể:
1. Fork dự án này.
2. Tạo branch mới cho tính năng của bạn (`git checkout -b feature/AmazingFeature`).
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`).
4. Push lên branch (`git push origin feature/AmazingFeature`).
5. Tạo một Pull Request.

## 📄 Giấy phép
Dự án này được cấp phép theo Giấy phép **MIT**. Xem file `LICENSE` để biết thêm chi tiết.

---
**Developed by Nam Le** - *Data Mining Project*
