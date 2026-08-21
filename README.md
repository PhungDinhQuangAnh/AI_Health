<h1 align="center">Banking Intelligence Platform</h1>

[![Streamlit App](https://img.shields.io/badge/Truy%20cập%20ứng%20dụng%20trực%20tuyến-Click%20here-brightgreen)](https://banking-intelligence-platform.streamlit.app/)

**Banking Intelligence Platform** là nền tảng tích hợp **thuật toán AI** với **mục tiêu** hỗ trợ doanh nghiệp:
- Phân khúc chủ thẻ tín dụng dựa trên hành vi tài chính, nhằm tăng trưởng doanh số & kích cầu chi tiêu.
- Thẩm định rủi ro, giảm tải thủ công, tự động duyệt hồ sơ vay vốn an toàn & từ chối hồ sơ vay vốn rủi ro cao.

---

## Giao diện demo

<p align="center">
  <img src="https://github.com/PhungDinhQuangAnh/ai-health/blob/main/Demo/demo1.png" alt="Giao diện demo 1">
  <img src="https://github.com/PhungDinhQuangAnh/ai-health/blob/main/Demo/demo2.png" alt="Giao diện demo 2">
  <img src="https://github.com/PhungDinhQuangAnh/ai-health/blob/main/Demo/demo3.png" alt="Giao diện demo 3">
  <img src="https://github.com/PhungDinhQuangAnh/ai-health/blob/main/Demo/demo4.png" alt="Giao diện demo 4">
</p>

---

## Mô hình & Dữ liệu

| Bài toán             | Dataset                                                                                          | Mô hình sử dụng   |
|---------------------|--------------------------------------------------------------------------------------------------|-------------------|
| Phân Khúc Khách Hàng Thẻ Tín Dụng| [Credit Card Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)                | K-Means |
| Thẩm Định Rủi Ro & Duyệt Hồ Sơ Vay Vốn | [Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) | XGBClassifier |

---

## Cấu trúc dự án
<pre>  
Banking_Intelligence_Platform/
├── app.py                      # Giao diện sidebar chọn mô hình
├── requirements.txt            # Danh sách thư viện cần cài
├── README.md                   # Tài liệu mô tả dự án
├── LICENSE                     # Giấy phép sử dụng
├── Demo/                     
|    ├── demo1.png              # Hình ảnh demo giao diện web
│    ├── demo2.png
│    ├── demo3.png
│    └── demo4.png
│
├── Credit_Card_Segmentation/
│    ├── credit_card_app.py            # Giao diện trang Phân Khúc Khách Hàng Thẻ Tín Dụng
│    ├── dataset/
│    │    └── credit_card_dataset.csv   # Bộ dữ liệu
│    ├── model/
│    │    ├── credit_card_model.py      # Code huấn luyện mô hình
│    │    ├── iqr_bounds.pkl            # IQR xử lý outlier đã lưu
│    │    ├── kmeans_normal.pkl         # K-Means cho tập Normal đã lưu
│    │    ├── kmeans_outlier.pkl        # K-Means cho tập Outlier đã lưu
│    │    ├── scaler_normal.pkl         # Scaler cho tập Normal đã lưu
│    │    └── scaler_outlier.pkl        # Scaler cho tập Outlier đã lưu
│    └── report/
│         ├── credit_card_segmented_tsne.csv    # Kết quả lưu nhãn từng dữ liệu và tọa độ biểu đồ không gian
│         ├── elbow_normal.png                  # Hình ảnh minh họa Elbow Method cho tập Normal
│         └── elbow_outlier.png                 # Hình ảnh minh họa Elbow Method cho tập Outlier
│
└── Loan_Default_Prediction/
     ├── loan_default_app.py           # Giao diện trang Thẩm Định Rủi Ro & Duyệt Hồ Sơ Vay Vốn
     ├── dataset/
     │    └── loan_default_dataset.csv  # Bộ dữ liệu 
     ├── model/
     │    ├── loan_default_model.py     # Code huấn luyện mô hình
     │    └── loan_default_model.pkl    # Mô hình đã lưu
     └── report/
          ├── loan_default_classification_report.csv   # Chỉ số đánh giá (Accuracy, Precision, Recall, F1)
          └── loan_default_confusion_matrix.jpg        
</pre>

---

## Công nghệ sử dụng

- **Ngôn ngữ:** Python
- **Web UI:** streamlit, html, css
- **Xử lý dữ liệu:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Trực quan hóa:** matplotlib, seaborn, plotly

---

## Cách chạy ứng dụng

### Truy cập ứng dụng online

🔗 https://banking-intelligence-platform.streamlit.app/
