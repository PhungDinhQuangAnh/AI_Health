import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import streamlit.components.v1 as components

def run_credit_card_app():
    # --- Lấy các đường dẫn ---
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    IQR_BOUNDS_PATH = os.path.join(APP_DIR, "model", "iqr_bounds.pkl")
    KMEANS_NORMAL_PATH = os.path.join(APP_DIR, "model", "kmeans_normal.pkl")
    KMEANS_OUTLIER_PATH = os.path.join(APP_DIR, "model", "kmeans_outlier.pkl")
    SCALER_NORMAL_PATH = os.path.join(APP_DIR, "model", "scaler_normal.pkl")
    SCALER_OUTLIER_PATH = os.path.join(APP_DIR, "model", "scaler_outlier.pkl")
    CSV_PATH = os.path.join(APP_DIR, "report", "credit_card_segmented_tsne.csv")
    ELBOW_NORMAL_PATH = os.path.join(APP_DIR, "report", "elbow_normal.png")
    ELBOW_OUTLIER_PATH = os.path.join(APP_DIR, "report", "elbow_outlier.png")

    # --- Chuẩn bị thông tin các nhóm khách hàng ---
    STRATEGIES = {
        "Nhóm 1: Khách Hàng Thụ Động": {
            "short_label": "Nhóm 1", "old_id": "Normal_2", "icon": "snooze", "type": "Phổ thông",
            "color": "#10B981", "bg_light": "rgba(16, 185, 129, 0.12)",
            "desc": "Chiếm số đông nhưng rất ít khi phát sinh giao dịch quẹt thẻ chi tiêu mặc dù tài khoản luôn có số dư ổn định.",
            "action": "Gửi chiến dịch kích cầu tự động qua thông báo App: 'Quẹt thẻ chi tiêu từ 1 triệu đồng nhận ngay voucher mua sắm giảm 100k' vào khung giờ vàng cuối tuần."
        },
        "Nhóm 2: Tiêu Dùng Thông Minh": {
            "short_label": "Nhóm 2", "old_id": "Normal_1", "icon": "shopping_bag", "type": "Năng động",
            "color": "#3B82F6", "bg_light": "rgba(59, 130, 246, 0.12)",
            "desc": "Mức nợ tích lũy duy trì ở ngưỡng rất an toàn. Tần suất quẹt thẻ sắm sửa hàng hóa, đăng ký trả góp diễn ra cực kỳ đều đặn và khoa học.",
            "action": "Tặng đặc quyền hoàn tiền cố định (Cashback) từ 2% - 5% khi chi tiêu mua sắm, đồng thời chủ động gợi ý nâng hạn mức tín dụng trực tuyến."
        },
        "Nhóm 3: Vay Tiêu Dùng Nhỏ": {
            "short_label": "Nhóm 3", "old_id": "Normal_3", "icon": "payments", "type": "Dòng tiền",
            "color": "#8B5CF6", "bg_light": "rgba(139, 92, 246, 0.12)",
            "desc": "Lười quẹt thẻ tại các điểm bán hàng thông thường, tuy nhiên tần suất và giá trị rút tiền mặt trực tiếp tại các cây ATM lại rất lớn.",
            "action": "Gửi gợi ý chuyển đổi tính năng rút tiền mặt ngắn hạn sang sản phẩm 'Trả góp tiền mặt qua thẻ' với lãi suất cố định thấp để quản trị rủi ro."
        },
        "Nhóm 4: VIP Tiêu Dùng Cao Cấp": {
            "short_label": "Nhóm 4", "old_id": "Outlier_2", "icon": "military_tech", "type": "Thượng lưu",
            "color": "#D97706", "bg_light": "rgba(217, 119, 6, 0.12)",
            "desc": "Được cấp hạn mức lớn, quẹt món to thẳng tay cho các dịch vụ xa xỉ, giải trí cao cấp và luôn hoàn trả dư nợ sòng phẳng, đúng hạn.",
            "action": "Tích điểm thưởng nhân hệ số (X2, X3) để đổi quà tặng giá trị cao, phát hành thẻ đồng thương hiệu ưu đãi tại các trung tâm thương mại lớn."
        },
        "Nhóm 5: Siêu VIP Chi Tiêu Khủng": {
            "short_label": "Nhóm 5", "old_id": "Outlier_1", "icon": "diamond", "type": "Cá mập (Whale)",
            "color": "#DC2626", "bg_light": "rgba(220, 38, 38, 0.12)",
            "desc": "Nhóm khách hàng siêu cấp. Sức chi tiêu khổng lồ, tất toán dư nợ minh bạch tuyệt đối, đem lại nguồn lợi nhuận lớn nhất cho ngân hàng.",
            "action": "Cung cấp chuỗi đặc quyền tối thượng: Tặng thẻ đen quyền lực, miễn phí phòng chờ thương gia sân bay quốc tế, lượt chơi Golf và trợ lý VIP 1-1."
        },
        "Nhóm 6: Tín Dụng Rút Tiền Mặt Cao": {
            "short_label": "Nhóm 6", "old_id": "Outlier_3", "icon": "report_problem", "type": "Rủi ro cao",
            "color": "#EA580C", "bg_light": "rgba(234, 88, 12, 0.12)",
            "desc": "Dư nợ chạm trần, coi thẻ tín dụng như khoản vay thấu chi để liên tục rút tiền mặt. Đem lại doanh thu phí lớn nhưng tiềm ẩn rủi ro nợ xấu.",
            "action": "Thắt chặt kiểm soát điểm tín dụng chặt chẽ, áp trần giới hạn rút tiền mặt tối đa hàng tháng và hướng khách hàng sang các gói trả góp 0%."
        }  
    }
    REVERSE_MAP = {v["old_id"]: k for k, v in STRATEGIES.items()}

    COLOR_HTML_MAP = {k: f"<span style='color:{v['color']}; font-weight:800;'>{v['short_label']}</span>" for k, v in STRATEGIES.items()}
    COLOR_MAP_SHORT = {f"<span style='color:{v['color']}; font-weight:800;'>{v['short_label']}</span>": v["color"] for k, v in STRATEGIES.items()}

    # --- Tải dữ liệu ---
    @st.cache_data
    def load_data():
        df = pd.read_csv(CSV_PATH)
        df["Full_Nhóm"] = df["Cluster"].map(REVERSE_MAP).fillna(df["Cluster"])
        df["Nhóm"] = df["Full_Nhóm"].map(COLOR_HTML_MAP)
        return df
    df_final = load_data()

    st.markdown("""
    <style>
        /* Ép khung chứa chính luôn co giãn chuẩn */
        section[data-testid="stMain"] > div:first-child {
            width: 100% !important;
            max-width: 100% !important;
        }
    
        /* Đổi Container Query thành Media Query dựa trên màn hình */
        @media screen and (max-width: 992px) {
            /* Chuyển các cột Streamlit (st.columns) về dạng 1 cột dọc khi dưới 992px */
            div[data-testid="stHorizontal"] {
                flex-direction: column !important;
            }
            div[data-testid="stColumn"] {
                width: 100% !important;
                min-width: 100% !important;
            }
            
            /* Chỉnh lại Grid KPI / Cards của bạn */
            .card-grid {
                grid-template-columns: repeat(2, 1fr) !important;
            }
            .card-info__grid {
                grid-template-columns: repeat(1, 1fr) !important;
            }
        }
    
        @media screen and (max-width: 576px) {
            .card-grid {
                grid-template-columns: repeat(1, 1fr) !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # --- CSS Internal ---
    st.markdown("""
        <style>
            /* ----------------------------------------------------------- */
            /* CSS phần header và banner thông tin đầu trang */
            /* ----------------------------------------------------------- */

            .page-header {
                text-align: center;
                background: linear-gradient(135deg, #06B6D4 0%, #3B82F6 50%, #1E3A8A 100%); 
                -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; 
                font-weight: 800;
                font-size: clamp(24px, 3.5vw + 1rem, 38px) !important;
                margin-bottom: clamp(15px, 2vw, 25px);
            }
            .data-banner {
                display: flex;
                flex-direction: column;
                gap: clamp(10px, 1.2vw, 16px);
                margin-bottom: clamp(16px, 2vw, 25px);
            }
            .data-banner__card {
                display: flex;
                gap: clamp(12px, 1.5vw, 18px);
                align-items: center;
                background: #FFFFFF !important;
                border-top: 1px solid #E2E8F0 !important;
                border-bottom: 1px solid #E2E8F0 !important;
                border-left: 4px solid #2563EB !important; 
                border-right: 4px solid #2563EB !important; 
                border-radius: 16px !important;
                padding: clamp(12px, 1.5vw, 18px) clamp(16px, 2vw, 24px);
                box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04), 0 8px 16px -6px rgba(15, 23, 42, 0.04);
                transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            }
            .data-banner__card:hover {
                transform: translateY(-4px);
                border-left-color: #1D4ED8 !important;
                border-right-color: #1D4ED8 !important;
                border-top-color: #BFDBFE !important;
                border-bottom-color: #BFDBFE !important;
                box-shadow: 0 25px 35px -5px rgba(37, 99, 235, 0.15), 0 12px 20px -8px rgba(37, 99, 235, 0.12);
            }
            .data-banner__badge {
                display: flex;
                align-items: center;
                justify-content: center;
                min-width: clamp(36px, 4vw, 44px);
                height: clamp(36px, 4vw, 44px);
                border-radius: 12px;
                background: rgba(2, 132, 199, 0.08);
                color: #0284C7;
            }
            .data-banner__content {
                font-size: clamp(12px, 0.3vw + 0.7rem, 14px) !important;
                line-height: 1.6;
                color: #334155 !important;
            }
            .data-banner__highlight {
                color: #2563EB !important;
                font-weight: 700;
            }
            .data-banner__link {
                color: #1D4ED8 !important;
                text-decoration: none !important;
                font-weight: 700;
                border-bottom: 1.5px solid rgba(29, 78, 216, 0.2);
                transition: all 0.2s;
            }
            .data-banner__link:hover {
                color: #2563EB !important;
                border-bottom-color: #2563EB;
            }

            /* --- Tiêu đề mỗi section --- */
            .section {
                display: flex;
                gap: 8px;
                align-items: center;
                margin-bottom: clamp(10px, 1.2vw, 15px);
                margin-top: clamp(30px, 3vw, 35px);
            }
            .section__icon {
                color: #0284C7; 
                font-size: clamp(15px, 0.6vw + 1rem, 23px) !important; 
            }
            .section__title {
                font-size: clamp(16px, 1vw + 0.75rem, 20px) !important; 
                font-weight: 800; 
                color: #0F172A; 
            }

            /* ----------------------------------------------------------- */
            /* CSS Tab trực quan và Tab dự đoán */
            /* ----------------------------------------------------------- */

            /* --- Card KPI --- */
            .card-grid {
                display: grid !important;
                grid-template-columns: repeat(6, minmax(0, 1fr)) !important;
                gap: clamp(8px, 1vw, 12px) !important;
                width: 100% !important;
                margin-bottom: 1rem;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
            }
            .card-kpi {
                border: 2.5px solid var(--card-color) !important;
                border-radius: 16px !important;
                padding: clamp(8px, 1vw, 12px) !important;
                display: flex !important;
                flex-direction: column !important;
                justify-content: space-between !important;
                background: linear-gradient(180deg, #FFFFFF 60%, var(--card-bg-light) 100%) !important;
                box-sizing: border-box !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
                width: 100% !important;
                transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;     
            }
            .card-kpi__header {
                display: flex !important;
                align-items: center !important;
                gap: 4px !important;
                color: var(--card-color) !important;
                font-size: clamp(11px, 0.4vw + 0.6rem, 13px) !important;
                font-weight: 800 !important; 
                letter-spacing: 0.3px !important;
            }
            .card-kpi__icon {
                font-size: clamp(13px, 0.3vw + 0.7rem, 15px) !important;
            }
            .card-kpi__label {
                white-space: nowrap !important;
            }
            .card-kpi__num {
                font-size: clamp(18px, 1.2vw + 0.8rem, 25px) !important;
                font-weight: 800 !important;
                color: #0F172A !important;
                white-space: nowrap !important;
            }
            .card-kpi__text {
                font-size: clamp(10.5px, 0.3vw + 0.55rem, 12px) !important;
                font-weight: 800 !important;
                color: var(--card-color) !important;
                display: -webkit-box !important;
                -webkit-line-clamp: 2 !important;
                -webkit-box-orient: vertical !important;
                overflow: hidden !important;
                min-height: clamp(26px, 2.5vw, 31px) !important; 
            }
            .card-kpi__footer {
                display: flex !important;
                align-items: center !important;
                gap: 6px !important;
                margin-top: 2px !important;
            }
            .card-kpi__percentage-text {
                font-size: clamp(10px, 0.2vw + 0.55rem, 11px) !important;
                font-weight: 800 !important;
                color: #475569 !important;
                min-width: 32px !important;
            }
            .card-kpi__progress-bar {
                flex: 1 !important;
                height: 6px !important;
                background-color: #E2E8F0 !important;
                border-radius: 10px !important;
                overflow: hidden !important;
            }
            .card-kpi__progress-fill {
                height: 100% !important;
                background-color: var(--card-color) !important;
                border-radius: 10px !important;
                transition: width 0.4s ease-in-out !important;
            }
            .card-kpi:hover {
                transform: translateY(-5px) !important;
                box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.12) !important;
            }

            /* --- Card Thông tin từng nhóm khách hàng --- */
            .card-customer {
                border-radius: 24px;
                padding: clamp(16px, 2vw, 25px);
                border: 1.5px solid #E2E8F0 !important;
                border-top: 5px solid var(--profile-color) !important;
                box-shadow: 0 15px 25px -5px rgba(0, 0, 0, 0.06);
            }
            .card-customer__group, .predict-card__group {
                display: inline-flex; 
                align-items: center; 
                gap: 4px;
                background-color: var(--profile-bg-light); 
                color: var(--profile-color);
                padding: 5px 12px;
                font-size: clamp(11px, 0.3vw + 0.6rem, 13px) !important;
                font-weight: 800;
                border-radius: 30px;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            .card-customer__group-icon, .predict-card__group-icon {
                font-size: clamp(12px, 0.3vw + 0.65rem, 14px);
            }
            .card-customer__title, .predict-card__title {
                font-size: clamp(15px, 0.8vw + 0.8rem, 18px) !important;
                font-weight: 650; 
                color: #0F172A; 
                margin-bottom: 10px; 
                display: flex; 
                align-items: center; 
                gap: 8px; 
                line-height: 1.2;
            }
            .card-customer__title-icon, .predict-card__title-icon {
                color: var(--profile-color); 
                font-size: clamp(18px, 1vw + 0.8rem, 22px);
            }
            .card-customer__desc {
                color: #1E293B; 
                font-size: clamp(13px, 0.3vw + 0.7rem, 14.5px) !important;
                line-height: 1.6;
            }
            .card-customer__action {
                background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
                border-radius: 16px;
                padding: clamp(14px, 1.8vw, 20px);
                border: 2px dashed var(--profile-color);
                margin-top: clamp(14px, 1.8vw, 20px);
            }
            .card-customer__action-icon {
                font-size: clamp(20px, 1.1vw + 0.8rem, 24px);
            }
            .card-customer__action-title, .predict-card__action-title { 
                display: flex; 
                align-items: center; 
                gap: 6px;
                color: var(--profile-color);
                font-weight: 800; 
                font-size: clamp(12px, 0.4vw + 0.65rem, 14.5px) !important; 
                text-transform: uppercase; 
                letter-spacing: 0.5px; 
                margin-bottom: 6px; 
            }
            .card-customer__action-content, .predict-card__action-content { 
                line-height: 1.6; 
                font-weight: 500; 
                color: #334155;
                font-size: clamp(13px, 0.3vw + 0.7rem, 14.5px) !important;
            }

            /* --- Card kết quả dự đoán khách hàng --- */
            .predict-card {
                background: #FFFFFF;
                border: 2.5px solid var(--profile-color) !important;
                padding: clamp(16px, 2vw, 24px);
                border-radius: 16px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
                margin-bottom: 20px;
                transition: all 0.3s ease-in-out; 
            }
            .predict-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                border-color: var(--profile-color);
            }
            .predict-card__desc {
                color: #1E293B; 
                font-size: clamp(13.5px, 0.3vw + 0.75rem, 15px) !important; 
                line-height: 1.6; 
                margin-bottom: 16px;
            }
            .predict-card__action {
                border: 1px dashed var(--profile-color);
                padding: clamp(12px, 1.5vw, 16px);
                border-radius: 12px;
            }

            /* ----------------------------------------------------------- */
            /* CSS Tab thông tin */
            /* ----------------------------------------------------------- */

             /* --- Dữ liệu phân tích --- */
            .card-info {
                border: 1px solid #E2E8F0;
                border-radius: 14px;
                padding: clamp(16px, 2vw, 24px);
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03), 0 2px 4px -1px rgba(0, 0, 0, 0.02);
                transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
                margin-bottom: 1rem;
            }
            .card-info:hover {
                transform: translateY(-4px);
                border-color: #2563EB;
                box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
            }
            .card-info__description {
                font-size: clamp(13.5px, 0.3vw + 0.75rem, 15px) !important;
                color: #334155;
                line-height: 1.65;
            }
            .card-info__grid {
                display: grid !important;
                grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
                gap: clamp(10px, 1.2vw, 16px) !important;
                margin-bottom: 10px !important;
                margin-top: 15px !important;
            }
            .data-card {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                padding: clamp(12px, 1.5vw, 16px);
                transition: all 0.3s ease;
                box-sizing: border-box !important;
            }
            .data-card:hover {
                background: #FFFFFF;
                border-color: #3B82F6;
                box-shadow: 0 6px 16px rgba(59, 130, 246, 0.06);
            }
            .data-card__title {
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 700;
                font-size: clamp(13.5px, 0.3vw + 0.75rem, 15px) !important;
                color: #0F172A;
                margin-bottom: 8px;
            }
            .data-card__icon {
                font-size: clamp(16px, 0.3vw + 0.8rem, 18px); 
                color: #2563EB; 
            }
            .data-card__text {
                font-size: clamp(12.5px, 0.3vw + 0.65rem, 14px) !important; 
                color: #475569; 
                margin: 0;
            }

            /* --- Kỹ thuật phân tích --- */
            .step-card {
                background: #F8FAFC; 
                padding: clamp(14px, 1.8vw, 20px); 
                border-radius: 12px; 
                border-left: 4px solid #60A5FA; 
                box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
            }
            .step-card__title {
                display: flex; 
                align-items: center; 
                gap: 6px; 
                margin-bottom: 12px; 
                color: #2563EB; 
                font-weight: 700; 
                font-size: clamp(13.5px, 0.3vw + 0.75rem, 15px) !important;
            }
            .step-card__icon {
                font-size: clamp(16px, 0.3vw + 0.8rem, 18px);
            }
            .step-card__content {
                margin: 0; 
                color: #475569; 
                line-height: 1.6;
                font-size: clamp(13px, 0.3vw + 0.7rem, 14.5px) !important;
            }
            .step-card__data {
                display: inline-block; 
                background: #EFF6FF; 
                color: #3B82F6; 
                font-weight: 600; 
                font-size: clamp(11.5px, 0.2vw + 0.6rem, 13px) !important; 
                padding: 2px 8px; 
                border-radius: 6px; 
                margin: 4px 2px; 
                border: 1px solid #DBEAFE;
            }
            .elbow {
                display: flex; 
                align-items: center; 
                gap: 6px; 
                margin-bottom: 8px;
                justify-content: center;
            }
            .elbow__icon {
                font-size: clamp(16px, 0.3vw + 0.8rem, 18px); 
                color: #2563EB;
            }
            .elbow__header {
                font-weight: 700; 
                color: #475569; 
                font-size: clamp(12.5px, 0.3vw + 0.65rem, 14px) !important;
            }

            /* ----------------------------------------------------------- */
            /* Tinh chỉnh cho giao diện cân đối */
            /* ----------------------------------------------------------- */

            /* --- Căn cân đối ratio --- */
            div[data-testid="stElementContainer"],
            div[data-testid="stRadio"],
            div[data-testid="stRadio"] > div[data-testid="stRadioGroup"] {
                width: 100% !important;
            }
            div[data-testid="stRadio"] > div[data-testid="stRadioGroup"] {
                display: flex !important;
                justify-content: space-between !important;
            }
            
            /* Xóa biểu tượng bên cạnh page header */
            .st-emotion-cache-gi0tri {
                display: none !important;
            }
            .traces {
                display:flex;
                justify-content: center;
            }

            /* ----------------------------------------------------------- */
            /* Responsive theo main content */
            /* ----------------------------------------------------------- */
            
            section[data-testid="stMain"] {
                container-type: inline-size !important;
                container-name: main-viewport !important;
                width: 100% !important;
            }

            @container main-viewport (max-width: 765px) {
                .card-grid {
                    grid-template-columns: repeat(3, 1fr) !important;
                }
                .card-info__grid {
                    grid-template-columns: repeat(1, 1fr) !important;
                }
                div[data-testid="stHorizontal"] {
                    flex-direction: column !important;
                    gap: 0;
                }
                div[data-testid="stColumn"] {
                    width: 100% !important;
                }
                [data-testid="stForm"] .st-emotion-cache-1permvm {
                    flex-direction: column !important;
                }
                [data-testid="stForm"] .st-emotion-cache-hua6f6 {
                    width: 100% !important;
                }
            }

            @container main-viewport (max-width: 535px) {
                .card-grid {
                    grid-template-columns: repeat(2, 1fr) !important;
                }
            }
        </style>
    """, unsafe_allow_html=True)
    
    # --- 1. Tiêu đề trang ---
    st.markdown("""
        <h1 class="page-header">
            Phân Khúc Khách Hàng Thẻ Tín Dụng
        </h1>
    """, unsafe_allow_html=True)

    #  --- 2. Banner nguồn gốc dữ liệu & rào cản bảo mật ---
    st.markdown("""
        <div class="data-banner">
            <div class="data-banner__card">
                <div class="data-banner__badge">
                    <span class="material-icons">bar_chart</span>
                </div>
                <div class="data-banner__content">
                    <b>Hệ thống phân khúc hành vi tiêu dùng của </b>
                    <span class="data-banner__highlight">8.950 chủ thẻ tín dụng</span>
                    <b> thành </b>
                    <span class="data-banner__highlight">6 nhóm khách hàng</span>
                    <b> từ bộ dữ liệu uy tín </b> 
                    <a class="data-banner__link" href="https://www.kaggle.com/datasets/arjunbhasin2013/ccdata" target="_blank">
                        Credit Card Dataset for Clustering
                    </a> 
                    <b>do tác giả <i>Arjun Bhasin</i> thu thập và đăng tải.</b>
                </div>
            </div>
            <div class="data-banner__card">
                <div class="data-banner__badge">
                    <span class="material-icons">verified_user</span>
                </div>
                <div class="data-banner__content">
                    <b>Do rào cản bảo mật nghiêm ngặt tại Việt Nam </b>
                    (<span class="data-banner__highlight">Nghị định 13/2023/NĐ-CP</span>)
                    <b>, việc khai thác dữ liệu giao dịch thực tế từ các ngân hàng nội địa là không thể công khai. Bộ dữ liệu chuẩn hóa quốc tế này được lựa chọn vì phản ánh tốt các chỉ số cốt lõi tương đồng với hệ thống </b>
                    <span class="data-banner__highlight">Core Banking</span>: 
                    <b><i>Số dư nợ, Thói quen mua sắm, Tần suất rút tiền và Hạn mức tín dụng.</i></b>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 3. Chia 3 Tab (Trực quan - Dự đoán - Thông tin) ---
    tab_dashboard, tab_prediction, tab_infomation = st.tabs(["**Tổng Quan & Trực Quan Hóa**", "**Phân Khúc Khách Hàng**", "**Thông Tin Thêm**"])
    
    # 3_1. TAB Trực quan
    with tab_dashboard:
        # SECTION 1: PHÂN BỔ KHÁCH HÀNG
        st.markdown("""
            <div class="section">
                <span class="material-icons section__icon">people</span>
                <span class="section__title">PHÂN BỔ KHÁCH HÀNG</span>
            </div>
        """, unsafe_allow_html=True)

        counts = df_final["Full_Nhóm"].value_counts()
        total_customers = len(df_final)
        card_items_html = []

        for group_id, info in STRATEGIES.items():
            icon_name = info["icon"]
            color = info["color"]
            bg_light = info["bg_light"]
            short_label = info["short_label"]
            group_name = group_id.split(": ")[1]
            count_val = counts.get(group_id, 0)
            value_str = f"{count_val:,}"
            percentage = (count_val / total_customers * 100) if total_customers > 0 else 0

            card_item = f"""
            <div class="card-kpi" style="--card-color: {color}; --card-bg-light: {bg_light};">
                <div class="card-kpi__header">
                    <span class="material-icons card-kpi__icon">{icon_name}</span>
                    <span class="card-kpi__label">{short_label}</span>
                </div>
                <div class="card-kpi__body">
                    <div class="card-kpi__num">{value_str}</div>
                    <div class="card-kpi__text">{group_name}</div>
                </div>
                <div class="card-kpi__footer">
                    <span class="card-kpi__percentage-text">{percentage:.1f}%</span>
                    <div class="card-kpi__progress-bar">
                        <div class="card-kpi__progress-fill" style="width: {percentage:.1f}%;"></div>
                    </div>
                </div>
            </div>
            """
            card_items_html.append(card_item.replace("\n", "").strip())

        card_html = f'<div class="card-grid">{"".join(card_items_html)}</div>'

        st.markdown(card_html, unsafe_allow_html=True)

        c_left, c_right = st.columns([50,50])
        with c_left:
            # SECTION 2: BẢN ĐỒ NHÓM KHÁCH HÀNG
            st.markdown("""
                <div class="section">
                    <span class="material-icons section__icon">hub</span>
                    <span class="section__title">BẢN ĐỒ NHÓM KHÁCH HÀNG</span>
                </div>
            """, unsafe_allow_html=True)

            chart_type = st.radio(
                "Chọn kiểu hiển thị bản đồ t-SNE:",
                options=["Bản đồ Mặt phẳng 2D", "Bản đồ Không gian 3D"],
                horizontal=True,
                label_visibility="collapsed"
            )

            df_sorted = df_final.sort_values(by="Nhóm")

            tsne_custom_data = ["Nhóm", "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "PURCHASES_TRX", "PURCHASES_FREQUENCY", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS"]

            tsne_hovertemplate = (
                "<b>%{customdata[0]}</b><br>"
                "Balance: $%{customdata[1]:,.0f}<br>"
                "Purchases: $%{customdata[2]:,.0f}<br>"
                "Oneoff Purchases: $%{customdata[3]:,.0f}<br>"
                "Installments Purchases: $%{customdata[4]:,.0f}<br>"
                "Purchase Transactions: %{customdata[5]}<br>"
                "Purchase Frequency: %{customdata[6]:.2f}<br>"
                "Cash Advance: $%{customdata[7]:,.0f}<br>"
                "Credit Limit: $%{customdata[8]:,.0f}<br>"
                "Payments: $%{customdata[9]:,.0f}"
                "<extra></extra>"
            )

            legend_responsive = dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,                 
                title_text="",
                font=dict(size=13),  
                entrywidth=0.33,
                entrywidthmode="fraction"  
            )

            if chart_type == "Bản đồ Không gian 3D":
                fig3d = px.scatter_3d(
                    df_sorted, x="t-SNE 1", y="t-SNE 2", z="t-SNE 3", color="Nhóm",
                    custom_data=tsne_custom_data,
                    color_discrete_map=COLOR_MAP_SHORT, opacity=0.6
                )
                fig3d.update_traces(marker=dict(size=1.5), hovertemplate=tsne_hovertemplate)
                fig3d.update_layout(
                    margin=dict(t=0,l=0,b=0,r=0),
                    responsive=True,
                    autosize=True,
                    height=450,
                    scene=dict(
                        xaxis=dict(title=None, showgrid=True, gridcolor="#CBD5E1"), 
                        yaxis=dict(title=None, showgrid=True, gridcolor="#CBD5E1"), 
                        zaxis=dict(title=None, showgrid=True, gridcolor="#CBD5E1"),
                        bgcolor="#FFFFFF",
                    ),
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                    legend=legend_responsive
                )
                # Dùng config responsive=True để Plotly tự bắt sự kiện Resize của Browser
                st.plotly_chart(fig3d, width='stretch', config={'responsive': True})
                    
            else:
                fig2d = px.scatter(
                    df_sorted, x="t-SNE 1", y="t-SNE 2", color="Nhóm",
                    custom_data=tsne_custom_data,
                    color_discrete_map=COLOR_MAP_SHORT, opacity=0.6
                )
                fig2d.update_traces(marker=dict(size=4.5), hovertemplate=tsne_hovertemplate)
                fig2d.update_layout(
                    margin=dict(t=0,l=0,b=0,r=0),
                    responsive=True,
                    autosize=True,
                    height=450,
                    xaxis=dict(title=None, showgrid=True, gridcolor="#CBD5E1", zerolinecolor="#475569"),
                    yaxis=dict(title=None, showgrid=True, gridcolor="#CBD5E1", zerolinecolor="#475569"),
                    paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
                    legend=legend_responsive
                )
                
                st.plotly_chart(fig2d, width='stretch', config={'responsive': True})

        with c_right:
            # SECTION 3: Chiến Lược Khách Hàng
            st.markdown("""
                <div class="section">
                    <span class="material-icons section__icon">ads_click</span>
                    <span class="section__title">CHIẾN LƯỢC KHÁCH HÀNG</span>
                </div>
            """, unsafe_allow_html=True)

            selected_view = st.selectbox("Chọn cụm nghiệp vụ:", list(STRATEGIES.keys()), label_visibility="collapsed")
            current_data = STRATEGIES[selected_view]

            profile_html = f"""
            <div class="card-customer" style="--profile-color: {current_data['color']}; --profile-bg-light: {current_data['bg_light']};">
                <span class="card-customer__group">
                    <span class="material-icons card-customer__group-icon">label</span>
                    Phân khúc: {current_data['type']}
                </span>
                <div class="card-customer__title">
                    <span class="material-icons card-customer__title-icon">{current_data['icon']}</span>
                    <span>{selected_view}</span>
                </div>
                <div class="card-customer__desc">
                    <b>Đặc trưng hành vi tài chính:</b> <i>{current_data['desc']}</i>
                </div> 
                <div class="card-customer__action">
                    <div class="card-customer__action-title">
                        <span class="material-icons card-customer__action-icon">ads_click</span>
                        Gợi ý chiến lược / Giải pháp hành động
                    </div>
                    <div class="card-customer__action-content">{current_data['action']}</div>
                </div>
            </div>
            """
            st.markdown(profile_html.replace("\n", ""), unsafe_allow_html=True)

            sub_df = df_final[df_final["Full_Nhóm"] == selected_view]
            columns_mapping = {
                "BALANCE": "Số dư nợ hiện tại", 
                "PURCHASES": "Tổng tiền mua sắm", 
                "CASH_ADVANCE": "Rút tiền mặt từ thẻ", 
                "CREDIT_LIMIT": "Hạn mức tối đa thẻ", 
                "PAYMENTS": "Số tiền đã trả lại"
            }

            mean_values = [sub_df[col].mean() for col in columns_mapping.keys()]
            chart_data = pd.DataFrame({
                "Tên Tiếng Anh": list(columns_mapping.keys()), 
                "Tên Tiếng Việt": list(columns_mapping.values()), 
                "Giá trị trung bình ($)": mean_values
            })

            mini_fig = px.bar(
                chart_data, 
                x="Tên Tiếng Việt", 
                y="Giá trị trung bình ($)", 
                color="Tên Tiếng Việt", 
                custom_data=["Tên Tiếng Anh", "Tên Tiếng Việt"], 
                color_discrete_sequence=[current_data["color"]], 
            )
            
            mini_fig.update_traces(
                hovertemplate="<b>Chỉ báo tài chính:</b> %{customdata[0]}<br><b>%{customdata[1]}:</b> %{y:,.2f} $<extra></extra>"
            )

            short_labels = ["Số dư nợ<br>hiện tại", "Tổng tiền<br>mua sắm", "Rút tiền mặt<br>từ thẻ", "Hạn mức<br>tối đa thẻ", "Số tiền<br>đã trả lại"]

            mini_fig.update_layout(
                autosize=True,
                height=250,
                showlegend=False,
                xaxis=dict(
                    tickmode='array', 
                    tickvals=chart_data["Tên Tiếng Việt"], 
                    ticktext=short_labels, 
                    tickfont=dict(size=11, color="#334155"), 
                    tickangle=0
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor="#F1F5F9",
                    title=dict(text="Giá trị trung bình ($)", font=dict(size=11, color="#64748B"))
                ),
                paper_bgcolor='#FFFFFF', 
                plot_bgcolor='#FFFFFF',
                xaxis_title=None
            )

            st.plotly_chart(mini_fig, width='stretch',config={'responsive': True})    

    # 3_2. TAB dự đoán
    with tab_prediction:
        st.markdown("""
            <div class="section">
                <span class="material-icons section__icon">person_add</span>
                <span class="section__title">NHẬP THÔNG TIN KHÁCH HÀNG</span>
            </div>
        """, unsafe_allow_html=True)

        with st.form("input_form"):
            col1, col2 = st.columns(2)
            with col1:
                balance = st.number_input("**Số dư nợ hiện tại (Balance)**", min_value=0.0, value = 3000.0, step=0.1, icon=":material/attach_money:", help="Số nợ hiện tại chủ thẻ chưa hoàn trả")
                payments = st.number_input("**Tổng số tiền đã trả cho ngân hàng (Payments)**", min_value=0.0, value=2000.0, step=0.1, icon=":material/attach_money:", help="Tổng số tiền khách hàng đã thanh toán/chuyển khoản lại cho ngân hàng để trả nợ.")
                credit_limit = st.number_input("**Hạn mức tín dụng được cấp (Credit Limit)**", min_value=0.0, value=4000.0, step=0.1, icon=":material/attach_money:", help="Số tiền tối đa ngân hàng cấp cho chủ thẻ tiêu dùng.")  
            with col2:
                cash_advance = st.number_input("**Tổng giá trị rút tiền mặt (Cash Advance)**", min_value=0.0, value=0.0, step=0.1, icon=":material/attach_money:", help="Tổng số tiền mặt chủ thẻ đã ứng trực tiếp tại cây ATM hoặc quầy giao dịch.")
                purchases = st.number_input("**Tổng giá trị mua sắm (Purchases)**", min_value=0.0, value=1000.0, step=0.1, icon=":material/attach_money:", help="Tổng số tiền quẹt thẻ thanh toán hàng hóa, dịch vụ tại các máy POS/Online.")
                tenure = st.number_input("**Giai đoạn quan sát (Tenure - Tháng)**", min_value=6, max_value=12, value=12, step=1, icon=":material/calendar_month:", help="Số tháng tích lũy dữ liệu hành vi gần đây của khách hàng (Từ 6 - 12 tháng).")
            submitted = st.form_submit_button("**Tiến hành phân khúc khách hàng**")
            
            if submitted:
                iqr_bounds = joblib.load(IQR_BOUNDS_PATH)
                scaler_normal = joblib.load(SCALER_NORMAL_PATH)
                scaler_outlier = joblib.load(SCALER_OUTLIER_PATH)
                kmeans_normal = joblib.load(KMEANS_NORMAL_PATH)
                kmeans_outlier = joblib.load(KMEANS_OUTLIER_PATH)
                
                user_features = {
                    "BALANCE": balance, 
                    "PURCHASES": purchases, 
                    "CASH_ADVANCE": cash_advance, 
                    "CREDIT_LIMIT": credit_limit, 
                    "PAYMENTS": payments
                }
                
                is_user_outlier = False
                for col, val in user_features.items():
                    if val > iqr_bounds[col]["upper"] or val < iqr_bounds[col]["lower"]:
                        is_user_outlier = True
                        break
                
                # Thứ tự mảng: [BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS]
                input_df = pd.DataFrame([{
                "BALANCE": balance,
                "PURCHASES": purchases,
                "CASH_ADVANCE": cash_advance,
                "CREDIT_LIMIT": credit_limit,
                "PAYMENTS": payments
                }])
                
                if is_user_outlier:
                    scaled_vector = scaler_outlier.transform(input_df)
                    pred_raw = kmeans_outlier.predict(scaled_vector)[0]
                    cluster_mapping = {
                        0: "Nhóm 5: Siêu VIP Chi Tiêu Khủng", 
                        1: "Nhóm 4: VIP Tiêu Dùng Cao Cấp", 
                        2: "Nhóm 6: Tín Dụng Rút Tiền Mặt Cao"
                    } 
                    final_cluster_key = cluster_mapping.get(pred_raw)
                else:
                    scaled_vector = scaler_normal.transform(input_df)
                    pred_raw = kmeans_normal.predict(scaled_vector)[0]
                    cluster_mapping = {
                        0: "Nhóm 2: Tiêu Dùng Thông Minh", 
                        1: "Nhóm 1: Khách Hàng Thụ Động", 
                        2: "Nhóm 3: Vay Tiêu Dùng Nhỏ"
                    }
                    final_cluster_key = cluster_mapping.get(pred_raw)

                matched_strategy = None
                strategy_label = ""
                
                for key, info in STRATEGIES.items():
                    if info.get("old_id") == final_cluster_key or key == final_cluster_key:
                        matched_strategy = info
                        strategy_label = key
                        break
                
                if matched_strategy:
                    brand_color = matched_strategy.get('color', '#2563EB')
                    bg_light = matched_strategy.get('bg_light', 'rgba(37, 99, 235, 0.1)')
                    group_icon = matched_strategy.get('icon', 'person')
                    group_type = matched_strategy.get('type', 'Tổng quan')

                    profile_html = f"""
                        <div class="predict-card" style="--profile-color: {brand_color}; --profile-bg-light: {bg_light};">
                            <span class="predict-card__group">
                                <span class="material-icons predict-card__group-icon">label</span>
                                Phân khúc: {group_type}
                            </span>
                            <div class="predict-card__title">
                                <span class="material-icons predict-card__title-icon">{group_icon}</span>
                                <span>{strategy_label}</span>
                            </div>
                            <div class="predict-card__desc">
                                <b>Đặc trưng hành vi tài chính:</b> <i>{matched_strategy["desc"]}</i>
                            </div> 
                            <div class="predict-card__action">
                                <div class="predict-card__action-title">
                                    <span class="material-icons predict-card__action-icon">ads_click</span>
                                    Gợi ý chiến lược / Giải pháp hành động
                                </div>
                                <div class="predict-card__action-content">{matched_strategy["action"]}</div>
                            </div>
                        </div>
                        """
                    st.markdown(profile_html.replace("\n", ""), unsafe_allow_html=True)

    # 3_3. TAB Thông tin
    with tab_infomation:
        # SECTION 1: Dữ liệu phân tích
        st.markdown("""
            <div class="section">
                <span class="material-icons section__icon">storage</span>
                <span class="section__title">DỮ LIỆU PHÂN TÍCH</span>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card-info">
            <p class="card-info__description">
                Tập dữ liệu thô ban đầu của ngân hàng quản lý gồm <b>18 trường thông tin</b> khác nhau của chủ thẻ. Nhằm tối ưu hóa hiệu năng tính toán và tập trung giải quyết bài toán cốt lõi là <b>Tăng trưởng doanh số & Kích cầu chi tiêu</b>, hệ thống tiến hành sàng lọc và trích xuất ra <b>9 cột chỉ báo hành vi</b> quan trọng nhất:
            </p>
            <div class="card-info__grid">
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">account_balance_wallet</span>BALANCE
                    </div>
                    <p class="data-card__text"><b>Số dư tài khoản:</b> Số tiền nợ tín dụng hiện tại mà khách hàng chưa thanh toán cho ngân hàng.</p>
                </div>
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">shopping_bag</span>PURCHASES
                    </div>
                    <p class="data-card__text"><b>Tổng giá trị mua sắm:</b> Toàn bộ số tiền tài khoản đã quẹt thẻ chi tiêu mua sắm hàng hóa dịch vụ.</p>
                </div>
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">update</span>PURCHASES_FREQUENCY
                    </div>
                    <p class="data-card__text"><b>Tần suất quẹt thẻ:</b> Mức độ thường xuyên mua sắm của chủ thẻ (Chỉ số từ 0 đến 1).</p>
                </div>
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">payments</span>ONEOFF_PURCHASES
                    </div>
                    <p class="data-card__text"><b>Mua sắm thanh toán ngay:</b> Số tiền chi tiêu cho các giao dịch quẹt thẻ trả thẳng 1 lần.</p>
                </div>
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">credit_card</span>INSTALLMENTS_PURCHASES
                    </div>
                    <p class="data-card__text"><b>Mua sắm trả góp:</b> Giá trị tiền chi tiêu phục vụ cho các dịch vụ đăng ký trả góp hàng tháng.</p>
                </div>
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">receipt_long</span>PURCHASES_TRX
                    </div>
                    <p class="data-card__text"><b>Số lượt giao dịch mua:</b> Tổng số lần phát sinh hóa đơn chi tiêu mua sắm thành công.</p>
                </div>    
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">local_atm</span>CASH_ADVANCE
                    </div>
                    <p class="data-card__text"><b>Rút tiền mặt:</b> Tổng số tiền mặt mà chủ thẻ đã rút trực tiếp tại cây ATM qua thẻ tín dụng.</p>
                </div>
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">speed</span>CREDIT_LIMIT
                    </div>
                    <p class="data-card__text"><b>Hạn mức tín dụng:</b> Ngưỡng tiêu tiền tối đa được ngân hàng phê duyệt và cấp cho chủ thẻ.</p>
                </div> 
                <div class="data-card">
                    <div class="data-card__title">
                        <span class="material-icons data-card__icon">fact_check</span>PAYMENTS
                    </div>
                    <p class="data-card__text"><b>Số tiền đã trả:</b> Tổng tiền khách hàng đã nộp lại cho ngân hàng để thanh toán dư nợ kỳ trước.</p>
                </div>                   
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SECTION 2: Kỹ thuật phân tích
        st.markdown("""
            <div class="section">
                <span class="material-icons section__icon">alt_route</span>
                <span class="section__title">Kỹ THUẬT PHÂN TÍCH</span>
            </div>
        """, unsafe_allow_html=True)

        # Bước 1
        st.markdown("""
        <div class="card-info">
            <div class="step-card">
                <div class="step-card__title">
                    <span class="material-icons step-card__icon">psychology</span> Bước 1: Huấn luyện học máy trên 5 cột cốt lõi
                </div>
                <p class="step-card__content">
                    Nhằm tránh nhiễu toán học, thuật toán K-Means chỉ sử dụng <b>5 biến dòng tiền chính</b> để phân cụm:<br>
                    <span class="step-card__data">BALANCE</span>
                    <span class="step-card__data">PURCHASES</span>
                    <span class="step-card__data">CASH_ADVANCE</span>
                    <span class="step-card__data">CREDIT_LIMIT</span>
                    <span class="step-card__data">PAYMENTS</span><br>
                    Dữ liệu được chia thành tập <b>Số Đông (6,746 dòng)</b> và tập <b>Ngoại Lai (2,204 dòng)</b> để tiến hành tìm điểm gãy tối ưu (Elbow Method) và gán nhãn độc lập (Mỗi tập chia thành 3 cụm).
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Đồ thị elbow 
        col_el1, col_el2 = st.columns(2)

        with col_el1:
            st.markdown("""
            <div class="elbow">
                <span class="material-icons elbow__icon">trending_down</span>
                <span class="elbow__header">Elbow Method - Tập Số Đông (Normal) ➔ K = 3</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(ELBOW_NORMAL_PATH, width='stretch')

        with col_el2:
            st.markdown("""
            <div class="elbow">
                <span class="material-icons elbow__icon">trending_down</span>
                <span class="elbow__header">Elbow Method - Tập Ngoại Lai (Outliers) ➔ K = 3</span>
            </div>
            """, unsafe_allow_html=True)
            st.image(ELBOW_OUTLIER_PATH, width='stretch')

        # Bước 2
        st.markdown("""
        <div class="card-info">
            <div class="step-card">
                <div class="step-card__title">
                    <span class="material-icons step-card__icon">analytics</span> Bước 2: Phân tích đặc trưng dựa trên 9 cột
                </div>
                <p class="step-card__content">
                    <b>Sau khi đã phân tách thành công thành 6 cụm khách hàng rõ rệt</b>, hệ thống phân tích hành vi từng cụm từ trung bình <b>9 cột thuộc tính ban đầu</b>.<br>
                    Lúc này, hệ thống phân tích thêm các chỉ số mở rộng (tần suất, hình thức mua sắm, số lượt giao dịch) để khắc họa trọn vẹn chân dung đặc trưng của từng nhóm khách hàng.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
