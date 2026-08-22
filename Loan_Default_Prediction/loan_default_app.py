import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import joblib
from datetime import datetime
import os

def run_loan_default_app():
    # --- Lấy các đường dẫn ---
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(APP_DIR, "model", "loan_default_model.pkl")
    CLASSIFICATION_REPORT_PATH = os.path.join(APP_DIR, "report", "loan_default_classification_report.csv")
    CONFUSION_MATRIX_PATH = os.path.join(APP_DIR, "report", "loan_default_confusion_matrix.jpg")

    # --- CSS Internal ---
    st.markdown("""
    <style>
        /* Ẩn 1 số element mặc định của streamlit */
        .st-emotion-cache-gi0tri {
            display: none !important;
        }

        /* Header trang và tiêu đề mỗi section */
        .page-header {
            text-align: center;
            background: linear-gradient(to right, #60A5FA, #2563EB, #1E3A8A); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            font-weight: 800;
            font-size: clamp(24px, 3.5vw + 1rem, 38px) !important;
            margin-bottom: clamp(12px, 2vw, 30px) !important;
        } 

        hr {
            margin: 0 0 1rem 0 !important;
        }

        .section-title {
            font-size: clamp(16px, 0.85vw + 0.8rem, 25px) !important; 
            font-weight: 650 !important; 
            color: #0F172A; 
            margin-top: clamp(10px, 2vw, 25px) !important;
        }

        /* Card kết quả dự đoán */
        .result-banner {
            background-color: var(--bg-color) !important; 
            border-left: 6px solid var(--border-color) !important; 
            padding: clamp(13px, 2vw, 20px); 
            border-radius: 8px; 
            margin-bottom: clamp(0px, 1.25vw, 12px)
        }

        .result-banner__title {
            font-size: clamp(14.3px, 1vw + 0.5rem, 20px) !important;
            margin: 0 !important;
            color: var(--text-color) !important;
        }

        .result-banner__desc {
            color: var(--text-color);
            font-size: clamp(13.5px, 0.5vw + 0.6rem, 15px); 
            margin: 0 !important;
            line-height: 1.5;
            letter-spacing: 0.3px;
        }

        .card-metric {
            background: #FFFFFF; 
            margin-bottom: 1rem;
            border: 1px solid #E2E8F0; 
            padding: clamp(12px, 1.5vw, 15px); 
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04); 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; 
        }

        .card-metric__list {
            display: flex;
            flex-direction: column;
            justify-content: space-evenly;
        }

        .card-metric__item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: clamp(8px, 1vw, 12px) 0;
            border-bottom: 1px solid #F1F5F9;
        }

        .card-metric__label {
            display: flex; 
            align-items: center; 
            gap: 8px;
        }

        .card-metric__label-text {
            font-size: clamp(12px, 0.4vw + 0.55rem, 13.5px); 
            color: #334155; 
            font-weight: 500;
        }

        .card-metric__label-icon {
            color: #64748B; 
            font-size: clamp(15px, 1vw + 0.2rem, 18px);
        }

        /* Card hiệu suất mô hình */
        .card-report {
            padding: 15px; 
            border-radius: 10px;
            box-sizing: border-box;  
            height: 100% !important;
        }
        
        .card-report--green {
            background-color: #F0FDF4;
            border-top: 2px solid #86EFAC;
        }

        .card-report--red {
            background-color: #FEF2F2;
            border-top: 2px solid #FCA5A5;
        }
        
        .card-report--yellow {
            background-color: #FFFBEB;
            border: 2px dashed #FDE047;
            margin-bottom: 1rem;
            margin-top: 10px;
        }

        .card-report__title {
            display: flex !important;
            gap: 5px !important;
            align-items: center !important;
            margin-bottom: 8px !important;
            padding: 0 !important;
        }

        .card-report__icon {
            font-size: clamp(19px, 1.5vw, 24px) !important;
            line-height: 1 !important;
        }

        .card-report__title-content {
            font-size: clamp(14.5px, 1.1vw, 16px) !important;
            font-weight: 700 !important;
            letter-spacing: 0.3px;
        }

        .card-report__title--green {
            color: #166534 !important;
        }

        .card-report__title--red {
            color: #991B1B !important;
        }

        .card-report__title--yellow {
            color: #854D0E !important;
        }

        .card-report__list {
            font-size: clamp(13px, 0.95vw, 14.5px) !important;
            line-height: 1.6 !important;
            padding-left: 20px !important; 
            margin: 0 !important;
            color: #334155;
        }

        .card-report__list li {
            margin-bottom: 8px;
        }

        .card-report__list li:last-child {
            margin-bottom: 0;
        }

        .card-report__sublist {
            padding-left: 20px !important;
            margin-top: 4px !important;
            margin-bottom: 3px !important;
            list-style-type: circle !important;
        }

        /* Responsive theo main content */
        section[data-testid="stMain"] {
            container-type: inline-size !important;
            container-name: main-viewport !important;
            width: 100% !important;
        }

        div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        @container main-viewport (max-width: 730px) {
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }
            div[data-testid="stColumn"] {
                width: 100% !important;
            }
            .card-report--red {
                border: 2px solid #FCA5A5;
                margin-top: 10px;
            }
            .card-report--green {
                border: 2px solid #86EFAC;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # --- 1. Tiêu đề trang ---
    st.markdown("""
    <h1 class="page-header">
        Thẩm Định Rủi Ro & Duyệt Hồ Sơ Vay Vốn
    </h1>
    """, unsafe_allow_html=True
    )
    st.markdown("---")

    # --- 2. Load mô hình ---
    model = joblib.load(MODEL_PATH)

    # --- 3. Form nhập liệu người dùng ---
    st.markdown("""
        <h2 class="section-title">
            NHẬP THÔNG TIN KHÁCH HÀNG
        </h2>
    """, unsafe_allow_html=True
    )
    with st.form("input_form"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "▥ **Chi tiết Khoản vay**",
            "▣ **Thông tin Định danh**",
            "▧ **Năng lực Tài chính**",
            "▤ **Lịch sử Tín dụng**"
        ])

        # --- TAB 1: Chi tiết Khoản vay ---
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                LoanAmount = st.number_input("Số tiền vay (Loan Amount - $)", min_value=5000, value=15000, max_value=249999, step=1, icon=":material/attach_money:")
                LoanTerm = st.number_input("Kỳ hạn vay (Loan Term - Tháng)", min_value=12, max_value=60, value=24, step=12, icon=":material/calendar_month:")
            with col2:
                LoanPurpose = st.selectbox("Mục đích vay (Loan Purpose)", ["Vay mua nhà (Home)", "Vay mua ô tô (Auto)", "Vay học tập / Du học (Education)", "Vay sản xuất kinh doanh (Business)", "Mục đích tiêu dùng khác (Other)"])
                HasMortgage = st.selectbox("Đang có khoản vay thế chấp khác? (Has Mortgage)", ["Có", "Không"])
            InterestRate = st.slider("Lãi suất áp dụng (Interest Rate - %)", min_value=2.0, max_value=25.0, value=5.05, step=0.01, format="%.2f %%")

        # --- TAB 2: Thông tin Định danh ---
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                MaritalStatus = st.selectbox("Tình trạng hôn nhân (Marital Status)", ["Độc thân", "Kết hôn", "Đã ly hôn"]) 
                Education =  st.selectbox("Học vấn (Education)", ["Trung học Phổ thông (High School)", "Cử nhân / Kỹ sư (Bachelor's)", "Thạc sĩ (Master's)", "Tiến sĩ (PhD)"])   
            with col2:
                HasDependents = st.selectbox("Có người thân phụ thuộc? (Has Dependents)", ["Có", "Không"])
                HasCoSigner = st.selectbox("Có người đồng ký tên/bảo lãnh không? (Has Cosigner)", ["Có", "Không"])
            Age = st.slider("Tuổi (Age)", min_value=18, max_value=69, value=30, step=1) 

        # --- TAB 3: Năng lực Tài chính---
        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                Income = st.number_input("Tổng thu nhập hàng năm", min_value=15000, max_value=149999, value=30000, step=1, icon=":material/attach_money:", help="Tổng thu nhập trước thuế trong một năm")
            with col2:
                MonthsEmployed = st.number_input("Thâm niên làm việc theo tháng", min_value=0, max_value=119, value=50, step=1, icon=":material/calendar_month:", help="Số tháng làm việc liên tục tại công ty hiện tại")
            with col3:
                EmploymentType = st.selectbox("Loại hình công việc", ["Full-time", "Part-time", "Self-employed", "Unemployed"], help="Hình thức hợp đồng lao động hiện tại")

        # --- TAB 4: Lịch sử Tín dụng ---
        with tab4:
            col1, col2, col3 = st.columns(3)
            with col1:
                CreditScore = st.number_input("Điểm tín dụng (Credit Score)", min_value=300, max_value=849, value=650, step=1, icon=":material/credit_score:")
            with col2:
                NumCreditLines = st.number_input("Số thẻ đang mở (Num Credit Lines)", min_value=1, max_value=4, value=2, step=1, icon=":material/credit_card:")
            with col3:
                DTIRatio = st.number_input("Tỷ lệ Nợ/Thu nhập (DTI Ratio)", min_value=0.1, max_value=0.9, value=0.3, step=0.01, icon=":material/balance:")

        submitted = st.form_submit_button("**Dự đoán**")
    
    # --- 4. Dự đoán & Hiển thị kết quả ---
    # Khởi tạo session_state cho lịch sử tín dụng
    if "loan_history" not in st.session_state:
        st.session_state["loan_history"] = []
    proba = None

    if submitted:
        # 4_1. Mapping giá trị về dạng mô hình đã học
        yes_no_map = {"Không": "No", "Có": "Yes"}
        
        MaritalStatus_map = {
            "Độc thân": "Single", 
            "Kết hôn": "Married", 
            "Đã ly hôn": "Divorced"
        }
        
        Education_map = {
            "Trung học Phổ thông (High School)": "High School",
            "Cử nhân / Kỹ sư (Bachelor's)": "Bachelor's",
            "Thạc sĩ (Master's)": "Master's",
            "Tiến sĩ (PhD)": "PhD"
        }
        
        LoanPurpose_map = {
            "Vay mua nhà (Home)": "Home",
            "Vay mua ô tô (Auto)": "Auto",
            "Vay học tập / Du học (Education)": "Education",
            "Vay sản xuất kinh doanh (Business)": "Business",
            "Mục đích tiêu dùng khác (Other)": "Other"
        }

        # 4_2. Tạo dictionary cho dữ liệu đầu vào
        input_data = {
            "Age": Age,
            "Income": Income,
            "LoanAmount": LoanAmount,
            "CreditScore": CreditScore,
            "MonthsEmployed": MonthsEmployed,
            "NumCreditLines": NumCreditLines,
            "InterestRate": InterestRate,
            "LoanTerm": LoanTerm,
            "DTIRatio": DTIRatio,
            "Education": Education_map[Education],
            "EmploymentType": EmploymentType,
            "MaritalStatus": MaritalStatus_map[MaritalStatus],
            "HasMortgage": yes_no_map[HasMortgage], 
            "HasDependents": yes_no_map[HasDependents],
            "LoanPurpose": LoanPurpose_map[LoanPurpose],
            "HasCoSigner": yes_no_map[HasCoSigner]           
        }
        input_df = pd.DataFrame([input_data])
        
        # 4_3. Dự đoán xác suất vỡ nợ
        proba = model.predict_proba(input_df)[0][1]

        # 4_4. Lưu vào lịch sử dự đoán
        record = {
            "Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Xác suất (%)": round(proba * 100, 2),
            "Tuổi": Age,
            "Thu nhập hàng năm ($)": Income,
            "Số tiền vay ($)": LoanAmount,
            "Điểm tín dụng": CreditScore,
            "Thâm niên (tháng)": MonthsEmployed,
            "Số khoản vay đang mở": NumCreditLines,
            "Lãi suất (%)": InterestRate,
            "Kỳ hạn (tháng)": LoanTerm,
            "Tỷ lệ Nợ/Thu nhập": DTIRatio,
            "Học vấn": Education,
            "Loại hình công việc": EmploymentType,
            "Tình trạng hôn nhân": MaritalStatus,
            "Đang vay thế chấp khác?": HasMortgage, 
            "Có người phụ thuộc?": HasDependents,
            "Mục đích vay": LoanPurpose,
            "Có người bảo lãnh?": HasCoSigner 
        }
        st.session_state["loan_history"].append(record)

    # --- ĐƯA PHẦN HIỂN THỊ RA NGOÀI FORM ĐỂ GIAO DIỆN RỘNG RÃI, ĐẸP MẮT ---
    if proba is not None:
        st.markdown("""
            <h2 class="section-title">
                KẾT QUẢ PHÂN TÍCH
            </h2>
        """, unsafe_allow_html=True
        )
        
        # Xác định nhóm màu sắc và nội dung thông báo dựa trên xác suất rủi ro
        if proba < 0.17:
            bg_color = "#E8F8F5"
            border_color = "#2ECC71"
            text_color = "#196F3D"
            status_title = "HỒ SƠ ĐỦ ĐIỀU KIỆN (AN TOÀN)"
            status_desc = "✅ <b>Khuyến nghị:</b> Nên chấp nhận duyệt giải ngân khoản vay. Khách hàng có điểm uy tín tốt và các chỉ số tài chính nằm trong ngưỡng an toàn."
            bar_color = "#27AE60"
        elif proba <= 0.75:
            bg_color = "#FEF9E7"
            border_color = "#F4D03F"
            text_color = "#7D6608"
            status_title = "HỒ SƠ CẦN THẨM ĐỊNH THÊM (RỦI RO TRUNG BÌNH)"
            status_desc = "⚠️ <b>Khuyến nghị:</b> Nên chuyển hồ sơ sang bước kiểm tra thủ công. Cần yêu cầu bổ sung chứng minh thu nhập hoặc tài sản bảo đảm..."
            bar_color = "#F39C12"
        else:
            bg_color = "#FDEDEC"
            border_color = "#EC7063"
            text_color = "#78281F"
            status_title = "HỒ SƠ BỊ TỪ CHỐI (RỦI RO CAO)"
            status_desc = "❗ <b>Khuyến nghị:</b> Nên từ chối phê duyệt khoản vay. Xác suất xảy ra vỡ nợ hoặc nợ xấu vượt mức chịu đựng rủi ro của doanh nghiệp."
            bar_color = "#C0392B"

        # Thiết kế khối thông tin kết quả dạng Banner
        st.markdown(f"""
            <div class="result-banner" style="--bg-color:{bg_color}; --border-color:{border_color}">
                <h4 class="result-banner__title" style="--text-color:{text_color}">{status_title}</h4>
                <p class="result-banner__desc" style="--text-color:{text_color}">{status_desc}</p>
            </div>
        """, unsafe_allow_html=True)

        # Chia layout: Bên trái đặt biểu đồ Gauge, Bên phải đặt các chỉ số tóm tắt nhanh
        res_col1, res_col2 = st.columns([5, 5])
        
        with res_col1:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = round(proba * 100, 2),
                number = {'suffix': "%", 'font': {'size': 44, 'color': '#2C3E50', 'family': 'Arial'}},
                title = {'text': "Xác suất khách hàng vỡ nợ dự báo", 'font': {'size': 16, 'color': '#5D6D7E'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#BDC3C7"},
                    'bar': {'color': bar_color}, # Màu thanh kim sẽ thay đổi động theo kết quả phân loại
                    'bgcolor': "#F4F6F7",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 17], 'color': '#D4EFDF'},  # Xanh pastel nhạt
                        {'range': [17, 75], 'color': '#FCF3CF'}, # Vàng pastel nhạt
                        {'range': [75, 100], 'color': '#FADBD8'} # Đỏ pastel nhạt
                    ]
                }
            ))
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=50, b=5, pad=0))
            st.plotly_chart(fig, width='stretch')
            
        with res_col2:
            st.markdown(f"""
                <div class="card-metric">
                    <div class="card-metric__list">
                        <div class="card-metric__item">
                            <div class="card-metric__label">
                                <span class="material-icons card-metric__label-icon">speed</span>
                                <span class="card-metric__label-text">Mức độ rủi ro</span>
                            </div>
                            <span style="font-size: 14px; font-weight: 700; color: {text_color};">{proba * 100:.2f}%</span>
                        </div>
                        <div class="card-metric__item">
                            <div class="card-metric__label">
                                <span class="material-icons card-metric__label-icon">account_balance_wallet</span>
                                <span class="card-metric__label-text">Chỉ số Nợ/Thu nhâp (DTI)</span>
                            </div>
                            <span style="font-size: 13px; font-weight: 700; color: #0F172A; background: #F1F5F9; padding: 2px 8px; border-radius: 6px; border: 1px solid #E2E8F0;">{DTIRatio}</span>
                        </div>  
                        <div class="card-metric__item">
                            <div class="card-metric__label">
                                <span class="material-icons card-metric__label-icon">attach_money</span>
                                <span class="card-metric__label-text">Số tiền vay</span>
                            </div>
                            <span style="font-size: 14px; font-weight: 700; color: #0F172A; letter-spacing: -0.01em;">${LoanAmount:,}</span>
                        </div>
                        <div class="card-metric__item">
                            <div class="card-metric__label">
                                <span class="material-icons card-metric__label-icon">calendar_today</span>
                                <span class="card-metric__label-text">Kỳ hạn</span>
                            </div>
                            <span style="font-size: 12px; font-weight: 600; color: #2563EB; background: #EFF6FF; padding: 2px 8px; border-radius: 6px; border: 1px solid #DBEAFE;">{LoanTerm} tháng</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    # --- 5. Hiển thị lịch sử dự đoán ---
    if st.session_state["loan_history"]:
        st.markdown("""
            <h2 class="section-title">
                LỊCH SỬ PHÂN TÍCH
            </h2>
        """, unsafe_allow_html=True
        )
        df_history = pd.DataFrame(st.session_state["loan_history"])

        def highlight_last(s):
            return ['background-color: #EBF5FB' if i == len(s) - 1 else '' for i in range(len(s))]

        st.dataframe(
            df_history.style.apply(highlight_last, axis=0),
            width='stretch'
        )
        
        # --- 6. Hiệu suất mô hình ---
        st.markdown("""
            <h2 class="section-title">
                HIỆU SUẤT MÔ HÌNH
            </h2>
        """, unsafe_allow_html=True
        )
        with st.expander("**Xem chi tiết**"):
            report_df = pd.read_csv(CLASSIFICATION_REPORT_PATH, index_col=0)
            
            p_0 = report_df.loc["0", "precision"] * 100
            r_0 = report_df.loc["0", "recall"] * 100
            p_1 = report_df.loc["1", "precision"] * 100
            r_1 = report_df.loc["1", "recall"] * 100
            acc = report_df.loc["accuracy", "precision"] * 100

            total_support = report_df.loc["macro avg", "support"]
            report_df.loc["accuracy", "support"] = total_support

            report_df.rename(index={
                "0": "Khách hàng không vỡ nợ (Class 0)",
                "1": "Khách hàng vỡ nợ (Class 1)",
                "accuracy": "Độ chính xác toàn cục (Accuracy)",
                "macro avg": "Trung bình cộng (Macro Avg)",
                "weighted avg": "Trung bình có trọng số (Weighted Avg)"
            }, inplace=True)

            styled_df = (
                report_df.style
                .format("{:.2f}", subset=["precision", "recall", "f1-score"])
                .format("{:,.0f}", subset=["support"])
                .set_properties(**{
                    'font-weight': '600',       # Đẩy chữ toàn bảng đậm lên (mức 600)
                    'color': '#2C3E50'          # Đổi màu chữ sang xanh đen đậm thay vì xám mờ
                })
            )
            st.dataframe(styled_df, width='stretch')

            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown(f"""
                <div class="card-report card-report--green">
                    <div class="card-report__title card-report__title--green">
                        <span class="material-icons card-report__icon">verified_user</span>
                        <b class="card-report__title-content">NHÓM KHÔNG VỠ NỢ (CLASS 0)</b>
                    </div>
                    <ul class="card-report__list">
                        <li class="card-report__item">
                            <b>Precision ({p_0:.0f}%):</b> Dự đoán 100 người an toàn &rarr; đúng <b>{p_0:.0f}</b> người (sai {100 - p_0:.0f} người).
                        </li>
                        <li class="card-report__item">
                            <b>Recall ({r_0:.0f}%):</b> Có 100 người an toàn thực tế &rarr; nhận diện đúng <b>{r_0:.0f}</b> người (bỏ sót {100 - r_0:.0f} người).
                        </li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with analysis_col2:
                st.markdown(f"""
                <div class="card-report card-report--red">
                    <div class="card-report__title card-report__title--red">
                        <span class="material-icons card-report__icon">gpp_bad</span>
                        <b class="card-report__title-content">NHÓM VỠ NỢ (CLASS 1)</b>
                    </div>
                    <ul class="card-report__list">
                        <li class="card-report__item">
                            <b>Precision ({p_1:.0f}%):</b> Dự đoán 100 người vỡ nợ &rarr; đúng <b>{p_1:.0f}</b> người (sai {100 - p_1:.0f} người).
                        </li>
                        <li class="card-report__item">
                            <b>Recall ({r_1:.0f}%):</b> Có 100 người vỡ nợ thực tế &rarr; nhận diện đúng <b>{r_1:.0f}</b> người (bỏ sót {100 - r_1:.0f} người).
                        </li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div class="card-report card-report--yellow">
                <div class="card-report__title card-report__title--yellow">
                    <span class="material-icons card-report__icon">tips_and_updates</span>
                    <b class="card-report__title-content">NHẬN XÉT & GIẢI PHÁP</b>
                </div>
                <ul class="card-report__list">
                    <li><b>Phân loại mặc định (Ngưỡng 50%):</b> <code>&ge;50%</code> &rarr; Vỡ nợ (Class 1) | <code>&lt;50%</code> &rarr; An toàn (Class 0).</li>
                    <li><b>Thách thức & Hệ quả:</b> Dữ liệu mất cân bằng nặng (Nợ xấu chỉ chiếm <b>11.6%</b>) &rarr; Mô hình bị thiên vị, dễ bỏ sót ca vỡ nợ thực tế.</li>
                    <li><b>Giải pháp 3 vùng quyết định:</b>
                        <ul class="card-report__sublist">
                            <li><b>Vùng An toàn (xác suất vỡ nợ &lt; 17%):</b> Độ chính xác &ge; 95% &rarr; Auto duyệt.</li>
                            <li><b>Vùng Từ chối (xác suất vỡ nợ &gt; 75%):</b> Độ chính xác &ge; 70% &rarr; Auto loại.</li>
                            <li><b>Vùng Thẩm định lại (xác suất vỡ nợ: 17% - 75%):</b> Gom hồ sơ mập mờ &rarr; Chuyển duyệt tay.</li>
                        </ul>
                    </li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # --- 7. Thông tin thêm ---
        st.markdown("""
            <h2 class="section-title">
                THÔNG TIN THÊM
            </h2>
        """, unsafe_allow_html=True
        )
        with st.expander("**Xem chi tiết**"):
            st.markdown("""
            <div style='font-size: 15px'>
                <h5>Bộ dữ liệu</h5>
                <ul>
                    <li><b>Nguồn gốc:</b> Bộ dữ liệu được trích xuất từ cuộc thi thử thách công nghệ chính thức của <b>Coursera</b> và được chia sẻ lại trên nền tảng khoa học dữ liệu <a href="https://www.kaggle.com/datasets/nikhil1e9/loan-default" target="_blank"><b>Kaggle bởi tác giả Nikhil</b></a>.</li>
                    <li><b>Quy mô:</b> Gồm <b>255,347 hồ sơ</b> khách hàng vay vốn.</li>
                    <li><b>Mục đích:</b> Mô phỏng lại bài toán <b>Chấm điểm tín dụng (Credit Scoring)</b> kinh điển tại các ngân hàng thương mại: Dựa vào thông tin cá nhân và lịch sử tài chính để dự đoán xem ai sẽ trả nợ đúng hạn (0) và ai sẽ bùng nợ (1).</li>
                    <li><b>Đặc điểm cốt lõi:</b> Bộ dữ liệu bị <b>mất cân bằng nghiêm trọng</b> khi tỷ lệ người bùng nợ chỉ chiếm <b>11.6%</b>. Điều này phản ánh hoàn hảo thực tế ngành tài chính (khách hàng tốt luôn chiếm đại đa số, nợ xấu luôn là thiểu số) và là bài toán kiểm tra năng lực sàng lọc rủi ro của AI.</li>
                    <li><b>Tính thực tế & Hướng phát triển:</b> Do đặc thù ngành Ngân hàng tại Việt Nam bảo mật thông tin cực kỳ nghiêm ngặt, việc tiếp cận các bộ dữ liệu tín dụng nội địa công khai là rất khó khăn. Dù bộ dữ liệu hiện tại mang tính chất mô phỏng, nhưng <b>toàn bộ quy trình tiền xử lý (Pipeline), kỹ thuật xử lý mất cân bằng và thuật toán XGBoost</b> trong dự án này hoàn toàn có thể đóng gói để áp dụng trực tiếp cho bất kỳ tập dữ liệu thực tế nào. Hướng nghiên cứu tiếp theo của dự án sẽ là tối ưu hóa và thử nghiệm mô hình này trên các nguồn dữ liệu thực tế tại các tổ chức tài chính Việt Nam khi có cơ hội tiếp cận.</li>
                </ul>
                <hr> 
                <h5>Mô hình</h5>
                <ol>
                    <li><b>Lựa chọn Thuật toán: Tại sao lại là XGBoost Classifier?</b></li>
                        <ul style='font-size: 15px; padding-left: 20px;'>
                            <li><b>Thử nghiệm đa mô hình:</b> Dự án không chọn ngay thuật toán mà đã thử nghiệm qua nhiều mô hình với thư viện <b>LazyPredict</b>, từ đó chọn ra các mô hình tối ưu nhất và tinh chỉnh tham số. <b>XGBoost</b> được lựa chọn cuối cùng vì cho hiệu suất vượt trội hơn cả.</li>
                            <li><b>Cơ chế học sửa sai (Gradient Boosting)</b>: Thay vì bỏ phiếu độc lập, XGBoost xếp hàng trăm cây quyết định theo chuỗi tuần hoàn. Cây phía sau sẽ nhìn vào những ca đoán sai của cây phía trước để tập trung học lại, giúp phát hiện ra các hành vi quỵt nợ tinh vi ẩn sau hồ sơ đẹp.</li>
                        </ul>
                    <br>
                    <li><b>Quy trình Tiền xử lý dữ liệu tự động (Pipeline)</b></li>
                        <ul style='font-size: 15px; padding-left: 20px;'>
                            <li><b>Chuẩn hóa số liệu (StandardScaler):</b> Đưa các biến chênh lệch lớn (Thu nhập, Hạn mức,...) về cùng một thước đo để đánh giá trọng số rủi ro công bằng.</li>
                            <li><b>Mã hóa định dạng chữ (Encoder):</b> Tự động dịch các thông tin định tính (Học vấn, Việc làm, Mục đích vay,...) sang dạng số để máy tính làm toán.</li>
                        </ul>
                    <br>
                    <li><b>Tinh chỉnh tham số xử lý mất cân bằng dữ liệu (scale_pos_weight=2.1)</b></li>
                        <ul style='font-size: 15px; padding-left: 20px;'>
                            <li><b>Thách thức dữ liệu thực tế:</b> Tệp dữ liệu bị mất cân bằng nặng (nhóm bùng nợ chỉ chiếm vỏn vẹn <b>11.6%</b>). Nếu để học tự nhiên, AI sẽ bị "lười" và luôn đoán khách hàng là người tốt để ăn gian độ chính xác toàn cục (Accuracy)</li>
                            <li><b>Chiến lược phạt lỗi:</b> Hệ thống được thiết lập tham số ép mô hình phải chịu mức <b>phạt nặng gấp 2.1 lần</b> mỗi khi để "lọt lưới" một ca vỡ nợ (Class 1) so với việc nghi oan một khách hàng tốt (Class 0). Đây là chìa khóa cốt lõi giúp đẩy mạnh năng lực cảnh báo sớm và bảo vệ an toàn vốn cho ngân hàng.</li>
                        </ul> 
                </ol>
                <hr>
                <h5>Mục tiêu</h5>
                Hỗ trợ ra quyết định:
                <ul>
                    <li><b>Duyệt vay siêu tốc, giảm tải thủ công:</b> Tự động nhận diện nhóm khách hàng chắc chắn an toàn để giải ngân ngay lập tức, giúp doanh nghiệp tiết kiệm chi phí và tăng tốc độ phục vụ.</li>
                    <li><b>Chặn đứng nợ xấu, bảo vệ vốn:</b> Phát hiện và loại bỏ sớm các hồ sơ gian lận hoặc có nguy cơ quỵt nợ, bảo vệ túi tiền của doanh nghiệp khỏi nguy cơ mất trắng.</li>
                    <li><b>Tối ưu nguồn thu, không nghi oan khách tốt:</b> Lọc ra các ca mập mờ để chuyển con người thẩm định lại, giúp "minh oan" cho khách hàng tốt nhằm giữ lại nguồn doanh thu lãi vay quý báu.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
