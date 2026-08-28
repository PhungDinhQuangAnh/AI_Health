from streamlit_option_menu import option_menu
import streamlit as st
from Loan_Default_Prediction.loan_default_app import run_loan_default_app
from Credit_Card_Segmentation.credit_card_app import run_credit_card_app

# --- Cấu hình trang ---
st.set_page_config(page_title="Banking Intelligence Platform", page_icon=":bank:", layout="wide", initial_sidebar_state="collapsed")

# --- Nạp thư viện Icon ---
st.markdown('<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">', unsafe_allow_html=True)
st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">', unsafe_allow_html=True)

# --- CSS Internal ---
st.markdown("""
    <style>
        .sidebar-header {
            font-size: 20px;
            font-weight: 800;
            color: #1B4F72;
            border-bottom: 3px solid #2980B9;
            text-align: center;
            padding-bottom: 5px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .sidebar-section {
            font-size: 20px;
            font-weight: 600;
            color: #2E86C1;
            margin: 5px 0 10px 0;
            padding-bottom: 4px;
            border-bottom: 1px solid #D6EAF8;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .card {
            background-color: #ffffff;
            padding: 16px 18px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
            font-size: 15px;
            line-height: 1.6;
            margin-bottom: 25px;
            transition: all 0.25s ease-in-out;
        }       
        .card--bordered {
            border-left: 4px solid #5DADE2;
        }
        .card:hover {
            background-color: #f2fbff;;
            box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        }
        .card__list {
            padding-left: 10px;
            margin: 10px 0;
        }   
        .card__list-item {
            margin-bottom: 6px;
            text-align: justify;
        }
        .card__warning {
            font-size: 13px;
            color: #666;
        }
        .card__author-name {
            color: #1B4F72;
            font-weight: 700;
            font-size: 16px;
        }
        .card__copyright {
            font-size: 14px;
            color: #888888;
            margin-top: 4px;
        }
        .github-btn {
            text-align: center;
            margin-top: 5px;
        }
        .github-btn__link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            background-color: #24292e;
            color: #ffffff !important;
            text-decoration: none !important;
            padding: 10px 24px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
            transition: all 0.25s ease-in-out;
        }
        .github-btn__link:hover {
            background-color: #2f363d;
            transform: translateY(-3px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        }
        section[data-testid="stSidebar"] {
            box-shadow: 4px 0px 16px rgba(0, 0, 0, 0.15) !important;
        }     
        div[data-testid="stTabs"] [role="tablist"] {
            display: flex !important;
            width: 100% !important;
            justify-content: space-around !important;
        }
        @keyframes lightMenuPulse {
            0% {
                box-shadow: 0 0 0 0 rgba(46, 134, 193, 0.45);
            }
            70% {
                box-shadow: 0 0 0 9px rgba(46, 134, 193, 0);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(46, 134, 193, 0);
            }
        }
        button[data-testid="stExpandSidebarButton"] {
            position: fixed !important;
            top: 14px !important;
            left: 14px !important;
            z-index: 999999 !important;
            background-color: #FFFFFF !important;         
            border: 2px solid #2E86C1 !important;         
            border-radius: 20px !important;             
            padding: 5px 14px 5px 10px !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            animation: lightMenuPulse 2.2s infinite !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        button[data-testid="stExpandSidebarButton"]::after {
            content: "MENU";
            color: #1B4F72;                                  
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0.6px;
        }
        button[data-testid="stExpandSidebarButton"] span {
            color: #2E86C1 !important;     
            font-size: 1.3rem !important;
        }
        button[data-testid="stExpandSidebarButton"]:hover {
            animation: none !important;                       
            background-color: #E6F7FF !important;
            border-color: #1B4F72 !important;              
            transform: translateY(-2px) scale(1.03) !important;
            box-shadow: 0 6px 16px rgba(46, 134, 193, 0.25) !important;
        }
        @media (min-width: 1300px) {
            section[data-testid="stSidebar"] {
                position: relative !important;
                height: 100vh !important;
            }
        }
    </style>
""", unsafe_allow_html=True)

# --- Phần Sidebar ---
with st.sidebar:
    # 1. Header chính
    st.markdown("""
        <div class="sidebar-header">
            <span class="material-icons" style="font-size: 26px;">insights</span>
            <span>Banking Intelligence</span>
        </div>
    """, unsafe_allow_html=True)

    # 2. Section: Chọn mô hình
    st.markdown("""
        <div class="sidebar-section">
            <span class="material-icons">search</span>
            <span>Chọn mô hình</span>
        </div>
    """, unsafe_allow_html=True)

    selected_app = option_menu(
        menu_title=None,
        options=["Phân Khúc Khách Hàng Thẻ Tín Dụng", "Thẩm Định Rủi Ro & Duyệt Hồ Sơ Vay vốn"],
        icons=["people", "shield-check"],
        default_index=0,
        styles={
            "container": {
                "padding": "6px !important", 
                "background-color": "#ffffff",
                "border-left": "4px solid #5DADE2", 
                "border-radius": "10px", 
                "box-shadow": "0 2px 6px rgba(0,0,0,0.04)"
            },
            "icon": {
                "color": "#2E86C1", 
                "font-size": "18px"
            },
            "nav-link": {
                "font-size": "14px",
                "text-align": "left",
                "font-weight": "500",
                "color": "#566573",
                "padding": "10px 12px",
                "margin": "4px 0",
                "--hover-color": "#E6F7FF",
                "transition": "all 0.3s ease-in-out",
                "border-radius": "8px"
            },
            "nav-link-selected": {
                "background-color": "#AED6F1",
                "font-weight": "700",
                "color": "#1B4F72",
                "border-radius": "8px",
                "box-shadow": "0 2px 6px rgba(41, 128, 185, 0.15)"
            },
        }
    )

    # 3. Section: Giới thiệu
    st.markdown("""
        <div class="sidebar-section">
            <span class="material-icons">info</span>
            <span>Giới thiệu</span>
        </div>
        <div class="card card--bordered">
            <b>Banking Intelligence Platform</b> là nền tảng tích hợp <b>thuật toán AI</b> với <b>mục tiêu</b> hỗ trợ doanh nghiệp:
            <ul class="card__list">
                <li class="card__list-item">Phân khúc chủ thẻ tín dụng dựa trên hành vi tài chính, nhằm tăng trưởng doanh số & kích cầu chi tiêu.</li>
                <li class="card__list-item">Thẩm định rủi ro, giảm tải thủ công, tự động duyệt hồ sơ vay vốn an toàn & từ chối hồ sơ vay vốn rủi ro cao.</li>
            </ul>
            <span class="card__warning">
                ⚠️ <i>Kết quả mang tính tham khảo, hỗ trợ ra quyết định!</i>
            </span>
        </div>
    """, unsafe_allow_html=True)

    # 4. Section: Tác giả
    st.markdown("""
        <div class="sidebar-section">
            <span class="material-icons">person</span>
            <span>Tác giả</span>
        </div>
        <div class="card card--bordered">
            <div class="card__author-name">Phùng Đình Quang Anh</div>
            <div class="card__copyright">© 2026 All rights reserved.</div>
        </div>
    """, unsafe_allow_html=True)

    # 5. Footer: Nút GitHub
    st.markdown("""
        <div class="github-btn">
            <a href="https://github.com/PhungDinhQuangAnh/Banking_Intelligence_Platform" target="_blank" class="github-btn__link">
                <i class="fa-brands fa-github" style="font-size: 20px;"></i>
                <span>Mã nguồn GitHub</span>
            </a>
        </div>
    """, unsafe_allow_html=True)

# --- Điều hướng chức năng ---
if selected_app == "Phân Khúc Khách Hàng Thẻ Tín Dụng":
    run_credit_card_app()
elif selected_app == "Thẩm Định Rủi Ro & Duyệt Hồ Sơ Vay vốn":
    run_loan_default_app()
