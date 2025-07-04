from streamlit_option_menu import option_menu
import streamlit as st

st.set_page_config(page_title="AI HEALTH", page_icon="🤖", layout="wide")
with st.sidebar:
    # --- Header nổi bật ---
    st.markdown("""
    <div style="
        font-size: 24px;
        font-weight: bold;
        color: #1B4F72;
        border-bottom: 3px solid #2980B9;
        padding-bottom: 6px;
        margin-bottom: 18px;
        text-align: left;
        animation: slideIn 1s ease;
    ">
        🤖 AI Health
    </div>

    <style>
    @keyframes slideIn {
      from {opacity: 0; transform: translateX(-20px);}
      to {opacity: 1; transform: translateX(0);}
    }
    </style>
    """, unsafe_allow_html=True)

    # --- CSS chung ---
    st.markdown("""
    <style>
    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: 	#2E86C1;
        margin: 14px 0 10px;
        border-bottom: 1px solid #D6EAF8;
        padding-bottom: 4px;
    }
    .info-card {
        animation: fadeIn 1s ease-in-out;
        background-color: #fefefe;
        padding: 14px 18px;
        border-left: 4px solid #5DADE2;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        font-size: 14px;
        line-height: 1.65;
        color: #333;
        margin-bottom: 12px;
        text-align: justify;
    }
    .info-card:hover {
        background-color: #f4faff;
        transform: scale(1.015);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    .author-card {
        background-color: #fefefe;
        border-left: 4px solid #f5b041;
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        font-size: 14px;
        line-height: 1.65;
        font-weight: 500;
    }
    .author-card:hover {
        background-color: #FFFDF4;
        transform: scale(1.015);
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Menu chọn mô hình ---
    st.markdown("<div class='section-title'>🔎 Chọn mô hình</div>", unsafe_allow_html=True)
    selected_app = option_menu(
        menu_title=None,
        options=["Chi phí y tế", "Tiểu đường", "Bệnh tim"],
        icons=["cash-coin", "droplet-half", "heart-pulse"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fff"},
            "icon": {"color": "#2E86C1", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "--hover-color": "#EAF2F8",
                "transition": "all 0.3s ease-in-out",
                "border-radius": "6px"
            },
            "nav-link-selected": {
                "background-color": "#AED6F1",
                "font-weight": "bold",
                "color": "#1B4F72",
                "border-radius": "6px"
            },
        }
    )

    # --- Giới thiệu ---
    st.markdown("<div class='section-title'>📘 Giới thiệu</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <strong>AI-Health</strong> là ứng dụng hỗ trợ <b>dự đoán xu hướng chi phí y tế</b> và <b>khả năng mắc bệnh</b> bằng trí tuệ nhân tạo. Nhập thông tin cá nhân, hệ thống sẽ ước tính:
        <ul style="padding-left:18px;">
            <li>💰 Chi phí y tế hàng năm (bảo hiểm chi trả)</li>
            <li>🩸 Nguy cơ mắc bệnh tiểu đường</li>
            <li>❤️ Khả năng bị bệnh tim mạch</li>
        </ul>
        <span style="font-size:13px; color:#777;"><b><i>⚠️ Kết quả chỉ mang tính tham khảo và không thay thế tư vấn y tế chuyên môn.</i></b></span>
    </div>
    """, unsafe_allow_html=True)

    # --- Tác giả ---
    st.markdown("<div class='section-title'>👤 Tác giả</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="author-card">
        👨‍💻 <b>Phùng Đình Quang Anh</b><br><br>
        <p style="font-size:13px; color:#888;">
        <b><i>© 2025 Phùng Đình Quang Anh. All rights reserved.</i></b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # --- Nút GitHub ---
    st.markdown("""
    <style>
    a.github-button {
        display: inline-block;
        background-color: #24292e;
        color: white !important;
        text-decoration: none !important;
        padding: 10px 22px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        transition: all 0.25s ease;
    }
    a.github-button:hover {
        background-color: #2f363d;
        transform: translateY(-3px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.25);
    }
    </style>

    <div style="text-align: center; margin-top: 16px;">
        <a href="https://github.com/PhungDinhQuangAnh/streamlit-ai-health" target="_blank" class="github-button">
            🐱 Mã nguồn GitHub
        </a>
    </div>
    """, unsafe_allow_html=True)

# --- Nội dung tương ứng với từng mô hình ---
if selected_app == "Chi phí y tế":
    from Medical_Cost.medical_cost_app import run_medical_cost_app
    run_medical_cost_app()

elif selected_app == "Tiểu đường":
    from Diabetes.diabetes_app import run_diabetes_app
    run_diabetes_app()

elif selected_app == "Bệnh tim":
    from Heart.heart_app import run_heart_app
    run_heart_app()










