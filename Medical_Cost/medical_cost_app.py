def run_medical_cost_app():
    import streamlit as st
    import joblib
    import pandas as pd
    from datetime import datetime
    import json
    import os
    import numpy as np

    # Đường dẫn an toàn
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(APP_DIR, "model", "medical_cost_model.pkl")
    METRIC_PATH = os.path.join(APP_DIR, "report", "medical_cost_metrics.json")
    PLOT_AVP_PATH = os.path.join(APP_DIR, "report", "actual_vs_predicted.png")
    PLOT_ED_PATH = os.path.join(APP_DIR, "report", "error_distribution.png")

    # --- 1. Giao diện trang ---
    # st.set_page_config(page_title="Medical Cost Predictor", page_icon="💰", layout="wide")
    st.markdown("<h1 style='text-align:center; color:#E67E22;'>💰 Ước tính chi phí y tế hằng năm do bảo hiểm chi trả (Hoa Kỳ)</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # --- 2. Load mô hình ---
    model = joblib.load(MODEL_PATH)

    # --- 3. Form nhập liệu người dùng ---
    st.subheader("📋 Nhập thông tin cá nhân")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input(
                label="📅 Tuổi", min_value=18, max_value=64, value=30, step=1,
                help="Nhập tuổi thực tế từ 18 đến 64")

            bmi = st.number_input(
                label="⚖️ BMI (Chỉ số khối cơ thể)", min_value=15.00, max_value=54.00, value=20.00, step=0.01,
                help="BMI từ 18.5–24.9 là bình thường. <18.5: gầy, >25: thừa cân")

            children = st.selectbox(
                label="👶 Số con phụ thuộc", options=[0, 1, 2, 3, 4, 5],
                help="Chọn số con (dưới 18 tuổi) sống cùng bạn")

        with col2:
            sex = st.radio(
                label="⚧️ Giới tính",
                options=["male", "female"],
                format_func=lambda x: "Nam" if x == "male" else "Nữ",
                horizontal=True,
                help="Giới tính sinh học")

            smoker = st.radio(
                label="🚬 Bạn có hút thuốc không?",
                options=["no", "yes"],  # giữ nguyên để mapping
                format_func=lambda x: "Có" if x == "yes" else "Không",
                horizontal=True,
                help="Chọn 'Có' nếu bạn đang hoặc từng hút thuốc"
            )

            region = st.selectbox(
                label="🌍 Khu vực sinh sống (tại Hoa Kỳ)",
                options=["southeast", "southwest", "northeast", "northwest"],  # giữ nguyên để mapping
                format_func=lambda x: {
                    "southeast": "Đông Nam",
                    "southwest": "Tây Nam",
                    "northeast": "Đông Bắc",
                    "northwest": "Tây Bắc"
                }[x],
                help="Chọn khu vực bạn đang sinh sống tại Hoa Kỳ"
            )

        submitted = st.form_submit_button("🔮 Dự đoán chi phí")

    # --- 4. Dự đoán ---
    if submitted:
        input_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }
        input_df = pd.DataFrame([input_data])
        prediction = np.exp(model.predict(input_df)[0])

        st.subheader("💵 Kết quả dự đoán")
        usd_to_vnd = 25000
        vnd_amount = round(prediction * usd_to_vnd)

        with st.container():
            st.markdown(
                f"""
                <div style='padding: 20px; background-color: #FDEBD0; border-radius: 10px; text-align: center;'>
                    <h2 style='color: #E67E22;'>💰 Dự đoán chi phí bảo hiểm y tế</h2>
                    <h1 style='color: #229954;'>${round(prediction, 2):,.2f}</h1>
                    <p style='font-size: 18px; color: #555;'>≈ {vnd_amount:,.0f} VNĐ</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        # --- 5. Lịch sử dự đoán ---
        # 5_1. Khởi tạo session_state nếu chưa có
        if "medical_history" not in st.session_state:
            st.session_state["medical_history"] = []

        # 5_2. Nếu form đã submit, thêm dòng mới vào lịch sử
        if submitted:
            record = {
                "⏰ Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "💰 Dự đoán chi phí ($)": round(prediction, 2),
                "📅 Tuổi": age,
                "⚧️ Giới tính": "Nam" if sex == "male" else "Nữ",
                "⚖️ BMI": bmi,
                "👶 Số con": children,
                "🚬 Hút thuốc": "Có" if smoker == "yes" else "Không",
                "🌍 Khu vực": {
                    "southeast": "Đông Nam",
                    "southwest": "Tây Nam",
                    "northeast": "Đông Bắc",
                    "northwest": "Tây Bắc"
                }[region]
            }
            st.session_state["medical_history"].append(record)

        # 5_3. Hiển thị lịch sử dự đoán (gồm cả bản mới nhất)
        if st.session_state["medical_history"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📜 Lịch sử dự đoán")
            df_history = pd.DataFrame(st.session_state["medical_history"])

            # Tô đậm dòng cuối (mới nhất)
            def highlight_last(s):
                return ['background-color: #e0f7fa' if i == len(s) - 1 else '' for i in range(len(s))]

            st.dataframe(
                df_history.style.apply(highlight_last, axis=0),
                use_container_width=True
            )

        # --- 6. Hiệu suất mô hình ---
        st.subheader("🚀 Hiệu suất mô hình")

        tab1, tab2 = st.tabs(["📊 Chỉ số đánh giá", "📈 Biểu đồ minh họa"])

        # --- CSS ---
        st.markdown("""
        <style>
        .box {
            padding: 14px 16px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
            margin-bottom: 18px;
            font-size: 14px;
            border-left: 4px solid transparent;
        }
        .box:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        }
        </style>
        """, unsafe_allow_html=True)

        # --- TAB 1: Chỉ số đánh giá ---
        with open(METRIC_PATH, "r") as f:
            metrics = json.load(f)

        mae = round(metrics["MAE"], 2)
        rmse = round(metrics["RMSE"], 2)
        r2 = round(metrics["R2"], 3)

        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    f"""
                    <div class="box" style="background-color:#EAF2F8; border-left-color:#2E86C1;">
                        <h4 style='color:#2E86C1; margin-bottom:5px;'>📈 R² Score</h4>
                        <h3 style='color:#1F618D;'>{r2}</h3>
                        <p>Thể hiện mức độ mô hình giải thích được biến động của chi phí y tế.</p>
                        <p><b>Giải thích:</b> R² = {r2} tương đương mô hình giải thích được ≈ {round(r2 * 100)}% biến động trong dữ liệu chi phí.</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="box" style="background-color:#E9F7EF; border-left-color:#239B56;">
                        <h4 style='color:#239B56; margin-bottom:5px;'>📉 MAE</h4>
                        <h3 style='color:#27AE60;'>{mae:,.0f} USD</h3>
                        <p>Trung bình mỗi dự đoán sai lệch <b>{mae:,.0f} USD (≈ {mae*usd_to_vnd:,.0f} VNĐ)</b> so với thực tế.</p>
                        <p><b>Giải thích:</b> MAE càng thấp thì mô hình càng chính xác và ổn định.</p>
                    </div>
                    """, unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f"""
                    <div class="box" style="background-color:#FCF3CF; border-left-color:#CA6F1E;">
                        <h4 style='color:#CA6F1E; margin-bottom:5px;'>🚨 RMSE</h4>
                        <h3 style='color:#E67E22;'>{rmse:,.0f} USD</h3>
                        <p>Nhấn mạnh sai số lớn – nhạy hơn MAE với điểm bất thường.</p>
                        <p><b>Giải thích:</b> RMSE cao hơn MAE nghĩa là có thể tồn tại vài điểm sai lệch lớn.</p>
                    </div>
                    """, unsafe_allow_html=True
                )

        # --- TAB 2: Biểu đồ đánh giá ---
        with tab2:
            col_left, col_right = st.columns(2)

            # --- Left: Dự đoán vs Thực tế ---
            with col_left:
                st.image(PLOT_AVP_PATH, caption="Biểu đồ: Dự đoán vs Thực tế")
                st.markdown(f"""
                    <div class="box" style="background-color:#FEF9E7; border-left: 5px solid #F4D03F; padding: 10px;">
                        <h5 style="color:#CA6F1E;">🧠 Nhận xét biểu đồ Dự đoán vs Thực tế:</h5>
                        <ul>
                            <li><b>Trục X</b>: Chi phí y tế thực tế</li>
                            <li><b>Trục Y</b>: Chi phí mô hình dự đoán</li>
                            <li><b>Đường đỏ</b>: Đường lý tưởng (dự đoán = thực tế)</li>
                            <li><b>Các chấm</b>: Mỗi cá nhân trong tập kiểm tra</li>
                            <li>✅ Phần lớn điểm nằm gần đường đỏ → mô hình dự đoán khá chính xác, phù hợp với R² cao (~{round(r2 * 100)}%).</li>
                            <li>🟥 Một số điểm lệch dưới khỏi đường chéo ở vùng chi phí cao → mô hình có xu hướng <i>dự đoán thấp</i> trong các trường hợp đặc biệt.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

            # --- Right: Histogram lỗi ---
            with col_right:
                st.image(PLOT_ED_PATH, caption="Biểu đồ: Phân phối sai số")
                st.markdown(f"""
                    <div class="box" style="background-color:#FDF2E9; border-left: 5px solid #DC7633; padding: 10px;">
                        <h5 style="color:#CA6F1E;">📊 Nhận xét biểu đồ phân phối sai số:</h5>
                        <ul>
                            <li><b>Trục X</b>: Sai số (Dự đoán – Thực tế), đơn vị: USD</li>
                            <li><b>Trục Y</b>: Số lượng dự đoán tương ứng</li>
                            <li>✅ Phân phối tập trung mạnh quanh 0, gần chuẩn → mô hình ổn định, ít sai lệch hệ thống.</li>
                            <li>🟧 MAE ≈ {mae:,.0f} và RMSE ≈ {rmse:,.0f} → sai số trung bình ở mức chấp nhận được, nhưng có vài ca sai lệch lớn (-10000 -> -20000).</li>
                            <li>🟥 Đuôi trái dài hơn đuôi phải → mô hình có xu hướng <i>đoán thấp</i> hơn thực tế trong một số trường hợp.</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)

        # --- 9. Thông tin thêm ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📘 Thông tin thêm")
        with st.expander("📖 **Xem chi tiết**"):
            st.markdown("""
            <div style='font-size: 16px; line-height: 1.7; color: #333;'>
    
            <h5>🧠 Về ứng dụng này</h5> 
            <p>Ứng dụng giúp <b>ước tính chi phí bảo hiểm y tế</b> tại Hoa Kỳ dựa trên thông tin cá nhân như độ tuổi, chỉ số BMI, số con, vùng miền, giới tính và tình trạng hút thuốc.</p>
            <p>Dữ liệu được phân tích bằng mô hình <b>Machine Learning</b> để dự đoán chi phí gần đúng mà công ty bảo hiểm sẽ chi trả cho mỗi cá nhân.</p>
    
            <div style="background-color: #fff8e1; padding: 10px; border-left: 5px solid #f39c12; margin-top:10px;">
            ⚠️ <b>Lưu ý:</b> Kết quả chỉ mang tính chất tham khảo và minh hoạ mô hình AI, không đại diện cho chính sách chi trả của bất kỳ tổ chức bảo hiểm nào.
            </div>
    
            <hr>
    
            <h5>📚 Bộ dữ liệu</h5>
            <ul>
                <li>🌐 Dữ liệu lấy từ nền tảng Kaggle: 
                    <a href="https://www.kaggle.com/datasets/mirichoi0218/insurance" target="_blank"><b>Insurance Dataset</b></a>.
                </li>
                <li>📄 Dữ liệu gồm 1,338 dòng – mỗi dòng là hồ sơ của một cá nhân.</li>
                <li>📊 Các biến bao gồm:</li>
            </ul>
            
            <div style="display: flex; flex-wrap: wrap; gap: 8px; margin: -8px 0 10px 20px;">
              <span style="background: #e3f2fd; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 13px;">age</span>
              <span style="background: #e8f5e9; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 13px;">sex</span>
              <span style="background: #fff3e0; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 13px;">bmi</span>
              <span style="background: #f3e5f5; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 13px;">children</span>
              <span style="background: #ede7f6; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 13px;">smoker</span>
              <span style="background: #e0f7fa; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 13px;">region</span>
              <span style="background: #fce4ec; padding: 6px 10px; border-radius: 6px; font-family: monospace; font-size: 13px;">charges</span>
            </div>
            
            <div style="background-color: #f0f9ff; padding: 12px; border-left: 5px solid #3498db; border-radius: 5px; margin-left: 20px;">
                💡 <b>charges</b>: là số tiền (USD) mà bảo hiểm chi trả cho người đó trong một năm dựa trên thông tin sức khỏe và nhân khẩu học.
            </div>
    
    
            <hr>
    
            <h5>⚙️ Ứng dụng hoạt động thế nào?</h5>
            <ol>
                <li>Bạn nhập thông tin sức khỏe cơ bản như tuổi, BMI, vùng miền,...</li>
                <li>Mô hình <b>RandomForestRegressor</b> được huấn luyện với <b>bộ dữ liệu</b> thật sẽ phân tích và đưa ra dự đoán.</li>
                <li>Hệ thống hiển thị kết quả trực quan kèm biểu đồ hiệu suất mô hình để ngươi dùng tham khảo</li>
            </ol>
    
            <hr>
    
            <h5>🎯 Mục tiêu</h5>
            <p>Ứng dụng mang tính minh họa cho:</p>
            <ul>
                <li>🧠 Cách AI có thể ước lượng chi phí dựa trên dữ liệu</li>
                <li>📈 Trình bày kết quả <b>trực quan</b> qua chỉ số và biểu đồ</li>
                <li>🔍 Hỗ trợ học sinh – sinh viên nghiên cứu về mô hình <b>hồi quy</b> và phân tích dữ liệu</li>
            </ul>
    
            </div>
            """, unsafe_allow_html=True)



































