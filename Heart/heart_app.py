def run_heart_app():
    import streamlit as st
    import pandas as pd
    import plotly.graph_objects as go
    import joblib
    from datetime import datetime
    import os

    # Đường dẫn an toàn
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(APP_DIR, "model", "heart_model.pkl")
    CLASSIFICATION_REPORT_PATH = os.path.join(APP_DIR, "report", "heart_classification_report.csv")
    CONFUSION_MATRIX_PATH = os.path.join(APP_DIR, "report", "heart_confusion_matrix.jpg")

    # --- 1. Giao diện trang ---
    # st.set_page_config(page_title="Coronary Heart Disease Predictor", layout="wide", page_icon="❤️")
    st.markdown(
        "<h1 style='text-align:center; color:#E74C3C;'>💓 Dự đoán nguy cơ bệnh tim mạch vành</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # --- 2. Load mô hình ---
    model = joblib.load(MODEL_PATH)

    # --- 3. Form nhập liệu người dùng ---
    st.subheader("🔍 Nhập thông tin sức khỏe")
    with st.form("input_form"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 **Cá nhân**",
            "❤️ **Tim mạch**",
            "🩺 **Triệu chứng**",
            "🧪 **Sinh hóa**"
        ])

        # --- 👤 TAB 1: Thông tin cá nhân ---
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Tuổi (Age)", min_value=20, max_value=100, value=40, step=1,
                                      help="Tuổi hiện tại tính theo năm")
            with col2:
                sex = st.radio("Giới tính (Sex)", ["Nam", "Nữ"],
                               help="Giới tính sinh học của bạn", horizontal=True)

        # --- ❤️ TAB 2: Chỉ số tim mạch ---
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                resting_bp = st.number_input("Huyết áp nghỉ (RestingBP)", 80, 200, 115,
                                             help="Huyết áp đo khi nghỉ ngơi (mmHg)")
                max_hr = st.number_input("Nhịp tim tối đa (MaxHR)", 60, 210, 165,
                                         help="Nhịp tim tối đa đạt được khi vận động (bpm)")
            with col2:
                oldpeak = st.number_input("Chênh lệch ST (Oldpeak)", 0.0, 6.0, 0.5, step=0.1,
                                          help="Chênh lệch ST trong điện tim sau vận động")
                st_slope = st.selectbox("Dốc ST (ST_Slope)", ["Đi lên (Up)", "Phẳng (Flat)", "Đi xuống (Down)"],
                                        help="Hình dạng đoạn ST trong ECG khi gắng sức")
            resting_ecg = st.selectbox("Điện tim khi nghỉ (RestingECG)", [
                "Bình thường (Normal)",
                "ST bất thường (ST)",
                "Phì đại thất trái (LVH)"
            ], help="Kết quả điện tâm đồ khi nghỉ ngơi")

            # Cảnh báo
            if resting_bp > 170:
                st.warning("⚠️ Huyết áp khi nghỉ rất cao, kết quả dự đoán có thể thiếu chính xác.")
            if oldpeak > 3.75:
                st.warning("⚠️ Oldpeak cao bất thường, mô hình có thể không phản ánh chính xác.")

        # --- 🩺 TAB 3: Triệu chứng lâm sàng ---
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                chest_pain = st.selectbox("Loại đau ngực (ChestPainType)", [
                    "Đau thắt ngực điển hình (TA)",
                    "Đau thắt ngực không điển hình (ATA)",
                    "Đau không do tim mạch (NAP)",
                    "Không có triệu chứng (ASY)"
                ], help="Loại đau ngực mà bạn thường gặp")
            with col2:
                exercise_angina = st.radio("Đau ngực khi gắng sức?", ["Không", "Có"],
                                           help="Bạn có thấy đau ngực khi tập thể dục hoặc gắng sức?", horizontal=True)

        # --- 🧪 TAB 4: Sinh hóa ---
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 180,
                                              help="Lượng cholesterol toàn phần trong máu")
            with col2:
                fasting_bs = st.radio("Đường huyết lúc đói > 120?", ["Không", "Có"],
                                      help="Chọn 'Có' nếu từng đo đường huyết lúc đói > 120 mg/dL", horizontal=True)

            if cholesterol > 350:
                st.warning("⚠️ Cholesterol cao bất thường, nên kiểm tra sức khỏe sớm.")

        submitted = st.form_submit_button("🎯 **Dự đoán**")

    # --- 4. Dự đoán ---
    if submitted:
        # 4_1. Mapping giá trị về dạng model đã học
        sex_map = {"Nữ": "F", "Nam": "M"}
        bs_map = {"Không": 0, "Có": 1}
        angina_map = {"Không": "N", "Có": "Y"}
        ecg_map = {
            "Bình thường (Normal)": "Normal",
            "ST bất thường (ST)": "ST",
            "Phì đại thất trái (LVH)": "LVH"
        }
        pain_map = {
            "Đau thắt ngực điển hình (TA)": "TA",
            "Đau thắt ngực không điển hình (ATA)": "ATA",
            "Đau không do tim mạch (NAP)": "NAP",
            "Không có triệu chứng (ASY)": "ASY"
        }
        slope_map = {
            "Đi lên (Up)": "Up",
            "Phẳng (Flat)": "Flat",
            "Đi xuống (Down)": "Down"
        }

        # 4_2. Tạo dictionary cho dữ liệu đầu vào
        input_data = {
            "Age": age,
            "Sex": sex_map[sex],
            "ChestPainType": pain_map[chest_pain],
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": bs_map[fasting_bs],
            "RestingECG": ecg_map[resting_ecg],
            "MaxHR": max_hr,
            "ExerciseAngina": angina_map[exercise_angina],
            "Oldpeak": oldpeak,
            "ST_Slope": slope_map[st_slope]
        }
        input_df = pd.DataFrame([input_data])

        # 4_3. Dự đoán xác suất
        proba = model.predict_proba(input_df)[0][1]

        # --- 5. Gauge hiển thị phần trăm nguy cơ ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📈 Kết quả dự đoán")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 2),
            title={'text': "Nguy cơ mắc bệnh tim (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'bar': {'color': "crimson"}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # --- 6. Hiển thị kết quả ---
        if proba < 0.3:
            result = "✅ Nguy cơ THẤP mắc bệnh tim"
            st.success(f"{result} ({proba*100:.2f}%)")
        elif proba < 0.7:
            result = "⚠️ Nguy cơ TRUNG BÌNH mắc bệnh tim"
            st.warning(f"{result} ({proba*100:.2f}%)")
        else:
            result = "❗ Nguy cơ CAO mắc bệnh tim"
            st.error(f"{result} ({proba * 100:.2f}%)")

        # --- 7. Lịch sử dự đoán ---
        # 7_1. Khởi tạo session_state nếu chưa có
        if "heart_history" not in st.session_state:
            st.session_state["heart_history"] = []

        # 7_2. Nếu form đã submit, thêm dòng mới vào lịch sử
        record= {
            "🕒 Thời gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "📈 Xác suất (%)": round(proba * 100, 2),
            "🧑‍⚕️ Giới tính": sex,
            "🎂 Tuổi": age,
            "🩺 Huyết áp nghỉ": resting_bp,
            "🧬 Cholesterol": cholesterol,
            "🩸 Đường huyết lúc đói >120 mg/dL": fasting_bs,
            "📈 Điện tim (ECG)": resting_ecg,
            "❤️‍🔥 Nhịp tim tối đa": max_hr,
            "🏃‍♂️ Đau thắt ngực khi gắng sức": exercise_angina,
            "📉 Oldpeak (ST chênh)": oldpeak,
            "💓 Loại đau ngực": chest_pain,
            "📊 Độ dốc ST": st_slope
        }
        st.session_state["heart_history"].append(record)

        # 7_3. Hiển thị lịch sử dự đoán (gồm cả bản mới nhất)
        if st.session_state["heart_history"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📜 Lịch sử dự đoán")
            df_history = pd.DataFrame(st.session_state["heart_history"])

            # Tô đậm dòng cuối (mới nhất)
            def highlight_last(s):
                return ['background-color: #e0f7fa' if i == len(s) - 1 else '' for i in range(len(s))]

            st.dataframe(
                df_history.style.apply(highlight_last, axis=0),
                use_container_width=True
            )

        # --- 8. Đánh giá hiệu suất mô hình ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🚀 Hiệu suất mô hình")
        tab5, tab6 = st.tabs(["📊 **Confusion Matrix**","📋 **Classification Report**"])
        # 8_1. Confusion Matrix
        with tab5:
            with st.expander("", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(CONFUSION_MATRIX_PATH)
                with col2:
                    st.markdown("""
                    <div background-color: #f9f9f9; border-radius: 5px; border-left: 3px solid #FF4B4B;'>
                    <h5>💡<b>Tóm tắt:</b></h5>
                    <ul>
                      <li>✅ <b>Phát hiện đúng người có bệnh:</b> 99 người</li>
                      <li>❌ <b>Bỏ sót người bệnh:</b> 13 người</li>
                      <li>⚠️ <b>Cảnh báo nhầm người khỏe:</b> 9 người</li>
                      <li>🧍 <b>Phát hiện đúng người khỏe:</b> 63 người</li>
                      <li>➡️ <b>Phát hiện đúng:</b> 162/184 người</li>
                    </ul>
        
                    <h5>📌 <b>Kết luận:</b></h5>
                    <p style='color: #333;'>
                    Mô hình đã học từ dữ liệu y tế thật và hoạt động khá chính xác, nhưng:
                    <ul>
                      <li>🧑‍⚕️ Không thay thế bác sĩ.</li>
                      <li>📍 Nếu hệ thống cảnh báo bạn có nguy cơ, hãy <b>đi khám để xác nhận</b>.</li>
                      <li>😐 Nếu hệ thống nói bạn khỏe, nhưng bạn thấy <b>bất thường</b>, cũng nên đi khám nhé.</li>
                    </ul>
                    </p>
        
                    </div>
                    """, unsafe_allow_html=True)

        # 8_2. Classification Report
        with tab6:
            with st.expander("", expanded=True):
                # Đọc dữ liệu từ file CSV
                report_df = pd.read_csv(CLASSIFICATION_REPORT_PATH, index_col=0)
                report_df.rename(index={
                    "0": "Không mắc bệnh (Class 0)",
                    "1": "Có bệnh (Class 1)",
                    "accuracy": "Độ chính xác (Accuracy)",
                    "macro avg": "Trung bình cộng (Macro Avg)",
                    "weighted avg": "Trung bình có trọng số (Weighted Avg)"
                }, inplace=True)

                # Định dạng bảng đẹp
                styled_df = report_df.style.format("{:.2f}").set_properties(**{
                    'text-align': 'center'
                }).set_table_styles([
                    {"selector": "th", "props": [("text-align", "center")]}
                ])

                # Hiển thị bảng
                st.dataframe(styled_df, use_container_width=True)

                # Phân tích dễ hiểu
                st.markdown("""
                    <h5>📊 Phân tích chi tiết:</h5>
                    <ul>
                        <li>👤 <b>Không mắc bệnh (Class 0):</b>
                        <ul>
                            <li>📏 Precision: {:.0f}% → Trong số dự đoán là <i>không mắc bệnh</i>, có {:.0f}% là đúng.
                            <li>🎯 Recall: {:.0f}% → Trong số <i>thực sự không mắc bệnh</i>, mô hình phát hiện đúng {:.0f}%.
                        </ul>
                        <li>❤️ <b>Có bệnh (Class 1):</b>
                        <ul>
                            <li>📏 Precision: {:.0f}% → Trong số dự đoán là <i>có bệnh</i>, có {:.0f}% là đúng.
                            <li>🎯 Recall: {:.0f}% → Trong số <i>thực sự mắc bệnh</i>, mô hình phát hiện đúng {:.0f}%.
                        </ul>
                    </ul>
                    <h5>📈 Tổng thể:</h5>
                    <ul>
                        <li>✅ Accuracy (độ chính xác tổng thể): {:.0f}%
                        <li>⚖️ Mô hình có độ cân bằng tốt (F1-score trung bình ≈ {:.0f}%)
                        <li>🔍 Ưu tiên phát hiện đúng người có bệnh hơn → phù hợp cho mục tiêu sàng lọc nguy cơ.
                    </ul>
                    """.format(
                    report_df.loc["Không mắc bệnh (Class 0)", "precision"] * 100,
                    report_df.loc["Không mắc bệnh (Class 0)", "precision"] * 100,
                    report_df.loc["Không mắc bệnh (Class 0)", "recall"] * 100,
                    report_df.loc["Không mắc bệnh (Class 0)", "recall"] * 100,

                    report_df.loc["Có bệnh (Class 1)", "precision"] * 100,
                    report_df.loc["Có bệnh (Class 1)", "precision"] * 100,
                    report_df.loc["Có bệnh (Class 1)", "recall"] * 100,
                    report_df.loc["Có bệnh (Class 1)", "recall"] * 100,

                    report_df.loc["Độ chính xác (Accuracy)", "precision"] * 100,
                    report_df.loc["Trung bình cộng (Macro Avg)", "f1-score"] * 100
                    ), unsafe_allow_html=True)

        # --- 9. Thông tin thêm ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📘 Thông tin thêm")
        with st.expander("📖 **Xem chi tiết**"):
            st.markdown("""
            <div style='font-size: 16px; line-height: 1.7; color: #333;'>
            
            <h5>🧠 Về ứng dụng này</h5>
            
            <p>Ứng dụng giúp bạn <b>ước lượng nguy cơ mắc bệnh tim mạch vành</b> – một dạng bệnh tim phổ biến – dựa trên các thông tin sức khỏe cá nhân mà bạn cung cấp.</p>
            <p>Kết quả là <b>một con số phần trăm (%)</b> – càng cao thì nguy cơ mắc bệnh càng lớn.</p>
            <div style="background-color: #fff8e1; padding: 10px; border-left: 5px solid #f39c12; margin-top:10px;">
            ⚠️ <b>Lưu ý:</b> Ứng dụng chỉ mang tính chất tham khảo, không thay thế cho việc khám hoặc chẩn đoán bởi bác sĩ.
            </div>
            
            <hr>
            
            <h5>📚 Bộ dữ liệu</h5>
            <ul>
                <li>🔢 Dữ liệu thật từ hồ sơ y tế của hơn 900 người.</li>
                <li>🌐 Dữ liệu được người dùng <a href="https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction" target="_blank"><b>fedesoriano</b></a> trên Kaggle tổng hợp từ các bộ dữ liệu công khai (Cleveland, Hungary, Switzerland, VA Long Beach, Statlog).</li>
            </ul>
            
            <hr>
            
            <h5>⚙️ Ứng dụng hoạt động thế nào?</h5>
            <ol>
                <li>Bạn nhập các chỉ số sức khỏe vào biểu mẫu.</li>
                <li>Mô hình <b>RandomForestClassifier</b> được huấn luyện với <b>bộ dữ liệu</b> sẽ phân tích và dự đoán nguy cơ.</li>
                <li>Hiển thị kết quả bằng màu sắc và phần trăm dễ hiểu.</li>
            </ol>
            
            <hr>
            
            <h5>🎯 Mục tiêu</h5>
            <p>Ứng dụng này không thay thế bác sĩ mà được xây dựng với các mục tiêu sau:</p>
            <ul>
                <li>🧑‍⚕️ <b>Hỗ trợ chuyên gia y tế</b> trong việc đánh giá nguy cơ bệnh tim mạch dựa trên các chỉ số đã có.</li>
                <li>📊 <b>Tham khảo nhanh</b> cho người đã thực hiện các xét nghiệm cơ bản nhưng chưa được tư vấn rõ ràng.</li>
                <li>🧪 <b>Trình bày kết quả mô hình AI</b> trực quan, dễ hiểu cho mục đích học tập, nghiên cứu hoặc thử nghiệm.</li>
            </ul>
    
            </div>
            """, unsafe_allow_html=True)









