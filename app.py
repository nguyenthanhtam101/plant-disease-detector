import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ======================
# 1️⃣ Tiêu đề ứng dụng
# ======================
st.set_page_config(page_title="Rau AI - Phát hiện bệnh trên lá rau", page_icon="🥬")
st.title("🥦 ỨNG DỤNG DỰ ĐOÁN BỆNH TRÊN LÁ RAU")
st.write("Tải ảnh lá rau hoặc chụp ảnh trực tiếp để mô hình dự đoán xem lá có bị bệnh hay không.")

# ======================
# 2️⃣ Load mô hình
# ======================
URL = "https://drive.google.com/uc?id=1TR-XkfhtfTMiBhyzkeyTZG7vVnDmz10F"
MODEL_PATH = "model.h5"

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔽 Đang tải mô hình từ Google Drive..."):
        gdown.download(URL, MODEL_PATH, quiet=False)

if os.path.exists(MODEL_PATH):
    st.success("✅ Mô hình đã được tải thành công!")
else:
    st.error("❌ Không thể tải mô hình — kiểm tra lại link hoặc quyền chia sẻ Google Drive.")

model = tf.keras.models.load_model(MODEL_PATH)

# ======================
# 3️⃣ Chọn ảnh
# ======================
option = st.radio("Chọn cách nhập ảnh:", ("📁 Tải ảnh lên", "📸 Chụp ảnh bằng camera"))

if option == "📁 Tải ảnh lên":
    uploaded_file = st.file_uploader("Chọn ảnh lá rau...", type=["jpg", "jpeg", "png"])
elif option == "📸 Chụp ảnh bằng camera":
    uploaded_file = st.camera_input("Chụp ảnh lá rau")

# ======================
# 4️⃣ Xử lý ảnh và dự đoán
# ======================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Ảnh bạn đã chọn", use_column_width=True)

    # Resize ảnh đúng kích thước mô hình yêu cầu
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # Dự đoán
    preds = model.predict(img_array)
    prob = float(preds[0][0])

    # ======================
    # 5️⃣ Hiển thị kết quả
    # ======================
    if prob >= 0.5:
        st.error(f"🚨 Kết quả: Lá **CÓ THỂ BỊ BỆNH** ({prob*100:.2f}% xác suất)")
    else:
        st.success(f"🌿 Kết quả: Lá **KHỎE MẠNH** ({(1-prob)*100:.2f}% xác suất)")

    st.write("---")
    st.caption("Model: ResNet50 (Fine-tuned) | Framework: TensorFlow + Streamlit")

