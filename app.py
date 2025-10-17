import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
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
# 🧠 HÀM GRAD-CAM
# ======================
def get_gradcam(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        # Tự động tìm lớp conv cuối cùng
        last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # Với mô hình nhị phân (1 output sigmoid)
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


def overlay_heatmap(image_pil, heatmap, intensity=0.6):
    img = np.array(image_pil)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(img_rgb, 1 - intensity, heatmap_colored, intensity, 0)
    return img, cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

def calculate_infected_area(heatmap, threshold=0.5):
    mask = heatmap > threshold
    percent = np.sum(mask) / mask.size * 100
    return percent

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

        # --- Grad-CAM: hiển thị vùng bị bệnh ---
        heatmap = get_gradcam(img_array, model)
        img_orig, img_overlay = overlay_heatmap(image, heatmap)
        infected_percent = calculate_infected_area(heatmap, 0.5)

        st.image([img_orig, img_overlay],
                 caption=["Ảnh gốc", "Vùng bị sâu bệnh"],
                 width=300)
        st.write(f"**Tỷ lệ vùng bị sâu bệnh:** {infected_percent:.2f}%")

        if infected_percent > 60:
            st.error("⚠️ Khuyến nghị: Lá bị sâu bệnh nặng, **nên bỏ đi** để tránh lây lan.")
        elif infected_percent < 40:
            st.warning("💡 Khuyến nghị: Bị nhẹ, **có thể cắt bỏ phần bệnh** để tránh ảnh hưởng toàn cây.")
        else:
            st.info("🩺 Mức độ trung bình, nên theo dõi thêm.")
    else:
        st.success(f"🌿 Kết quả: Lá **KHỎE MẠNH** ({(1-prob)*100:.2f}% xác suất)")
        st.image(image, caption="Ảnh gốc (khỏe mạnh)", width=300)

    st.write("---")
    st.caption("Model: ResNet50 (Fine-tuned) | Framework: TensorFlow + Streamlit")

    st.markdown("---")
    st.caption("📌 Dự đoán chỉ mang tính chất tham khảo và có thể mắc lỗi. Vui lòng kiểm chứng các thông tin quan trọng.")
