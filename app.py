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
# 🧠 GRAD-CAM CẢI TIẾN
# ======================
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

def get_gradcam(img_array, model, last_conv_layer_name=None):
    """
    Tạo heatmap GradCAM tương thích ResNet50 hoặc CNN khác.
    """
    if last_conv_layer_name is None:
        # Tự động tìm layer cuối có chữ "conv"
        last_conv_layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        # Trường hợp sigmoid (1 class)
        if predictions.shape[-1] == 1:
            class_channel = predictions[:, 0]
        else:
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) == 0:
        return np.zeros_like(heatmap)
    heatmap /= np.max(heatmap)
    return heatmap


def calculate_infection_area_and_contours(heatmap, image_pil, threshold=0.4):
    """
    - Áp ngưỡng để tính % vùng bệnh
    - Làm mượt, tìm contour để khoanh vùng
    """
    img = np.array(image_pil)
    h, w, _ = img.shape

    # Resize heatmap bằng kích thước ảnh
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Tạo mask nhị phân
    mask = (heatmap_resized > threshold).astype(np.uint8)

    # Làm mượt mask để loại bỏ nhiễu
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Tính phần trăm vùng bệnh
    infected_pixels = np.sum(mask)
    total_pixels = mask.size
    infected_percent = (infected_pixels / total_pixels) * 100

    # Tìm contour và vẽ lên ảnh gốc
    img_draw = img.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_draw, contours, -1, (255, 0, 0), 2)

    # Tạo ảnh overlay heatmap
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return infected_percent, mask, overlay, img_draw


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

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    preds = model.predict(img_array)
    prob = float(preds[0][0])

    # ======================
    # 5️⃣ Hiển thị kết quả
    # ======================
    if prob >= 0.2:
    st.error(f"🚨 Kết quả: Lá **CÓ THỂ BỊ BỆNH** ({prob*100:.2f}% xác suất)")

    # --- GradCAM và khoanh vùng bệnh ---
    heatmap = get_gradcam(img_array, model)
    infected_percent, mask, overlay, img_contour = calculate_infection_area_and_contours(
        heatmap, image, threshold=0.4
    )

    st.image(
        [image, Image.fromarray(overlay), Image.fromarray(img_contour)],
        caption=["Ảnh gốc", "Bản đồ vùng bệnh (GradCAM)", "Khoanh vùng bệnh (Contour)"],
        width=300
    )

    st.write(f"**Tỷ lệ vùng bị sâu bệnh:** {infected_percent:.2f}%")

    # --- Gợi ý hành động ---
    if infected_percent > 60:
        st.error("⚠️ Khuyến nghị: Lá bị bệnh nặng, **nên loại bỏ để tránh lây lan.**")
    elif infected_percent < 40:
        st.warning("💡 Khuyến nghị: Bệnh nhẹ, **cắt bỏ phần bệnh** để tránh ảnh hưởng.")
    else:
        st.info("🩺 Mức độ trung bình, **nên theo dõi thêm.**")

else:
    st.success(f"🌿 Kết quả: Lá **KHỎE MẠNH** ({(1-prob)*100:.2f}% xác suất)")
    st.image(image, caption="Ảnh gốc (khỏe mạnh)", width=300)

    st.write("---")
    st.caption("Model: ResNet50 (Fine-tuned) | Framework: TensorFlow + Streamlit")
    st.markdown("---")
    st.caption("📌 Dự đoán chỉ mang tính chất tham khảo và có thể mắc lỗi. Vui lòng kiểm chứng các thông tin quan trọng.")
