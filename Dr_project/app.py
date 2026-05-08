import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import json
from tensorflow.keras.applications.efficientnet import preprocess_input
import warnings
warnings.filterwarnings("ignore")
# -----------------------------
# LOAD CSS
# -----------------------------
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(page_title="DR Detection", layout="wide")
load_css()

# -----------------------------
# USER AUTH SYSTEM
# -----------------------------
USER_FILE = "users.json"

def load_users():
    try:
        with open(USER_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

# -----------------------------
# SESSION STATE
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -----------------------------
# LOGIN / SIGNUP PAGE
# -----------------------------
if not st.session_state.logged_in:

    st.markdown("<h1>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-powered retinal analysis system</p>", unsafe_allow_html=True)

    menu = st.radio("", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    users = load_users()

    if menu == "Signup":
        if st.button("Create Account"):
            if username in users:
                st.error("User already exists")
            else:
                users[username] = password
                save_users(users)
                st.success("Account created! Please login.")

    if menu == "Login":
        if st.button("Login"):
            if username in users and users[username] == password:
                st.session_state.logged_in = True
                st.session_state.user = username
                st.rerun()
            else:
                st.error("Invalid credentials")

# -----------------------------
# MAIN APP
# -----------------------------
else:

    # HEADER
    st.markdown("<h1>Diabetic Retinopathy Detection</h1>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card">
        <p>Welcome, <b>{st.session_state.user}</b></p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Load model
    model = tf.keras.models.load_model("dr_model_stage1.keras")

    classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    # -----------------------------
    # IMAGE UPLOAD
    # -----------------------------
    st.markdown("### Upload Retinal Image")

    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Crop black borders
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        img = img[y:y+h, x:x+w]

        # Resize
        img_resized = cv2.resize(img, (224, 224))

        # Preprocess
        img_array = np.expand_dims(img_resized, axis=0)
        img_array = preprocess_input(img_array)

        # Prediction
        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])

        # -----------------------------
        # RESULT
        # -----------------------------
        st.markdown("### Result")

        st.markdown(f"""
        <div class="card">
            <h2>{classes[pred_class]}</h2>
            <p>Diagnosis result</p>
        </div>
        """, unsafe_allow_html=True)

        # -----------------------------
        # CHART
        # -----------------------------
        st.markdown("### Prediction Distribution")
        st.bar_chart(preds[0])

        # -----------------------------
        # GRAD-CAM
        # -----------------------------
        def get_gradcam(img_array, model):
            for layer in reversed(model.layers):
                if "conv" in layer.name:
                    last_conv = layer.name
                    break

            grad_model = tf.keras.models.Model(
                [model.inputs],
                [model.get_layer(last_conv).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
            conv_outputs = conv_outputs[0]

            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0)

            if tf.reduce_max(heatmap) != 0:
                heatmap /= tf.reduce_max(heatmap)

            return heatmap.numpy()

        def overlay_heatmap(heatmap, image):
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)

        heatmap = get_gradcam(img_array, model)
        result_img = overlay_heatmap(heatmap, img_resized)

        st.markdown("### Grad Cam Visualization")
        st.image(result_img, use_column_width=True)