import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Wildlife Image Classifier",
    page_icon="🐾",
    layout="wide"
)

# --------------------------------------------------------
# INDIAN FOREST BACKGROUND (Surprise HD)
# --------------------------------------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: linear-gradient(
        rgba(0,0,0,0.55),
        rgba(0,0,0,0.55)
    ), url("https://media.cntraveller.in/wp-content/uploads/2017/10/L2R4856-1366x768.jpg");
    background-size: cover;
    background-position: center;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
.sidebar .sidebar-content {
    background: rgba(15,15,15,0.7);
    border-radius: 12px;
}
h1 { color: #eaffea !important; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------
@st.cache_resource
def load_system():
    cnn = tf.keras.models.load_model("cnn_feature_extractor_resnet_final.h5")

    data = joblib.load("svm_model_resnet_final.joblib")
    svm = data["svm"]
    scaler = data["scaler"]
    class_indices = data["class_indices"]

    idx_to_class = {v: k for k, v in class_indices.items()}
    return cnn, svm, scaler, idx_to_class

cnn, svm, scaler, idx_to_class = load_system()

# --------------------------------------------------------
# EMOJIS YOU LIKED — CLEAN + NATURAL
# --------------------------------------------------------
animal_emoji = {
    "asiatic wildcat": "🐈‍⬛",
    "asiatic lion": "🦁",
    "bengal fox": "🦊",
    "chinkara": "🦌",
    "chital": "🦌",
    "four horned antelope": "🦌",
    "golden jackal": "🐺",
    "honey badger": "🦡",
    "indian leopard": "🐆",
    "snow leopard": "🐆",
    "indian cobra": "🐍",
    "monitor lizard": "🦎",
    "nilgai": "🐐",
    "rudy mongoose": "🦦",
    "striped hyena": "🐾",
    "wild boar": "🐗",
    "asian elephant": "🐘",
    "mugger crocodile": "🐊",
    "rhino": "🦏",
    "tiger": "🐅"
}

def get_emoji(name):
    name = name.lower()
    for key, emoji in animal_emoji.items():
        if key in name:
            return emoji
    return "🐾"

# --------------------------------------------------------
# TITLE (Clean — No 'Powered By' Line)
# --------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🐾 Wildlife Image Classifier</h1>", 
            unsafe_allow_html=True)

# --------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------
st.sidebar.title("🌿 App Info")
st.sidebar.write("This classifier identifies wildlife species using CNN + SVM.")

animal_list = [cls.title() for cls in idx_to_class.values()]

st.sidebar.subheader("Supported Animals")
st.sidebar.success(", ".join(animal_list))

selected_animal = st.sidebar.selectbox(
    "Choose expected animal (Optional):",
    ["Auto-detect"] + animal_list
)

# --------------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------------
uploaded = st.file_uploader(
    "📤 Upload Wildlife Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------------
# IMAGE PROCESSING
# --------------------------------------------------------
def process_image(file):
    img = Image.open(file).convert("RGB")
    resized = img.resize((224, 224))
    arr = img_to_array(resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return img, arr

# --------------------------------------------------------
# PREDICTION PIPELINE
# --------------------------------------------------------
if uploaded:

    img_original, processed_img = process_image(uploaded)

    col1, col2 = st.columns([1, 1])

    # LEFT — Image
    with col1:
        st.image(img_original, caption="📸 Uploaded Image", use_container_width=True)

    # RIGHT — Prediction
    with col2:
        with st.spinner("🔍 Analyzing..."):
            features = cnn.predict(processed_img, verbose=0)
            scaled = scaler.transform(features)
            probs = svm.predict_proba(scaled)[0]

        pred_idx = np.argmax(probs)
        predicted = idx_to_class[pred_idx].title()
        emoji = get_emoji(predicted)
        confidence = probs[pred_idx] * 100

        # Main prediction result
        st.success(f"### {emoji} Predicted Animal: *{predicted}*")
        st.metric("Confidence Level", f"{confidence:.2f}%")
        st.progress(int(confidence))

        # Expected animal match/mismatch
        if selected_animal != "Auto-detect":
            if selected_animal.lower() == predicted.lower():
                st.success(f"✔ Correct! The image likely contains *{predicted}*.")
            else:
                st.error(f"❌ Mismatch! You selected *{selected_animal}*, "
                         f"but the model predicts *{predicted}*.")

    # ----------------------------------------------------
    # PROBABILITY BREAKDOWN (NO BLANK TABS)
    # ----------------------------------------------------
    with st.expander("📊 See probabilities for all classes"):
        st.write("### Top Predictions:")
        
        # Loop through all classes
        for cls, prob in sorted(zip(idx_to_class.values(), probs), 
                                key=lambda x: x[1], reverse=True):
            icon = get_emoji(cls)
            st.write(f"*{icon} {cls.title()} → {prob*100:.2f}%*")
            st.progress(int(prob * 100))

# --------------------------------------------------------
# FOOTER
# --------------------------------------------------------
st.write(
    "<br><center style='color:#eaffea;'>🌿 Built for Wildlife Conservation using AI</center>",
    unsafe_allow_html=True
)