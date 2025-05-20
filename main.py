import streamlit as st
import tensorflow as tf
import numpy as np
import datetime
import pandas as pd
from collections import Counter
import joblib

rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# ========== Translations ==========
languages = {
    "English": {
        "title": "Plant Disease Recognition System",
        "upload": "Choose an Image:",
        "webcam": "📷 Or capture image from your webcam:",
        "show_image": "Show Image",
        "predict": "Predict",
        "result": "🔍 Model Prediction",
        "final": "✅ Final Prediction:",
        "summary": "🧾 Prediction Summary (This Session)",
        "download": "📥 Download CSV Report",
        "chart": "📊 Prediction Frequency",
        "home_intro": "Welcome to the Plant Disease Recognition System! 🌿🔍",
        "about": "About",
    },
    "Amharic": {
        "title": "የእፅዋት በሽታ መለየት ሲስተም",
        "upload": "ምስል ይመልከቱ፡",
        "webcam": "📷 ወይም ከካሜራዎ ያንሱ፡",
        "show_image": "ምስል አሳይ",
        "predict": "ትንተና አድርግ",
        "result": "🔍 የሞዴሉ ትንተና",
        "final": "✅ የመጨረሻ ትንተና:",
        "summary": "🧾 የትንተና ማጠቃለያ (በዚህ ክፍለ-ጊዜ)",
        "download": "📥 ሪፖርት አውርድ",
        "chart": "📊 የትንተና መደበኛነት",
        "home_intro": "እንኳን ወደ የእፅዋት በሽታ መለያ ሲስተም በደህና መጡ! 🌿🔍",
        "about": "ስለ ፕሮጀክቱ",
    }
}

# ========== Language Selection ==========
st.sidebar.title("Dashboard")
language = st.sidebar.selectbox("🌐 Language", list(languages.keys()))
txt = languages[language]

# ========== Load Model ==========
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

model = load_model()

# ========== Prediction Function ==========
def model_prediction(image_file, return_all=False):
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(128, 128))
    arr   = tf.keras.preprocessing.image.img_to_array(image)
    batch = np.expand_dims(arr, axis=0)
    preds = model.predict(batch)
    return preds if return_all else np.argmax(preds)

# ========== Page Navigation ==========
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ========== Home Page ==========
if app_mode == "Home":
    st.header(txt["title"])
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(f"### {txt['home_intro']}")

# ========== About Page ==========
elif app_mode == "About":
    st.header(txt["about"])
    st.markdown("""
        #### About Dataset
        This dataset is recreated using offline augmentation from the original dataset.
        It has ~87K RGB images across 38 classes.
        - **Train:** 70295  
        - **Validation:** 17572  
        - **Test:** 33  
    """)

# ========== Disease Recognition Page ==========
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # init session
    if "history" not in st.session_state:
        st.session_state.history = []
    if "detailed_history" not in st.session_state:
        st.session_state.detailed_history = []

    # file upload
    test_image = st.file_uploader(txt["upload"], type=["jpg","jpeg","png"])
    # built-in camera input
    st.markdown(f"### {txt['webcam']}")
    camera_image = st.camera_input("")
    if camera_image and st.button("Cancel Webcam"):
        camera_image = None
    source = test_image or camera_image

    if source:
        if st.button(txt["show_image"]):
            st.image(source, use_column_width=True)

        if st.button(txt["predict"]):
            st.snow()
            st.write(f"**{txt['result']}**")

            preds = model_prediction(source, return_all=True)
            idx   = np.argmax(preds)
            model = load_model()

# Add this:
            from tensorflow.keras import Model
            feature_model = Model(inputs=model.input, outputs=model.get_layer("flatten").output)
             # ========== Class Labels ==========
            class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

            # Extract features from CNN (same as before)
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import image

            img = image.load_img(source, target_size=(128, 128))
            img_array = image.img_to_array(img) / 255.
            img_array = np.expand_dims(img_array, axis=0)

            # Use the CNN feature extractor
            features = feature_model.predict(img_array)
            features_scaled = scaler.transform(features)

            # Predict with Random Forest
            rf_pred = rf_model.predict(features_scaled)[0]

            # Map index to class name
            rf_label = class_name[rf_pred]
            st.info(f"🌲 Random Forest Prediction: **{rf_label}**")


            # list of class labels
            # ========== Class Labels ==========
            class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]

            # top-3
            for i in preds[0].argsort()[-3:][::-1]:
                st.write(f"**{class_name[i]}**: {preds[0][i]*100:.2f}%")

            label = class_name[idx]
            st.success(f"{txt['final']} **{label}**")

            # record
            conf = float(np.max(preds))
            now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.detailed_history.append({
                "Predicted Class": label,
                "Confidence (%)": f"{conf*100:.2f}",
                "Timestamp": now
            })
            st.session_state.history.append(label)

    # summary table + CSV
    if st.session_state.detailed_history:
        st.subheader(txt["summary"])
        df = pd.DataFrame(st.session_state.detailed_history)
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(txt["download"], csv, "prediction_summary.csv", "text/csv")

    # frequency chart
    if st.session_state.history:
        st.subheader(txt["chart"])
        counts = Counter(st.session_state.history)
        freq_df = pd.DataFrame.from_dict(counts, orient="index", columns=["Count"])
        st.bar_chart(freq_df)
