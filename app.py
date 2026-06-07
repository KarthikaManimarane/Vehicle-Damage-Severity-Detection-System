import streamlit as st
import cv2
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
import supervision as sv

from ultralytics import YOLO
import pandas as pd
import xgboost as xgb
import pickle

# Page configuration
st.set_page_config(
    page_title="Car Damage Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .result-card h3 {
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .result-card p {
        margin: 0.5rem 0 0 0;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .severity-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 1rem 0;
        border: 3px solid rgba(255,255,255,0.3);
    }
    .severity-card h2 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .severity-card p {
        margin: 0.5rem 0;
        font-size: 1.2rem;
        opacity: 0.95;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px dashed #cbd5e0;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-badge {
        display: inline-block;
        background-color: #3b82f6;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Cache models
# -----------------------------
@st.cache_resource
def load_models():
    """Load all ML models"""
    try:
        model_angle = load_model('model.keras')
        model_part = load_model('CarPartINExmodel.h5')
        
        yolo_models = {
            "airbag": YOLO("model/airbag.pt"),
            "damage": YOLO("model/damage.pt"),
            "carpart": YOLO("model/carpart.pt")
        }
        
        # Load XGBoost severity model
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb_model.load_model("xgb_model.json")
        
        # Load encoders
        with open('feature_encoders.pkl', 'rb') as f:
            feature_encoders = pickle.load(f)
        
        with open('severity_encoder.pkl', 'rb') as f:
            severity_encoder = pickle.load(f)
        
        return model_angle, model_part, yolo_models, xgb_model, feature_encoders, severity_encoder
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None, None

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(image, size):
    """Preprocess image for classification"""
    img = image.convert('RGB')
    img = np.asarray(img)
    img = cv2.resize(img, (size, size))
    img = img / 255.0
    return img

# -----------------------------
# YOLO Detection
# -----------------------------
def yolo_single_detection(image, model, model_name):
    """Perform YOLO detection on a single image"""
    # Convert PIL to CV2
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run YOLO
    results = model(img_cv)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # Annotate
    annotated_image = img_cv.copy()
    detection_list = []
    detected_classes = []
    
    if len(detections) > 0:
        annotated_image = sv.BoxAnnotator(thickness=2).annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = sv.LabelAnnotator(text_thickness=2, text_scale=0.8).annotate(
            scene=annotated_image, detections=detections
        )
        
        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            label = model.names[int(class_id)]
            detected_classes.append(label)
            detection_list.append({
                "class_name": label,
                "confidence": float(confidence),
                "bbox": [float(x) for x in box]
            })
    
    # Convert back to RGB
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    # Get primary detection
    if detected_classes:
        primary_detection = detected_classes[0]
    else:
        if model_name == "airbag":
            primary_detection = "Not Deployed"
        elif model_name == "damage":
            primary_detection = "No Damage"
        else:
            primary_detection = "Unknown"
    
    return annotated_image, detection_list, primary_detection

# -----------------------------
# Classification
# -----------------------------
def classify_angle_and_part(img_angle, img_part, model_angle, model_part):
    """Classify angle and part from images"""
    # Preprocess
    data_angle = preprocess(img_angle, 331)
    data_part = preprocess(img_part, 224)
    
    # Expand dims
    xtest_angle = np.expand_dims(data_angle, axis=0)
    xtest_part = np.expand_dims(data_part, axis=0)
    
    # Predict
    Y_pred_angle = model_angle.predict([xtest_angle,xtest_angle], verbose=0)
    Y_pred_part = model_part.predict(xtest_part, verbose=0)
    
    # Get predictions
    pred_class_angle = np.argmax(Y_pred_angle, axis=1)[0]
    pred_class_part = np.argmax(Y_pred_part, axis=1)[0]
    
    confidence_angle = np.max(Y_pred_angle) * 100
    confidence_part = np.max(Y_pred_part) * 100
    
    # Map predictions
    part_labels = [
        'Air intake', 'Console', 'Dashboard', 'Fog light', 
        'Gear stick', 'Headlight', 'Steering wheel', 'Tail light'
    ]
    pred_part_name = part_labels[pred_class_part]
    
    angle_label_map = {'0': 0, '130': 1, '180': 2, '230': 3, '270': 4, '320': 5, '40': 6, '90': 7}
    angle_mapping = {
        "right": 0, "front_left": 130, "left": 180, "rear_left": 230,
        "rear": 270, "rear_right": 320, "front_right": 40, "front": 90
    }
    
    index_to_angle = {v: k for k, v in angle_label_map.items()}
    angle_value = int(index_to_angle[pred_class_angle])
    angle_name = {v: k for k, v in angle_mapping.items()}[angle_value]
    
    return pred_part_name, angle_name, angle_value, confidence_part, confidence_angle, data_angle, data_part

# -----------------------------
# Severity Prediction
# -----------------------------
def predict_severity(angle_name, pred_part, damage_type, carpart_name, airbag_status, 
                    xgb_model, feature_encoders, severity_encoder):
    """Predict damage severity using XGBoost model"""
    
    # Determine view_type based on part
    internal_parts = ['Console', 'Dashboard', 'Gear stick', 'Steering wheel', 'Air intake']
    view_type = 'internal' if pred_part in internal_parts else 'external'
    
    # Map airbag status
    if "Deployed" in airbag_status or "deployed" in airbag_status.lower():
        airbag_mapped = "deployed"
    else:
        airbag_mapped = "not_applicable"
    
    # Create dataframe
    data = pd.DataFrame({
        'view_type': [view_type],
        'angle': [angle_name],
        'part': [carpart_name],
        'damage': [damage_type],
        'airbag': [airbag_mapped]
    })
    
    # Encode features
    for col in ['view_type', 'angle', 'part', 'damage', 'airbag']:
        if col in feature_encoders:
            le = feature_encoders[col]
            try:
                data[col] = le.transform(data[col])
            except ValueError:
                # Handle unknown labels - use a default value
                data[col] = 0
    
    # Predict severity
    try:
        y_pred_encoded = xgb_model.predict(data)
        y_pred_decoded = severity_encoder.inverse_transform(y_pred_encoded)
        severity = y_pred_decoded[0]
        
        # Get prediction probability
        y_pred_proba = xgb_model.predict_proba(data)
        confidence = np.max(y_pred_proba) * 100
        
        return severity, confidence, view_type
    except Exception as e:
        return "Unable to predict", 0.0, view_type

# -----------------------------
# Main App
# -----------------------------
def main():
    st.markdown('<h1 class="main-header">🚗 Complete Car Damage Analysis System</h1>', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        model_angle, model_part, yolo_models, xgb_model, feature_encoders, severity_encoder = load_models()
    
    if model_angle is None:
        st.error("❌ Failed to load models. Please check model files.")
        return
    
    st.success("✅ All AI models loaded successfully!")
    
    # Sidebar Instructions
    # 🤖 KANMANI
    st.markdown("### 🤖 KANMANI Assistant")

    st.link_button(
    "Launch KANMANI",
    "http://127.0.0.1:5000"
        )
    st.sidebar.header("📋 Upload Instructions")
    st.sidebar.markdown("""
    ### Required Images (5 Total):
    
    1️⃣ **Airbag Image** - For airbag deployment detection
    
    2️⃣ **Damage Image** - For damage type detection
    
    3️⃣ **Vehicle Part Image** - For car part detection
    
    4️⃣ **Angle Image** - For camera angle classification
    
    5️⃣ **Part Classification Image** - For internal/external part classification
    
    ---
    
    ### Analysis Output:
    - ✅ Angle Classification
    - ✅ Part Type
    - ✅ Damage Type
    - ✅ Vehicle Part
    - ✅ Airbag Status
    - ✅ **Damage Severity**
    """)
    
    # Main Upload Section
    st.markdown('<div class="sub-header">📤 Upload 5 Images for Complete Analysis</div>', unsafe_allow_html=True)
    
    # Create upload sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 YOLO Detection Images")
        
        airbag_file = st.file_uploader(
            "1️⃣ Airbag Image", 
            type=['jpg', 'jpeg', 'png'], 
            key="airbag",
            help="Upload image to detect airbag deployment"
        )
        
        damage_file = st.file_uploader(
            "2️⃣ Damage Image", 
            type=['jpg', 'jpeg', 'png'], 
            key="damage",
            help="Upload image to detect damage type"
        )
        
        carpart_file = st.file_uploader(
            "3️⃣ Vehicle Part Image", 
            type=['jpg', 'jpeg', 'png'], 
            key="carpart",
            help="Upload image to detect car part"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 Classification Images")
        
        angle_file = st.file_uploader(
            "4️⃣ Angle Image", 
            type=['jpg', 'jpeg', 'png'], 
            key="angle",
            help="Upload image for angle classification"
        )
        
        part_file = st.file_uploader(
            "5️⃣ Part Classification Image", 
            type=['jpg', 'jpeg', 'png'], 
            key="part",
            help="Upload image for part type classification"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if all images uploaded
    all_uploaded = all([airbag_file, damage_file, carpart_file, angle_file, part_file])
    
    if all_uploaded:
        st.markdown('<div class="success-box">✅ All 5 images uploaded successfully! Ready for analysis.</div>', unsafe_allow_html=True)
    else:
        uploaded_count = sum([bool(airbag_file), bool(damage_file), bool(carpart_file), bool(angle_file), bool(part_file)])
        st.info(f"📊 {uploaded_count}/5 images uploaded. Please upload all 5 images to run the analysis.")
    
    # Analyze Button
    st.markdown("---")
    
    if st.button("🚀 RUN COMPLETE ANALYSIS", type="primary", use_container_width=True, disabled=not all_uploaded):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Initialize results
        results = {}
        
        # 1. Airbag Detection
        status_text.text("🔍 Analyzing airbag status...")
        progress_bar.progress(16)
        img_airbag = Image.open(airbag_file)
        airbag_annotated, airbag_detections, airbag_status = yolo_single_detection(
            img_airbag, yolo_models["airbag"], "airbag"
        )
        results['airbag'] = {'annotated': airbag_annotated, 'detections': airbag_detections, 'status': airbag_status}
        
        # 2. Damage Detection
        status_text.text("🔍 Detecting damage type...")
        progress_bar.progress(33)
        img_damage = Image.open(damage_file)
        damage_annotated, damage_detections, damage_type = yolo_single_detection(
            img_damage, yolo_models["damage"], "damage"
        )
        results['damage'] = {'annotated': damage_annotated, 'detections': damage_detections, 'type': damage_type}
        
        # 3. Car Part Detection
        status_text.text("🔍 Identifying vehicle part...")
        progress_bar.progress(50)
        img_carpart = Image.open(carpart_file)
        carpart_annotated, carpart_detections, carpart_name = yolo_single_detection(
            img_carpart, yolo_models["carpart"], "carpart"
        )
        results['carpart'] = {'annotated': carpart_annotated, 'detections': carpart_detections, 'name': carpart_name}
        
        # 4. Angle and Part Classification
        status_text.text("🎯 Classifying angle and part type...")
        progress_bar.progress(66)
        img_angle = Image.open(angle_file)
        img_part = Image.open(part_file)
        pred_part, angle_name, angle_value, conf_part, conf_angle, data_angle, data_part = classify_angle_and_part(
            img_angle, img_part, model_angle, model_part
        )
        results['classification'] = {
            'part': pred_part, 
            'angle_name': angle_name, 
            'angle_value': angle_value,
            'conf_part': conf_part,
            'conf_angle': conf_angle,
            'img_angle': data_angle,
            'img_part': data_part
        }
        
        # 5. PREDICT SEVERITY
        status_text.text("🎯 Predicting damage severity...")
        progress_bar.progress(83)
        severity, severity_confidence, view_type = predict_severity(
            angle_name=angle_name,
            pred_part=pred_part,
            damage_type=damage_type,
            carpart_name=carpart_name,
            airbag_status=airbag_status,
            xgb_model=xgb_model,
            feature_encoders=feature_encoders,
            severity_encoder=severity_encoder
        )
        results['severity'] = {
            'level': severity,
            'confidence': severity_confidence,
            'view_type': view_type
        }
        
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Display Results
        st.markdown('<div class="sub-header">📊 Complete Analysis Results</div>', unsafe_allow_html=True)
        
        # SEVERITY DISPLAY - PROMINENT POSITION
        st.markdown("### 🚨 Damage Severity Assessment")
        
        # Color coding based on severity
        severity_colors = {
            'minor': "linear-gradient(135deg, #56ab2f 0%, #a8e063 100%)",
            'moderate': "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
            'severe': "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
        }
        severity_color = severity_colors.get(severity.lower(), "linear-gradient(135deg, #667eea 0%, #764ba2 100%)")
        
        st.markdown(f"""
        <div class="severity-card" style="background: {severity_color};">
            <h2>⚠️ {severity.upper()}</h2>
            <p>Confidence: {severity_confidence:.1f}%</p>
            <p>View Type: {view_type.title()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary Cards
        st.markdown("### 🎯 Quick Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="result-card">
                <h3>Camera Angle</h3>
                <p>{angle_name.replace('_', ' ').title()}</p>
                <small>{angle_value}° - {conf_angle:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>Part Type</h3>
                <p>{pred_part}</p>
                <small>{conf_part:.1f}% confidence</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>Damage Type</h3>
                <p>{damage_type}</p>
                <small>YOLO Detection</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>Vehicle Part</h3>
                <p>{carpart_name}</p>
                <small>YOLO Detection</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            airbag_color = "linear-gradient(135deg, #fa709a 0%, #fee140 100%)" if "Deployed" in airbag_status or "deployed" in airbag_status.lower() else "linear-gradient(135deg, #30cfd0 0%, #330867 100%)"
            st.markdown(f"""
            <div class="result-card" style="background: {airbag_color};">
                <h3>Airbag Status</h3>
                <p>{airbag_status}</p>
                <small>YOLO Detection</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed Results
        st.markdown("---")
        st.markdown('<div class="sub-header">🔍 Detailed Detection Results</div>', unsafe_allow_html=True)
        
        # YOLO Detections
        tab1, tab2, tab3 = st.tabs(["🎈 Airbag Detection", "💥 Damage Detection", "🚗 Vehicle Part Detection"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(airbag_annotated, caption="Airbag Detection Result", use_container_width=True)
            with col2:
                st.markdown(f"### Detection Status")
                st.markdown(f"**Status:** {airbag_status}")
                if airbag_detections:
                    st.markdown("### Detections:")
                    for det in airbag_detections:
                        st.markdown(f"""
                        <div class="info-badge">
                            {det['class_name']} - {det['confidence']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No airbag deployment detected")
        
        with tab2:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(damage_annotated, caption="Damage Detection Result", use_container_width=True)
            with col2:
                st.markdown(f"### Damage Analysis")
                st.markdown(f"**Type:** {damage_type}")
                if damage_detections:
                    st.markdown("### Detections:")
                    for det in damage_detections:
                        st.markdown(f"""
                        <div class="info-badge">
                            {det['class_name']} - {det['confidence']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No damage detected")
        
        with tab3:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(carpart_annotated, caption="Vehicle Part Detection Result", use_container_width=True)
            with col2:
                st.markdown(f"### Part Identification")
                st.markdown(f"**Part:** {carpart_name}")
                if carpart_detections:
                    st.markdown("### Detections:")
                    for det in carpart_detections:
                        st.markdown(f"""
                        <div class="info-badge">
                            {det['class_name']} - {det['confidence']:.1%}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No specific part detected")
        
        # Classification Results
        st.markdown("---")
        st.markdown('<div class="sub-header">🎯 Classification Results</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📐 Angle Classification")
            st.image(data_angle, caption=f"Angle: {angle_name.replace('_', ' ').title()} - {angle_value}°", use_container_width=True)
            st.metric("Predicted Angle", f"{angle_name.replace('_', ' ').title()} ({angle_value}°)", f"{conf_angle:.1f}% confidence")
        
        with col2:
            st.markdown("### 🔧 Part Type Classification")
            st.image(data_part, caption=f"Part: {pred_part}", use_container_width=True)
            st.metric("Predicted Part", pred_part, f"{conf_part:.1f}% confidence")
        
        # Download Report Button
        st.markdown("---")
        st.download_button(
            label="📄 Download Analysis Report (JSON)",
            data=str(results),
            file_name="car_damage_analysis_report.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()