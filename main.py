import streamlit as st
import cv2
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
# from keras.models import load_model   # Uncomment when models are available
# import supervision as sv              # Uncomment when models are available
# from ultralytics import YOLO          # Uncomment when models are available
import pandas as pd
# import xgboost as xgb                 # Uncomment when models are available
import pickle
from auth import init_db, register_user, login_user

# ─────────────────────────────────────────────
# Page configuration  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Car Damage Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Initialise DB
# ─────────────────────────────────────────────
init_db()

# ─────────────────────────────────────────────
# Global CSS  (auth pages + main app)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #0a0a0f;
    color: #000000;
}

/* ── Hide Streamlit chrome on auth pages ── */
.auth-page [data-testid="stSidebar"] { display: none; }

/* ── Auth wrapper ── */
.auth-outer {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        radial-gradient(ellipse 80% 60% at 20% 10%, rgba(59,130,246,.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 80% 90%, rgba(139,92,246,.14) 0%, transparent 60%),
        #0a0a0f;
    padding: 2rem 1rem;
}

/* ── Card ── */
.auth-card {
    width: 100%;
    max-width: 460px;
    background: rgba(255,255,255,.04);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 24px;
    padding: 2.5rem 2.8rem;
    box-shadow:
        0 0 0 1px rgba(59,130,246,.06),
        0 32px 64px rgba(0,0,0,.5);
    margin: 0 auto;
}

/* ── Brand ── */
.auth-brand {
    text-align: center;
    margin-bottom: 2rem;
}
.auth-brand .logo {
    font-size: 2.8rem;
    display: block;
    margin-bottom: .4rem;
    filter: drop-shadow(0 0 18px rgba(59,130,246,.6));
}
.auth-brand h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -.5px;
    color: #fff;
    margin: 0;
    line-height: 1.2;
}
.auth-brand p {
    color: rgba(255,255,255,.45);
    font-size: .85rem;
    margin: .4rem 0 0;
    letter-spacing: .3px;
}

/* ── Tab bar ── */
.tab-bar {
    display: flex;
    gap: .5rem;
    background: rgba(255,255,255,.05);
    border-radius: 12px;
    padding: .35rem;
    margin-bottom: 1.8rem;
}
.tab-btn {
    flex: 1;
    padding: .55rem 0;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: .9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all .2s ease;
    background: transparent;
    color: rgba(255,255,255,.45);
}
.tab-btn.active {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: #fff;
    box-shadow: 0 4px 14px rgba(59,130,246,.35);
}

/* ── Form labels ── */
.field-label {
    display: block;
    font-size: .78rem;
    font-weight: 500;
    letter-spacing: .6px;
    text-transform: uppercase;
    color: rgba(255,255,255,.5);
    margin-bottom: .4rem;
}

/* ── Streamlit input overrides ── */
[data-testid="stTextInput"] > div > div > input,
[data-testid="stTextInput"] > div > div > input:focus {
    background: rgba(255,255,255,.06) !important;
    border: 1px solid rgba(255,255,255,.12) !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: .7rem 1rem !important;
    font-size: .95rem !important;
    box-shadow: none !important;
    transition: border-color .2s ease !important;
}
[data-testid="stTextInput"] > div > div > input:focus {
    border-color: rgba(59,130,246,.5) !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.12) !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"],
.auth-submit .stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: .3px !important;
    padding: .75rem 0 !important;
    transition: all .2s ease !important;
    box-shadow: 0 4px 20px rgba(59,130,246,.3) !important;
}
.stButton > button[kind="primary"]:hover,
.auth-submit .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(59,130,246,.45) !important;
}

/* ── Alert overrides ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: .9rem !important;
    border: none !important;
}

/* ── Divider ── */
.auth-divider {
    display: flex;
    align-items: center;
    gap: .8rem;
    margin: 1.2rem 0;
    color: rgba(255,255,255,.2);
    font-size: .8rem;
}
.auth-divider::before, .auth-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,.08);
}

/* ── User info pill ── */
.user-pill {
    display: flex;
    align-items: center;
    gap: .6rem;
    background: rgba(59,130,246,.12);
    border: 1px solid rgba(59,130,246,.2);
    border-radius: 50px;
    padding: .4rem .9rem .4rem .5rem;
    width: fit-content;
}
.user-pill .avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: .85rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}
.user-pill span { font-size: .88rem; color: rgba(255,255,255,.8); }

/* ── Password strength meter ── */
.strength-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: .4rem;
    transition: all .3s ease;
}
.strength-label {
    font-size: .75rem;
    margin-top: .3rem;
    font-weight: 500;
}

/* ─────── MAIN APP STYLES ─────── */
.main-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(135deg, #3b82f6 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    letter-spacing: -1px;
}
.sub-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8e8f0;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid rgba(59,130,246,.4);
    padding-bottom: .5rem;
}
.result-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    color: white;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,.2);
    margin: .5rem 0;
}
.result-card h3 { margin: 0; font-size: 1rem; opacity: .9; }
.result-card p  { margin: .5rem 0 0; font-size: 1.6rem; font-weight: bold; }
.severity-card {
    padding: 2rem;
    border-radius: 1rem;
    color: white;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,.3);
    margin: 1rem 0;
    border: 2px solid rgba(255,255,255,.2);
}
.severity-card h2 { margin: 0; font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800; }
.severity-card p  { margin: .5rem 0; font-size: 1.1rem; opacity: .9; }
.upload-section {
    background: rgba(255,255,255,.03);
    padding: 1.5rem;
    border-radius: .8rem;
    border: 1px dashed rgba(255,255,255,.15);
    margin: 1rem 0;
}
.success-box {
    background: rgba(34,197,94,.1);
    border: 1px solid rgba(34,197,94,.3);
    color: #86efac;
    padding: 1rem;
    border-radius: .6rem;
    margin: 1rem 0;
}
.info-badge {
    display: inline-block;
    background: rgba(59,130,246,.2);
    border: 1px solid rgba(59,130,246,.3);
    color: #93c5fd;
    padding: .3rem .8rem;
    border-radius: 1rem;
    font-size: .85rem;
    margin: .2rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session-state defaults
# ─────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = {}
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"   # "login" | "register"


# ─────────────────────────────────────────────
# Helper: password-strength indicator
# ─────────────────────────────────────────────
def password_strength_html(password: str) -> str:
    score = 0
    if len(password) >= 8:   score += 1
    if any(c.isupper() for c in password): score += 1
    if any(c.isdigit() for c in password): score += 1
    if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password): score += 1

    labels  = ["", "Weak", "Fair", "Good", "Strong"]
    colours = ["", "#ef4444", "#f59e0b", "#22c55e", "#3b82f6"]
    widths  = ["0%", "25%", "50%", "75%", "100%"]

    if not password:
        return ""
    return f"""
    <div class="strength-bar" style="width:{widths[score]};background:{colours[score]};"></div>
    <div class="strength-label" style="color:{colours[score]};">{labels[score]}</div>
    """


# ─────────────────────────────────────────────
# AUTH PAGE
# ─────────────────────────────────────────────
def show_auth_page():
    # Center column trick
    _, center, _ = st.columns([1, 2, 1])

    with center:
        # ── Brand ──
        st.markdown("""
        <div class="auth-brand">
            <span class="logo">🚗</span>
            <h1>CarDamage AI</h1>
            <p>Professional vehicle damage assessment platform</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Tab switcher ──
        col_l, col_r = st.columns(2)
        with col_l:
            if st.button("Sign In", key="tab_login",
                         type="primary" if st.session_state.auth_tab == "login" else "secondary",
                         use_container_width=True):
                st.session_state.auth_tab = "login"
                st.rerun()
        with col_r:
            if st.button("Create Account", key="tab_register",
                         type="primary" if st.session_state.auth_tab == "register" else "secondary",
                         use_container_width=True):
                st.session_state.auth_tab = "register"
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ── LOGIN FORM ──
        if st.session_state.auth_tab == "login":
            with st.form("login_form", clear_on_submit=False):
                st.markdown('<span class="field-label">Username or Email</span>', unsafe_allow_html=True)
                identifier = st.text_input("", placeholder="johndoe  or  john@example.com",
                                           key="li_id", label_visibility="collapsed")

                st.markdown('<span class="field-label">Password</span>', unsafe_allow_html=True)
                password = st.text_input("", placeholder="••••••••",
                                         type="password", key="li_pw",
                                         label_visibility="collapsed")

                submitted = st.form_submit_button("Sign In →", use_container_width=True, type="primary")

            if submitted:
                if not identifier or not password:
                    st.error("Please fill in all fields.")
                else:
                    success, message, user_data = login_user(identifier, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user = user_data
                        st.success(f"Welcome back, {user_data['full_name'].split()[0]}! 👋")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

        # ── REGISTER FORM ──
        else:
            with st.form("register_form", clear_on_submit=False):
                st.markdown('<span class="field-label">Full Name</span>', unsafe_allow_html=True)
                full_name = st.text_input("", placeholder="John Doe",
                                          key="rg_name", label_visibility="collapsed")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown('<span class="field-label">Username</span>', unsafe_allow_html=True)
                    username = st.text_input("", placeholder="johndoe",
                                            key="rg_user", label_visibility="collapsed")
                with col_b:
                    st.markdown('<span class="field-label">Email</span>', unsafe_allow_html=True)
                    email = st.text_input("", placeholder="john@example.com",
                                         key="rg_email", label_visibility="collapsed")

                st.markdown('<span class="field-label">Password</span>', unsafe_allow_html=True)
                password = st.text_input("", placeholder="Min 8 chars, 1 uppercase, 1 number",
                                         type="password", key="rg_pw",
                                         label_visibility="collapsed")

                st.markdown('<span class="field-label">Confirm Password</span>', unsafe_allow_html=True)
                confirm = st.text_input("", placeholder="Re-enter password",
                                        type="password", key="rg_conf",
                                        label_visibility="collapsed")

                submitted = st.form_submit_button("Create Account →", use_container_width=True, type="primary")

            # Live password strength (outside form so it renders dynamically)
            if st.session_state.get("rg_pw"):
                st.markdown(password_strength_html(st.session_state["rg_pw"]), unsafe_allow_html=True)

            if submitted:
                if password != confirm:
                    st.error("❌ Passwords do not match.")
                else:
                    success, message = register_user(full_name, email, username, password)
                    if success:
                        st.success(f"✅ {message} Please sign in.")
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

        # ── Footer note ──
        st.markdown("""
        <div style="text-align:center;margin-top:1.8rem;color:rgba(255,255,255,.25);font-size:.78rem;">
            🔒 Your data is stored securely on this server
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL LOADING  (unchanged from your original)
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all ML models"""
    try:
        from keras.models import load_model as keras_load
        import supervision as sv
        from ultralytics import YOLO
        import xgboost as xgb

        model_angle = keras_load('model.keras')
        model_part  = keras_load('CarPartINExmodel.h5')

        yolo_models = {
            "airbag":  YOLO("model/airbag.pt"),
            "damage":  YOLO("model/damage.pt"),
            "carpart": YOLO("model/carpart.pt")
        }

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )
        xgb_model.load_model("xgb_model.json")

        with open('feature_encoders.pkl', 'rb') as f:
            feature_encoders = pickle.load(f)
        with open('severity_encoder.pkl', 'rb') as f:
            severity_encoder = pickle.load(f)

        return model_angle, model_part, yolo_models, xgb_model, feature_encoders, severity_encoder
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None, None


# ─────────────────────────────────────────────
# PREPROCESSING / DETECTION HELPERS  (unchanged)
# ─────────────────────────────────────────────
def preprocess(image, size):
    img = image.convert('RGB')
    img = np.asarray(img)
    img = cv2.resize(img, (size, size))
    return img / 255.0


def yolo_single_detection(image, model, model_name):
    import supervision as sv
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img_cv)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = img_cv.copy()
    detection_list, detected_classes = [], []

    if len(detections) > 0:
        annotated_image = sv.BoxAnnotator(thickness=2).annotate(scene=annotated_image, detections=detections)
        annotated_image = sv.LabelAnnotator(text_thickness=2, text_scale=0.8).annotate(scene=annotated_image, detections=detections)

        for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
            label = model.names[int(class_id)]
            detected_classes.append(label)
            detection_list.append({"class_name": label, "confidence": float(confidence), "bbox": [float(x) for x in box]})

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    primary = detected_classes[0] if detected_classes else (
        "Not Deployed" if model_name == "airbag" else "No Damage" if model_name == "damage" else "Unknown"
    )
    return annotated_image, detection_list, primary


def classify_angle_and_part(img_angle, img_part, model_angle, model_part):
    data_angle = preprocess(img_angle, 331)
    data_part  = preprocess(img_part, 224)
    xtest_angle = np.expand_dims(data_angle, axis=0)
    xtest_part  = np.expand_dims(data_part,  axis=0)

    Y_pred_angle = model_angle.predict([xtest_angle, xtest_angle], verbose=0)
    Y_pred_part  = model_part.predict(xtest_part, verbose=0)

    pred_class_angle = np.argmax(Y_pred_angle, axis=1)[0]
    pred_class_part  = np.argmax(Y_pred_part,  axis=1)[0]
    confidence_angle = np.max(Y_pred_angle) * 100
    confidence_part  = np.max(Y_pred_part)  * 100

    part_labels = ['Air intake','Console','Dashboard','Fog light','Gear stick','Headlight','Steering wheel','Tail light']
    pred_part_name = part_labels[pred_class_part]

    angle_label_map = {'0':0,'130':1,'180':2,'230':3,'270':4,'320':5,'40':6,'90':7}
    angle_mapping   = {"right":0,"front_left":130,"left":180,"rear_left":230,"rear":270,"rear_right":320,"front_right":40,"front":90}
    index_to_angle  = {v: k for k, v in angle_label_map.items()}
    angle_value     = int(index_to_angle[pred_class_angle])
    angle_name      = {v: k for k, v in angle_mapping.items()}[angle_value]

    return pred_part_name, angle_name, angle_value, confidence_part, confidence_angle, data_angle, data_part


def predict_severity(angle_name, pred_part, damage_type, carpart_name, airbag_status,
                     xgb_model, feature_encoders, severity_encoder):
    import pandas as pd
    internal_parts = ['Console','Dashboard','Gear stick','Steering wheel','Air intake']
    view_type      = 'internal' if pred_part in internal_parts else 'external'
    airbag_mapped  = "deployed" if "deployed" in airbag_status.lower() else "not_applicable"

    data = pd.DataFrame({'view_type':[view_type],'angle':[angle_name],
                         'part':[carpart_name],'damage':[damage_type],'airbag':[airbag_mapped]})

    for col in ['view_type','angle','part','damage','airbag']:
        if col in feature_encoders:
            try:    data[col] = feature_encoders[col].transform(data[col])
            except: data[col] = 0

    try:
        y_pred_encoded = xgb_model.predict(data)
        severity       = severity_encoder.inverse_transform(y_pred_encoded)[0]
        confidence     = float(np.max(xgb_model.predict_proba(data)) * 100)
        return severity, confidence, view_type
    except Exception as e:
        return "Unable to predict", 0.0, view_type


# ─────────────────────────────────────────────
# MAIN APP  (protected, shown after login)
# ─────────────────────────────────────────────
def show_main_app():
    user = st.session_state.user

    # ── Sidebar ──
    with st.sidebar:
        # User card
        initials = "".join(w[0].upper() for w in user.get("full_name","U U").split()[:2])
        st.markdown(f"""
        <div style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);
                    border-radius:14px;padding:1rem 1.2rem;margin-bottom:1rem;">
            <div style="display:flex;align-items:center;gap:.8rem;">
                <div style="width:42px;height:42px;background:linear-gradient(135deg,#3b82f6,#6366f1);
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-family:'Syne',sans-serif;font-weight:800;font-size:1rem;color:#fff;flex-shrink:0;">
                    {initials}
                </div>
                <div>
                    <div style="font-weight:600;font-size:.95rem;color:#e8e8f0;">{user.get('full_name','User')}</div>
                    <div style="font-size:.78rem;color:rgba(255,255,255,.4);">@{user.get('username','')}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        

        st.divider()
        
        st.header("📋 Upload Instructions")
        st.markdown("""
        ### Required Images (5 Total):
        1️⃣ **Airbag Image** – airbag deployment detection  
        2️⃣ **Damage Image** – damage type detection  
        3️⃣ **Vehicle Part Image** – car part detection  
        4️⃣ **Angle Image** – camera angle classification  
        5️⃣ **Part Classification Image** – internal/external part  

        ---
        ### Analysis Output:
        - ✅ Angle Classification
        - ✅ Part Type
        - ✅ Damage Type
        - ✅ Vehicle Part
        - ✅ Airbag Status
        - ✅ **Damage Severity**
        """)

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = {}
            st.rerun()

    # ── Header ──
    st.markdown('<h1 class="main-header">🚗 Complete Car Damage Analysis System</h1>', unsafe_allow_html=True)

    # ── Load models ──
    with st.spinner("🔄 Loading AI models..."):
        model_angle, model_part, yolo_models, xgb_model, feature_encoders, severity_encoder = load_models()

    if model_angle is None:
        st.error("❌ Failed to load models. Please check model files.")
        return

    st.success("✅ All AI models loaded successfully!")

    # ── Upload section ──
    st.markdown('<div class="sub-header">📤 Upload 5 Images for Complete Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 YOLO Detection Images")
        airbag_file  = st.file_uploader("1️⃣ Airbag Image",       type=['jpg','jpeg','png'], key="airbag",  help="Upload image to detect airbag deployment")
        damage_file  = st.file_uploader("2️⃣ Damage Image",        type=['jpg','jpeg','png'], key="damage",  help="Upload image to detect damage type")
        carpart_file = st.file_uploader("3️⃣ Vehicle Part Image",  type=['jpg','jpeg','png'], key="carpart", help="Upload image to detect car part")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 Classification Images")
        angle_file = st.file_uploader("4️⃣ Angle Image",               type=['jpg','jpeg','png'], key="angle", help="Upload image for angle classification")
        part_file  = st.file_uploader("5️⃣ Part Classification Image",  type=['jpg','jpeg','png'], key="part",  help="Upload image for part type classification")
        st.markdown('</div>', unsafe_allow_html=True)

    all_uploaded   = all([airbag_file, damage_file, carpart_file, angle_file, part_file])
    uploaded_count = sum(map(bool, [airbag_file, damage_file, carpart_file, angle_file, part_file]))

    if all_uploaded:
        st.markdown('<div class="success-box">✅ All 5 images uploaded! Ready for analysis.</div>', unsafe_allow_html=True)
    else:
        st.info(f"📊 {uploaded_count}/5 images uploaded. Please upload all 5 images to run the analysis.")

    st.markdown("---")

    if st.button("🚀 RUN COMPLETE ANALYSIS", type="primary", use_container_width=True, disabled=not all_uploaded):

        progress_bar = st.progress(0)
        status_text  = st.empty()
        results      = {}

        # 1. Airbag
        status_text.text("🔍 Analyzing airbag status...")
        progress_bar.progress(16)
        img_airbag = Image.open(airbag_file)
        airbag_annotated, airbag_detections, airbag_status = yolo_single_detection(img_airbag, yolo_models["airbag"], "airbag")
        results['airbag'] = {'annotated': airbag_annotated, 'detections': airbag_detections, 'status': airbag_status}

        # 2. Damage
        status_text.text("🔍 Detecting damage type...")
        progress_bar.progress(33)
        img_damage = Image.open(damage_file)
        damage_annotated, damage_detections, damage_type = yolo_single_detection(img_damage, yolo_models["damage"], "damage")
        results['damage'] = {'annotated': damage_annotated, 'detections': damage_detections, 'type': damage_type}

        # 3. Car part
        status_text.text("🔍 Identifying vehicle part...")
        progress_bar.progress(50)
        img_carpart = Image.open(carpart_file)
        carpart_annotated, carpart_detections, carpart_name = yolo_single_detection(img_carpart, yolo_models["carpart"], "carpart")
        results['carpart'] = {'annotated': carpart_annotated, 'detections': carpart_detections, 'name': carpart_name}

        # 4. Classification
        status_text.text("🎯 Classifying angle and part type...")
        progress_bar.progress(66)
        img_angle = Image.open(angle_file)
        img_part  = Image.open(part_file)
        pred_part, angle_name, angle_value, conf_part, conf_angle, data_angle, data_part = classify_angle_and_part(
            img_angle, img_part, model_angle, model_part
        )
        results['classification'] = {'part': pred_part, 'angle_name': angle_name, 'angle_value': angle_value,
                                     'conf_part': conf_part, 'conf_angle': conf_angle,
                                     'img_angle': data_angle, 'img_part': data_part}

        # 5. Severity
        status_text.text("🎯 Predicting damage severity...")
        progress_bar.progress(83)
        severity, severity_confidence, view_type = predict_severity(
            angle_name=angle_name, pred_part=pred_part, damage_type=damage_type,
            carpart_name=carpart_name, airbag_status=airbag_status,
            xgb_model=xgb_model, feature_encoders=feature_encoders, severity_encoder=severity_encoder
        )
        results['severity'] = {'level': severity, 'confidence': severity_confidence, 'view_type': view_type}

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        # ── Results ──
        st.markdown('<div class="sub-header">📊 Complete Analysis Results</div>', unsafe_allow_html=True)

        st.markdown("### 🚨 Damage Severity Assessment")
        severity_colors = {
            'minor':    "linear-gradient(135deg,#56ab2f 0%,#a8e063 100%)",
            'moderate': "linear-gradient(135deg,#f093fb 0%,#f5576c 100%)",
            'severe':   "linear-gradient(135deg,#fa709a 0%,#fee140 100%)",
        }
        sev_color = severity_colors.get(severity.lower(), "linear-gradient(135deg,#667eea 0%,#764ba2 100%)")
        st.markdown(f"""
        <div class="severity-card" style="background:{sev_color};">
            <h2>⚠️ {severity.upper()}</h2>
            <p>Confidence: {severity_confidence:.1f}%</p>
            <p>View Type: {view_type.title()}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🎯 Quick Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f'<div class="result-card"><h3>Camera Angle</h3><p>{angle_name.replace("_"," ").title()}</p><small>{angle_value}° – {conf_angle:.1f}%</small></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="result-card" style="background:linear-gradient(135deg,#f093fb 0%,#f5576c 100%);"><h3>Part Type</h3><p>{pred_part}</p><small>{conf_part:.1f}% confidence</small></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="result-card" style="background:linear-gradient(135deg,#4facfe 0%,#00f2fe 100%);"><h3>Damage Type</h3><p>{damage_type}</p><small>YOLO Detection</small></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="result-card" style="background:linear-gradient(135deg,#43e97b 0%,#38f9d7 100%);"><h3>Vehicle Part</h3><p>{carpart_name}</p><small>YOLO Detection</small></div>', unsafe_allow_html=True)
        with c5:
            ac = "linear-gradient(135deg,#fa709a 0%,#fee140 100%)" if "deployed" in airbag_status.lower() else "linear-gradient(135deg,#30cfd0 0%,#330867 100%)"
            st.markdown(f'<div class="result-card" style="background:{ac};"><h3>Airbag Status</h3><p>{airbag_status}</p><small>YOLO Detection</small></div>', unsafe_allow_html=True)

        # Detailed tabs
        st.markdown("---")
        st.markdown('<div class="sub-header">🔍 Detailed Detection Results</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["🎈 Airbag Detection", "💥 Damage Detection", "🚗 Vehicle Part Detection"])

        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1: st.image(airbag_annotated, caption="Airbag Detection Result", use_container_width=True)
            with col2:
                st.markdown(f"### Detection Status\n**Status:** {airbag_status}")
                if airbag_detections:
                    for det in airbag_detections:
                        st.markdown(f'<div class="info-badge">{det["class_name"]} – {det["confidence"]:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.info("No airbag deployment detected")

        with tab2:
            col1, col2 = st.columns([2, 1])
            with col1: st.image(damage_annotated, caption="Damage Detection Result", use_container_width=True)
            with col2:
                st.markdown(f"### Damage Analysis\n**Type:** {damage_type}")
                if damage_detections:
                    for det in damage_detections:
                        st.markdown(f'<div class="info-badge">{det["class_name"]} – {det["confidence"]:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.info("No damage detected")

        with tab3:
            col1, col2 = st.columns([2, 1])
            with col1: st.image(carpart_annotated, caption="Vehicle Part Detection Result", use_container_width=True)
            with col2:
                st.markdown(f"### Part Identification\n**Part:** {carpart_name}")
                if carpart_detections:
                    for det in carpart_detections:
                        st.markdown(f'<div class="info-badge">{det["class_name"]} – {det["confidence"]:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.info("No specific part detected")

        # Classification results
        st.markdown("---")
        st.markdown('<div class="sub-header">🎯 Classification Results</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📐 Angle Classification")
            st.image(data_angle, caption=f"Angle: {angle_name.replace('_',' ').title()} – {angle_value}°", use_container_width=True)
            st.metric("Predicted Angle", f"{angle_name.replace('_',' ').title()} ({angle_value}°)", f"{conf_angle:.1f}% confidence")
        with col2:
            st.markdown("### 🔧 Part Type Classification")
            st.image(data_part, caption=f"Part: {pred_part}", use_container_width=True)
            st.metric("Predicted Part", pred_part, f"{conf_part:.1f}% confidence")

        st.markdown("---")
        st.download_button(
            label="📄 Download Analysis Report (JSON)",
            data=str({k: {sk: sv for sk, sv in v.items() if not isinstance(sv, np.ndarray)} for k, v in results.items()}),
            file_name="car_damage_analysis_report.txt",
            mime="text/plain"
        )


def main():
    if st.session_state.authenticated:
        show_main_app()
    else:
        show_auth_page()


if __name__ == "__main__":
    main()