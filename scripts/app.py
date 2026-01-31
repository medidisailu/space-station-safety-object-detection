import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import time

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="SpaceGuard AI | Orbital Security",
    page_icon="🛡️",
    layout="wide",  # Changed to wide for website feel
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom Styling (Website Theme)
# -----------------------------
st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@400;700&family=Orbitron:wght@400;700&display=swap');

        /* Hide Streamlit Default Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="stToolbar"] {visibility: hidden;}
        
        /* General Body */
        .stApp {
            background-color: #0b0c10;
            background-image: 
                radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
                radial-gradient(at 50% 0%, hsla(225,39%,30%,1) 0, transparent 50%), 
                radial-gradient(at 100% 0%, hsla(339,49%,30%,1) 0, transparent 50%);
            color: #c5c6c7;
        }

        /* Navbar */
        .navbar {
            padding: 1rem 2rem;
            background: rgba(11, 12, 16, 0.95);
            border-bottom: 1px solid rgba(102, 252, 241, 0.2);
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 999;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .nav-logo {
            font-family: 'Orbitron', sans-serif;
            font-size: 1.5rem;
            font-weight: bold;
            color: #66fcf1;
            text-decoration: none;
        }
        .nav-links {
            display: flex;
            gap: 20px;
        }
        .nav-link {
            font-family: 'Exo 2', sans-serif;
            color: #c5c6c7;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        .nav-link:hover {
            color: #45a29e;
        }

        /* Hero Section */
        .hero {
            text-align: center;
            padding: 8rem 2rem 4rem;
            background: linear-gradient(180deg, rgba(11,12,16,0) 0%, rgba(31,40,51,0.5) 100%);
        }
        .hero h1 {
            font-family: 'Orbitron', sans-serif;
            font-size: 3.5rem;
            color: #ffffff;
            margin-bottom: 1rem;
            text-shadow: 0 0 20px rgba(102, 252, 241, 0.5);
        }
        .hero p {
            font-family: 'Exo 2', sans-serif;
            font-size: 1.2rem;
            max-width: 800px;
            margin: 0 auto 2rem;
            color: #c5c6c7;
        }

        /* App Container */
        .app-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
            background: rgba(31, 40, 51, 0.6);
            border: 1px solid rgba(102, 252, 241, 0.1);
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
        }

        /* Custom Button */
        div.stButton > button {
            background-color: #45a29e;
            color: #0b0c10;
            font-family: 'Orbitron', sans-serif;
            font-weight: bold;
            border-radius: 5px;
            border: none;
            padding: 12px 24px;
            transition: all 0.3s;
        }
        div.stButton > button:hover {
            background-color: #66fcf1;
            box-shadow: 0 0 15px rgba(102, 252, 241, 0.6);
            color: #000;
        }

        /* Footer */
        .footer {
            margin-top: 5rem;
            padding: 2rem;
            text-align: center;
            border-top: 1px solid rgba(255,255,255,0.1);
            color: #888;
            font-family: 'Exo 2', sans-serif;
            font-size: 0.9rem;
        }

        /* Result Cards */
        .result-card {
            background: rgba(11, 12, 16, 0.8);
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #66fcf1;
            margin-bottom: 10px;
        }
        .result-title {
            color: #66fcf1;
            font-family: 'Orbitron', sans-serif;
            font-size: 1.1rem;
        }
    </style>
    
    <!-- Navbar HTML -->
    <div class="navbar">
        <a href="#" class="nav-logo">🛡️ SPACEGUARD AI</a>
        <div class="nav-links">
            <a href="#" class="nav-link">Home</a>
            <a href="#" class="nav-link">Technology</a>
            <a href="#" class="nav-link">Mission Control</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# -----------------------------
# Logic & Helpers
# -----------------------------
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    # Path relative to where you run streamlit
    MODEL_PATH = "runs/detect/train6/weights/best.pt" 
    return YOLO(MODEL_PATH)

def resize_for_display(image_source, target_size=(800, 450), bg_color="#0b0c10"):
    """
    Resizes image to target size while maintaining aspect ratio (padding with bg_color).
    Accepts file path, PIL Image, or Numpy Array (BGR).
    """
    if isinstance(image_source, str):
        img = Image.open(image_source)
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
    else:
        img = image_source
        
    return ImageOps.pad(img, target_size, color=bg_color, centering=(0.5, 0.5))

model = load_model()

# -----------------------------
# Main Content
# -----------------------------

# Hero Section
st.markdown("""
<div class="hero">
    <h1>Autonomous Safety Protocol</h1>
    <p>Advanced computer vision system for real-time anomaly detection in zero-gravity environments. 
    Securing the future of space exploration, one frame at a time.</p>
</div>
""", unsafe_allow_html=True)

# Main App Container
with st.container():
    st.markdown('<div class="app-container">', unsafe_allow_html=True)
    
    # Layout: Two Columns (Left: Controls/Upload, Right: Results)
    col1, col_spacer, col2 = st.columns([1, 0.1, 1.5])
    
    with col1:
        st.markdown("### 📡 Uplink Data")
        uploaded_file = st.file_uploader("Upload Imagery (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        # Hidden slider for calibration (still functional but less obtrusive)
        conf_threshold = st.slider("Signal Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05, label_visibility="collapsed")
        
        if uploaded_file:
            # Save file locally
            os.makedirs("uploads", exist_ok=True)
            image_path = os.path.join("uploads", uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.markdown("---")
            detect_btn = st.button("RUN DIAGNOSTICS")

    with col2:
        if uploaded_file:
            # Show original if no detection yet
            if 'detect_btn' not in locals() or not detect_btn:
                img_display = resize_for_display(image_path)
                st.image(img_display, caption="Original Feed", use_container_width=True)

            if 'detect_btn' in locals() and detect_btn:
                with st.spinner("Processing neural network layers..."):
                    # Simulate processing time for effect
                    time.sleep(0.5) 
                    
                    results = model.predict(source=image_path, save=False, conf=conf_threshold)
                    
                    # Plot Results
                    res_plotted = results[0].plot()
                    res_display = resize_for_display(res_plotted)
                    st.image(res_display, caption="Processed Feed | Anomalies Highlighted", use_container_width=True)
                    
                    # Extract Data
                    names = results[0].names
                    boxes = results[0].boxes
                    detected_data = {}
                    
                    for box in boxes:
                        cls_id = int(box.cls)
                        class_name = names[cls_id]
                        conf = float(box.conf)
                        if class_name not in detected_data:
                            detected_data[class_name] = []
                        detected_data[class_name].append(conf)

                    # Show Metadata
                    st.markdown("### 📊 Telemetry Report")
                    if detected_data:
                        for name, confs in detected_data.items():
                            avg_conf = sum(confs)/len(confs)
                            st.markdown(f"""
                            <div class="result-card">
                                <div class="result-title">{name.upper()}</div>
                                <div>Count: <strong>{len(confs)}</strong> &nbsp;|&nbsp; Avg. Confidence: <strong>{avg_conf:.2f}</strong></div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.success("✅ Nominal. No anomalies detected in this sector.")
        else:
            # Placeholder State
            st.markdown("""
            <div style="text-align: center; padding: 100px 0; border: 2px dashed #45a29e; border-radius: 10px; color: #45a29e;">
                <h4>Awaiting Visual Input</h4>
                <p>Upload file via the uplink panel to begin.</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    <p>© 2026 SpaceGuard AI Systems. Restricted Access. Authorized Personnel Only.</p>
</div>
""", unsafe_allow_html=True)