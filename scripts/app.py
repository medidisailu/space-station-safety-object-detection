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
    page_title="SpaceSafety AI | Professional Dashboard",
    page_icon="�️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Custom Styling (Glassmorphism Professional Website)
# -----------------------------
st.markdown("""
    <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
        
        /* Main Container Background */
        .stApp {
            background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
            font-family: 'Outfit', sans-serif;
        }

        /* Remove Extra Streamlit Whitespace */
        .block-container {
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            max-width: 95% !important; /* Use more screen width */
        }
        
        /* Glassmorphism Classes */
        .glass-card {
            background: rgba(255, 255, 255, 0.65);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            padding: 1.5rem; /* Reduced padding */
            margin-bottom: 1rem; /* Reduced margin */
            transition: all 0.3s ease;
        }
        
        /* Navbar Style */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem 2rem;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 0 0 20px 20px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        /* ... existing styles ... */
        .nav-logo { font-size: 1.5rem; font-weight: 700; color: #2c3e50; gap: 10px; display: flex; align-items: center; }
        .nav-links a { text-decoration: none; color: #5d6d7e; margin-left: 20px; font-weight: 500; transition: color 0.3s; }
        .nav-links a:hover { color: #3498db; }
        h1, h2, h3 { color: #2c3e50; margin-top: 0 !important; } /* Fix header margins */
        p { color: #5d6d7e; }
        
        div.stButton > button {
            background: linear-gradient(90deg, #66a6ff 0%, #89f7fe 100%);
            color: white; border: none; padding: 0.6rem 1.2rem; border-radius: 12px; font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 166, 255, 0.4); width: 100%;
        }
        
        .metric-container { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 1rem; }
        .metric-card {
            background: white; padding: 10px; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); flex: 1; min-width: 100px;
            text-align: center; border: 1px solid rgba(0,0,0,0.05);
        }
        .metric-value { font-size: 1.2rem; font-weight: 700; color: #3498db; }
        .metric-label { font-size: 0.7rem; color: #95a5a6; text-transform: uppercase; }
        
        /* Force Images to 16:9 Container feel if needed visually */
        img { border-radius: 12px; }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Logic
# -----------------------------
@st.cache_resource
def load_model():
    path = "runs/detect/train6/weights/best.pt"
    if os.path.exists(path):
        return YOLO(path)
    return YOLO("yolov8n.pt")

model = load_model()

def resize_16_9(image):
    """
    Resizes an image to a 16:9 aspect ratio (1280x720) with padding to maintain clear aspect.
    This ensures uniformity in the UI.
    """
    target_ratio = 16/9
    target_width = 800
    target_height = int(target_width / target_ratio) # 450
    
    # Use padding (expand) to ensure we don't crop out important details
    # But fill the graphical space nicely.
    return ImageOps.pad(image, (target_width, target_height), color=(255, 255, 255), centering=(0.5, 0.5))


# -----------------------------
# UI Layout
# -----------------------------

# Navbar
st.markdown("""
    <div class="navbar">
        <div class="nav-logo">
            <span>�️</span> SpaceSafety AI
        </div>
        <div class="nav-links">
            <a href="#">Dashboard</a>
            <a href="#">Analysis</a>
            <a href="#">Reports</a>
            <a href="#">System Status</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Main Content Grid
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Control Center")
    st.write("Upload mission imagery for automated defect analysis.")
    
    tab1, tab2 = st.tabs(["📂 Upload File", "📸 Live Camera"])
    
    image_source = None
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            image_source = Image.open(uploaded_file)
            
    with tab2:
        cam_img = st.camera_input("Capture Feed")
        if cam_img:
            image_source = Image.open(cam_img)

    st.markdown("---")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    if image_source:
        st.success("Image Loaded Successfully")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Stats (Placeholder)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📡 System Status")
    st.markdown("**Model:** YOLOv8 Custom")
    st.markdown("**Latency:** 45ms (Avg)")
    st.markdown("**Active Nodes:** 4/4")
    st.markdown('</div>', unsafe_allow_html=True)


with col_right:
    st.markdown('<div class="glass-card" style="min-height: 500px;">', unsafe_allow_html=True)
    st.markdown("### 👁️ Visual Analysis Feed")
    
    if image_source:
        # 1. Resize Input for Processing/Display (Standard 16:9)
        display_img = resize_16_9(image_source)
        
        # Inference
        # We run inference on the original source for accuracy, but display the resized/padded version
        # OR we can run on the processed 16:9 version if we want the output box to match exactly.
        # Let's run on the standardized 16:9 version to ensure visual alignment.
        results = model.predict(source=display_img, save=False, conf=conf_threshold)
        
        # Display Result
        res_plotted = results[0].plot()
        res_img = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        
        st.image(res_img, use_container_width=True, caption="Processed Output (16:9 Stream)")
        
        # Results Metrics
        st.markdown("#### 📊 Detection Metrics")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            # ... existing metric code ...
            detected_counts = {}
            for box in boxes:
                name = model.names[int(box.cls)]
                detected_counts[name] = detected_counts.get(name, 0) + 1
            
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            for name, count in detected_counts.items():
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{count}</div>
                    <div class="metric-label">{name.upper()} Detected</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No anomalies detected in the current frame.")
            
    else:
        # Empty State
        st.markdown("""
        <div style="text-align: center; padding: 100px 0; opacity: 0.5;">
            <div style="font-size: 80px; margin-bottom: 20px;">🖼️</div>
            <h3>Waiting for Input</h3>
            <p>Select an image from the Control Center to begin analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)