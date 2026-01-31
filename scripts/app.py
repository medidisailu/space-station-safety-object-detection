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
        
        /* Main Container Background - Light Sky Blue Gradient */
        .stApp {
            background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);
            font-family: 'Outfit', sans-serif;
        }

        /* Glassmorphism Classes */
        .glass-card {
            background: rgba(255, 255, 255, 0.65);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            padding: 2rem;
            margin-bottom: 2rem;
            transition: all 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
        }

        /* Navbar Style */
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 0 0 20px 20px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
        }
        
        .nav-logo {
            font-size: 1.5rem;
            font-weight: 700;
            color: #2c3e50;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .nav-links a {
            text-decoration: none;
            color: #5d6d7e;
            margin-left: 20px;
            font-weight: 500;
            transition: color 0.3s;
        }
        .nav-links a:hover {
            color: #3498db;
        }

        /* Typography */
        h1, h2, h3 {
            color: #2c3e50;
        }
        p {
            color: #5d6d7e;
        }

        /* Custom Button Styling */
        div.stButton > button {
            background: linear-gradient(90deg, #66a6ff 0%, #89f7fe 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 12px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 166, 255, 0.4);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        div.stButton > button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(102, 166, 255, 0.6);
            color: white;
        }

        /* Results & Stats */
        .metric-container {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            flex: 1;
            min-width: 140px;
            text-align: center;
            border: 1px solid rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #3498db;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #95a5a6;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
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
        # Inference
        results = model.predict(source=image_source, save=False, conf=conf_threshold)
        
        # Display Result
        res_plotted = results[0].plot()
        res_img = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
        
        st.image(res_img, use_container_width=True, caption="Processed Output")
        
        # Results Metrics
        st.markdown("#### 📊 Detection Metrics")
        
        boxes = results[0].boxes
        if len(boxes) > 0:
            stats_cols = st.columns(3)
            # Create metrics for first 3 detections max for display
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