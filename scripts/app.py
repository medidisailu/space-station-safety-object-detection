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
    page_title="SpaceSafety AI",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom Styling (Mobile-App Like Design)
# -----------------------------
st.markdown("""
    <style>
        /* Import Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        
        /* Base Styles - Dark Mode Force */
        .stApp {
            background-color: #0d1117;
            font-family: 'Inter', sans-serif;
        }
        
        /* Remove default Streamlit padding/header */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 500px; /* Mobile width simulation */
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Header Section */
        .header-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }
        .app-title-box {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .app-icon {
            font-size: 2rem;
            background: #161b22;
            padding: 8px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .title-text {
            color: #ffffff;
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .subtitle-text {
            color: #8b949e;
            font-size: 0.8rem;
            font-weight: 400;
        }
        .status-badge {
            background-color: #1a2e1f;
            color: #3fb950;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid rgba(63, 185, 80, 0.2);
            margin-top: 8px;
        }
        .status-dot {
            width: 6px;
            height: 6px;
            background-color: #3fb950;
            border-radius: 50%;
        }

        /* Toggle Icon (Visual Only) */
        .theme-toggle {
            color: #f0f6fc;
            font-size: 1.2rem;
            cursor: pointer;
        }

        /* Main Display Card */
        .display-card {
            background-color: #161b22;
            border-radius: 24px;
            padding: 2rem;
            text-align: center;
            border: 1px solid #30363d;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            min-height: 350px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-bottom: 1.5rem;
            position: relative;
            overflow: hidden;
        }
        
        .placeholder-icon {
            font-size: 4rem;
            color: #30363d;
            margin-bottom: 1rem;
        }
        .placeholder-text {
            color: #8b949e;
            font-size: 0.9rem;
        }

        /* Action Buttons Container */
        .action-container {
            display: flex;
            gap: 1rem;
            justify-content: space-between;
        }
        
        /* Custom Button Styling via Streamlit Widgets */
        div.stButton > button {
            background-color: #161b22;
            color: #f0f6fc;
            border: 1px solid #30363d;
            border-radius: 20px;
            height: 100px;
            width: 100%;
            font-weight: 600;
            transition: all 0.2s;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        div.stButton > button:hover {
            background-color: #21262d;
            border-color: #8b949e;
            transform: translateY(-2px);
        }
        div.stButton > button:active {
            transform: translateY(0);
        }

        /* Specific Button Colors (Simulated) */
        /* Note: We can't target specific buttons easily without more complex CSS hacks, 
           so we keep them uniform but premium. */
        
        /* Results Styles */
        .result-stats {
            margin-top: 1rem;
            padding: 1rem;
            background: rgba(63, 185, 80, 0.1);
            border-radius: 12px;
            width: 100%;
        }
        .stat-item {
            display: flex;
            justify-content: space-between;
            color: #f0f6fc;
            font-size: 0.9rem;
            margin-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Logic & Helpers
# -----------------------------
@st.cache_resource
def load_model():
    # Use standard model if custom not found (removed hard dependency on specific path for stability)
    path = "runs/detect/train6/weights/best.pt"
    if os.path.exists(path):
        return YOLO(path)
    return YOLO("yolov8n.pt") # Fallback

model = load_model()

# Initialize State
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = None # 'gallery' or 'camera'

if 'detected_image' not in st.session_state:
    st.session_state.detected_image = None
    
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# Function to run inference
def run_inference(image_source):
    if isinstance(image_source, str):
        img = Image.open(image_source)
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB))
    else:
        img = image_source # PIL Image

    results = model.predict(source=img, save=False, conf=0.25)
    
    # Plot results
    res_plotted = results[0].plot()
    st.session_state.detected_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
    
    # Store data
    boxes = results[0].boxes
    data = []
    for box in boxes:
        cls_id = int(box.cls)
        name = model.names[cls_id]
        conf = float(box.conf)
        data.append((name, conf))
    st.session_state.detection_results = data

# -----------------------------
# App Layout
# -----------------------------

# 1. Header
st.markdown("""
<div class="header-container">
    <div class="app-title-box">
        <div class="app-icon">🛡️</div>
        <div>
            <div class="title-text">SpaceSafety AI</div>
            <div class="subtitle-text">Secure Object Detection</div>
            <div class="status-badge"><span class="status-dot"></span>SYSTEM ACTIVE</div>
        </div>
    </div>
    <div class="theme-toggle">☀️</div>
</div>
""", unsafe_allow_html=True)

# 2. Main Display Area
placeholder = st.empty()

# Determine content for main display
if st.session_state.detected_image:
    with placeholder.container():
        st.markdown('<div class="display-card" style="padding: 0;">', unsafe_allow_html=True)
        st.image(st.session_state.detected_image, use_container_width=True)
        
        # Stats Overlay
        if st.session_state.detection_results:
            st.markdown('<div class="result-stats">', unsafe_allow_html=True)
            for name, conf in st.session_state.detection_results:
                st.markdown(f"""
                <div class="stat-item">
                    <span>{name.upper()}</span>
                    <span style="color: #3fb950; font-weight: bold;">{int(conf*100)}%</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Close Button to reset
        if st.button("❌ Close Scan", use_container_width=True):
            st.session_state.detected_image = None
            st.session_state.detection_results = None
            st.rerun()

else:
    # Default State or Input Mode
    with placeholder.container():
        st.markdown('<div class="display-card">', unsafe_allow_html=True)
        
        if st.session_state.input_mode == 'gallery':
            st.markdown("#### 📂 Select Image")
            uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
            if uploaded_file:
                # Process upload
                image = Image.open(uploaded_file)
                run_inference(image)
                st.rerun()
                
        elif st.session_state.input_mode == 'camera':
            st.markdown("#### 📸 Capture")
            cam_img = st.camera_input("Take photo", label_visibility="collapsed")
            if cam_img:
                image = Image.open(cam_img)
                run_inference(image)
                st.rerun()
                
        else:
            # Empty State
            st.markdown("""
                <div class="placeholder-icon">🖼️</div>
                <div class="placeholder-text">Select a source below to begin scan</div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

# 3. Action Buttons
# Only show if not viewing a result (or allow switching)
if not st.session_state.detected_image:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🖼️ Gallery", use_container_width=True):
            st.session_state.input_mode = 'gallery'
            st.rerun()
    with c2:
        if st.button("📸 Camera", use_container_width=True):
            st.session_state.input_mode = 'camera'
            st.rerun()