import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import time
import io

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

def optimize_image(image, max_size=1280):
    """
    Optimizes image for faster processing by resizing if too large.
    Maintains aspect ratio and improves upload/processing speed.
    """
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

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

def filter_empty_boxes(boxes, min_area=10):
    """
    Filters out empty or invalid bounding boxes.
    """
    valid_boxes = []
    for box in boxes:
        # Get box coordinates
        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
        x1, y1, x2, y2 = xyxy
        
        # Calculate area
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Filter out boxes with invalid dimensions or too small area
        if width > 0 and height > 0 and area >= min_area:
            valid_boxes.append(box)
    
    return valid_boxes

def draw_boxes_on_image(image, boxes, model_names, conf_threshold=0.25):
    """
    Manually draw only valid boxes on the image to avoid empty boxes.
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Filter valid boxes
    valid_boxes = filter_empty_boxes(boxes, min_area=10)
    
    # Draw only valid boxes
    for box in valid_boxes:
        # Get coordinates
        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Get class and confidence
        cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], 'cpu') else int(box.cls[0])
        conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], 'cpu') else float(box.conf[0])
        
        if conf >= conf_threshold:
            # Get class name
            class_name = model_names[cls_id]
            
            # Draw rectangle
            color = (0, 255, 0)  # Green color
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_cv, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(img_cv, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Convert back to RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


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
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'], help="Upload an image file for analysis")
        if uploaded_file is not None:
            try:
                # Read image directly from bytes for faster processing
                image_bytes = uploaded_file.read()
                image_source = Image.open(io.BytesIO(image_bytes))
                # Optimize image size for faster processing
                image_source = optimize_image(image_source)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                image_source = None
            
    with tab2:
        cam_img = st.camera_input("Capture Feed")
        if cam_img is not None:
            try:
                # Read image directly from bytes for faster processing
                image_bytes = cam_img.read()
                image_source = Image.open(io.BytesIO(image_bytes))
                # Optimize image size for faster processing
                image_source = optimize_image(image_source)
            except Exception as e:
                st.error(f"Error loading camera image: {str(e)}")
                image_source = None

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
        try:
            with st.spinner("Processing image..."):
                # 1. Resize Input for Processing/Display (Standard 16:9)
                display_img = resize_16_9(image_source)
                
                # Inference - use optimized settings for faster processing
                results = model.predict(
                    source=display_img, 
                    save=False, 
                    conf=conf_threshold,
                    imgsz=640,  # Standard size for faster inference
                    verbose=False  # Reduce output for speed
                )
                
                # Filter out empty boxes and draw only valid ones
                boxes = results[0].boxes
                if len(boxes) > 0:
                    valid_boxes = filter_empty_boxes(boxes)
                    
                    # Only draw boxes if we have valid ones
                    if len(valid_boxes) > 0:
                        # Manually draw only valid boxes to avoid empty boxes
                        res_img = draw_boxes_on_image(display_img, boxes, model.names, conf_threshold)
                    else:
                        # No valid boxes, show original image
                        res_img = display_img
                else:
                    # No boxes detected, show original image
                    res_img = display_img
                
                st.image(res_img, use_container_width=True, caption="Processed Output (16:9 Stream)")
                
                # Results Metrics
                st.markdown("#### 📊 Detection Metrics")
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # Filter empty boxes before counting
                    valid_boxes = filter_empty_boxes(boxes)
                    
                    if len(valid_boxes) > 0:
                        detected_counts = {}
                        for box in valid_boxes:
                            cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], 'cpu') else int(box.cls[0])
                            name = model.names[cls_id]
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
                        st.info("No valid detections found (empty boxes filtered out).")
                else:
                    st.info("No anomalies detected in the current frame.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.info("Please try uploading the image again.")
            
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