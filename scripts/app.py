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
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------
# Theme Management
# -----------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'light_blue'

def set_theme(theme_name):
    st.session_state.theme = theme_name
    st.rerun()

# Theme CSS Definitions - Removed Dark Blue theme
themes = {
    'light_blue': """
        /* Light Blue Theme */
        .stApp {
            background: linear-gradient(135deg, #e0f7fa 0%, #b3e5fc 50%, #81d4fa 100%);
            font-family: 'Outfit', sans-serif;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.85);
            box-shadow: 0 8px 32px 0 rgba(33, 150, 243, 0.2);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(179, 229, 252, 0.6);
        }
        .navbar {
            background: rgba(255, 255, 255, 0.9);
            border-bottom: 1px solid rgba(179, 229, 252, 0.7);
            box-shadow: 0 4px 20px rgba(33, 150, 243, 0.15);
        }
        .nav-logo { color: #0288d1; }
        .nav-links a { color: #0277bd; }
        .nav-links a:hover { color: #01579b; }
        h1, h2, h3 { color: #01579b; }
        p { color: #0277bd; }
        div.stButton > button {
            background: linear-gradient(90deg, #29b6f6 0%, #4fc3f7 100%);
            box-shadow: 0 4px 15px rgba(41, 182, 246, 0.4);
        }
        .metric-value { color: #0288d1; }
    """,
    'soft_purple': """
        /* Soft Purple Theme */
        .stApp {
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 50%, #ce93d8 100%);
            font-family: 'Outfit', sans-serif;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.9);
            box-shadow: 0 8px 32px 0 rgba(156, 39, 176, 0.2);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(225, 190, 231, 0.6);
        }
        .navbar {
            background: rgba(255, 255, 255, 0.95);
            border-bottom: 1px solid rgba(225, 190, 231, 0.7);
            box-shadow: 0 4px 20px rgba(156, 39, 176, 0.15);
        }
        .nav-logo { color: #7b1fa2; }
        .nav-links a { color: #6a1b9a; }
        .nav-links a:hover { color: #4a148c; }
        h1, h2, h3 { color: #4a148c; }
        p { color: #6a1b9a; }
        div.stButton > button {
            background: linear-gradient(90deg, #9c27b0 0%, #ab47bc 100%);
            box-shadow: 0 4px 15px rgba(156, 39, 176, 0.4);
        }
        .metric-value { color: #8e24aa; }
    """
}

# -----------------------------
# Custom Styling with Theme Support - Fixed CSS syntax
# -----------------------------
st.markdown(f"""
    <style>
        /* Import Font */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;700&display=swap');
        
        /* Main Container Background - Dynamic Theme */
        {themes[st.session_state.theme]}
        
        /* Remove Extra Streamlit Whitespace */
        .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 0rem !important;
            max-width: 95% !important;
        }}
        
        /* Streamlined Glassmorphism Classes */
        .glass-card {{
            background: rgba(255, 255, 255, 0.85);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.4);
            padding: 1.2rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }}
        
        /* Navbar Style - Fixed to show full title */
        .navbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 0 0 16px 16px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            min-width: 100%;
            width: 100%;
            box-sizing: border-box;
        }}
                
        /* Navigation Logo - Ensured full visibility */
        .nav-logo {{ 
            font-size: 1.6rem; 
            font-weight: 700; 
            color: #2c3e50; 
            gap: 12px; 
            display: flex; 
            align-items: center;
            white-space: nowrap;
            overflow: visible;
            min-width: fit-content;
        }}
        .nav-links a {{ 
            text-decoration: none; 
            color: #5d6d7e; 
            margin-left: 20px; 
            font-weight: 500; 
            transition: color 0.3s;
        }}
        .nav-links a:hover {{ color: #3498db; }}
        
        /* Typography */
        h1, h2, h3 {{ 
            color: #2c3e50; 
            margin-top: 0 !important; 
            font-weight: 600;
        }}
        p {{ 
            color: #5d6d7e; 
            line-height: 1.6; 
        }}
        
        /* Buttons */
        div.stButton > button {{
            background: linear-gradient(90deg, #66a6ff 0%, #89f7fe 100%);
            color: white; 
            border: none; 
            padding: 0.6rem 1.2rem; 
            border-radius: 12px; 
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 166, 255, 0.4); 
            width: 100%;
            transition: all 0.3s ease;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.2) !important;
        }}
        
        /* Metrics */
        .metric-container {{ 
            display: flex; 
            gap: 10px; 
            flex-wrap: wrap; 
            margin-top: 1rem; 
        }}
        .metric-card {{
            background: white; 
            padding: 10px; 
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); 
            flex: 1; 
            min-width: 100px;
            text-align: center; 
            border: 1px solid rgba(0,0,0,0.05);
        }}
        .metric-value {{ 
            font-size: 1.2rem; 
            font-weight: 700; 
            color: #3498db; 
        }}
        .metric-label {{ 
            font-size: 0.7rem; 
            color: #95a5a6; 
            text-transform: uppercase; 
        }}
        
        /* Images */
        img {{ 
            border-radius: 12px; 
        }}
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
            <span>🚀</span> SpaceSafety AI
        </div>
        <div class="nav-links">
            <a href="#">Dashboard</a>
            <a href="#">Analysis</a>
            <a href="#">Reports</a>
            <a href="#">System Status</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# Theme Toggle Section - Removed Dark Blue theme
theme_col1, theme_col2 = st.columns([1, 1])
with theme_col1:
    if st.button("🔵 Light Blue", key="theme_light_blue", help="Switch to Light Blue theme"):
        set_theme('light_blue')
with theme_col2:
    if st.button("🟣 Soft Purple", key="theme_soft_purple", help="Switch to Soft Purple theme"):
        set_theme('soft_purple')

# Main Content Grid
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Control Center")
    st.write("Upload mission imagery for automated defect analysis.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'], help="Upload an image file for analysis")
    image_source = None
    
    if uploaded_file is not None:
        try:
            # Read image directly from bytes for faster processing
            image_bytes = uploaded_file.read()
            image_source = Image.open(io.BytesIO(image_bytes))
            # Optimize image size for faster processing
            image_source = optimize_image(image_source)
            st.success("Image Loaded Successfully")
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            image_source = None
    
    # Analyze button
    analyze_clicked = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


with col_right:
    # Only show the visualization card when there's content and analyze is clicked
    if image_source and analyze_clicked:
        st.markdown('<div class="glass-card" style="min-height: 500px;">', unsafe_allow_html=True)
        st.markdown("### 👁️ Visual Analysis Feed")
        
        # Process image without creating extra containers
        try:
            # Resize Input for Processing/Display (Standard 16:9)
            display_img = resize_16_9(image_source)
            
            # Show processing message instead of spinner to avoid extra container
            st.info("Processing image...")
            
            # Inference - use optimized settings for faster processing with fixed confidence threshold
            results = model.predict(
                source=display_img, 
                save=False, 
                conf=0.25,  # Fixed confidence threshold
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
                    res_img = draw_boxes_on_image(display_img, boxes, model.names, 0.25)
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
            
        st.markdown('</div>', unsafe_allow_html=True)
    elif image_source and not analyze_clicked:
        # Image loaded but not analyzed yet
        st.markdown("""
        <div style="text-align: center; padding: 50px 0; opacity: 0.7;">
            <div style="font-size: 60px; margin-bottom: 15px;">✅</div>
            <h3>Image Ready for Analysis</h3>
            <p style="font-size: 1.1em;">Click the 'Analyze Image' button to start defect detection</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # No image uploaded
        st.markdown("""
        <div style="text-align: center; padding: 50px 0; opacity: 0.6;">
            <div style="font-size: 60px; margin-bottom: 15px;">🖼️</div>
            <h3>Upload an Image to Begin Analysis</h3>
            <p style="font-size: 1.1em;">Use the Control Center to upload mission imagery for automated defect detection</p>
        </div>
        """, unsafe_allow_html=True)