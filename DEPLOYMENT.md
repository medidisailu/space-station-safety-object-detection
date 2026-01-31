# 🚀 Deployment Guide for SpaceGuard AI

## 🎯 Deployment Options

### Option 1: Streamlit Cloud (Recommended - FREE)

Perfect for final-year projects, demos, and sharing with reviewers.

#### Steps:

1. **Create GitHub Repository**
   - Go to [github.com](https://github.com) and create a new repository
   - Name it something like `space-safety-object-detection`
   - Don't initialize with README (we already have one)

2. **Push Code to GitHub**
   ```bash
   git remote add origin https://github.com/your-username/space-safety-object-detection.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy to Streamlit Cloud**
   - Visit [https://share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Fill in:
     - **Repository**: your-username/space-safety-object-detection
     - **Branch**: main
     - **Main file path**: scripts/app.py
   - Click "Deploy"

4. **Get Your URL**
   - You'll receive a URL like: `https://your-app-name.streamlit.app`
   - This is publicly accessible by anyone!

#### Benefits:
- ✅ Completely FREE
- ✅ No server management
- ✅ Automatic HTTPS
- ✅ Works on mobile & desktop
- ✅ Perfect for presentations

---

### Option 2: Local Network Access

For temporary sharing within your local network (WiFi/LAN).

#### Steps:
```bash
streamlit run scripts/app.py --server.address 0.0.0.0
```

Then others on the same network can access:
```
http://YOUR_IP_ADDRESS:8501
```

Find your IP with:
```bash
ipconfig  # Windows
ifconfig  # Mac/Linux
```

---

### Option 3: Docker Deployment (Advanced)

For production environments or custom servers.

#### Dockerfile:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "scripts/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run:
```bash
docker build -t spaceguard-ai .
docker run -p 8501:8501 spaceguard-ai
```

---

## 📁 Project Structure for Deployment

```
space-safety-object-detection/
├── scripts/
│   └── app.py              # Main Streamlit application
├── runs/
│   └── detect/
│       └── train6/
│           └── weights/
│               └── best.pt  # Trained model weights
├── data/
│   └── data.yaml           # Dataset configuration
├── requirements.txt        # Python dependencies
├── packages.txt           # System dependencies
├── runtime.txt            # Python version
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

---

## ⚠️ Important Notes

### File Size Limitations:
- GitHub: 100MB per file limit
- Streamlit Cloud: 1GB total repository size
- Keep only essential files in repo

### Model File:
- `best.pt` (~22MB) is included in the repo
- For larger models, consider:
  1. Using a model hosting service
  2. Downloading model at runtime
  3. Using Git LFS for large files

### Dependencies:
Make sure `requirements.txt` includes:
```
streamlit
ultralytics
torch
torchvision
opencv-python-headless
numpy
Pillow
```

---

## 🎓 For Your Final Year Project

### What to Tell Your HOD/Reviewer:

> "This project is deployed as a cloud-based AI web application using Streamlit Cloud, allowing real-time object detection through a browser interface accessible globally. The system uses YOLOv8 for detection and provides an interactive dashboard for uploading images and viewing results."

### Technical Highlights:
- Real-time object detection using computer vision
- Deployed on cloud platform (Streamlit Cloud)
- Accessible via web browser (no installation required)
- Responsive design works on desktop and mobile
- Professional UI with space-themed interface

---

## 🆘 Troubleshooting

### Common Issues:

1. **Deployment fails due to file size**
   - Remove unnecessary files from `runs/detect/`
   - Keep only `best.pt` in weights folder
   - Use `.gitignore` to exclude large files

2. **Model not found error**
   - Ensure `runs/detect/train6/weights/best.pt` exists
   - Check model path in `app.py` line 173

3. **Import errors**
   - Verify all dependencies in `requirements.txt`
   - Check Python version in `runtime.txt`

4. **Slow loading**
   - Model loading takes time on first run
   - Subsequent predictions are faster
   - Consider model optimization for production

---

## 📞 Support

For deployment issues:
1. Check Streamlit Cloud status: [status.streamlit.io](https://status.streamlit.io)
2. Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
3. Check GitHub repository settings
4. Verify all files are committed and pushed

---
*Last updated: January 2026*