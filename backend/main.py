from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import uuid
import os
import uvicorn

app = FastAPI(title="SpaceGuard AI API")

# Load model (ensure best.pt is in the same directory)
model = YOLO("best.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Create temp filename
    ext = file.filename.split(".")[-1]
    img_name = f"{uuid.uuid4()}.{ext}"
    img_path = f"temp_{img_name}"

    # Save uploaded file
    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run inference
    results = model(img_path)

    # Process results
    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": bbox
            })

    # Cleanup temp file
    if os.path.exists(img_path):
        os.remove(img_path)

    return {
        "detections": detections
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
