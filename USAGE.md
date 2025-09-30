# Face Detection & Emotion Classification GUI

A desktop application that combines YOLOv12 face detection with EfficientNet emotion classification.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required model files in the `models/` directory:
```
models/
├── yolov12n-face.pt                    # YOLOv12 face detection model (default)
├── yolov8n.pt, yolov8s.pt, yolov8m.pt  # Alternative YOLO models
├── efficientnet_b4_Tuned2_best.pth     # EfficientNet emotion model (default)
└── best_emotion_model.pth              # Alternative emotion model
```

## Running the Application

```bash
python main.py
```

## Features

### 1. **Image Loading**
- **Load Image Button**: Click to browse and select an image file
- **Drag & Drop**: Drag image files directly onto the canvas (if tkinterdnd2 is installed)
- **Supported Formats**: PNG, JPG, JPEG, BMP, TIFF

### 2. **Model Configuration**
- **YOLO Model Selection**: Choose from face-specific (yolov12n-face) or general models (yolov8n/s/m)
- **Emotion Model Selection**: Switch between different emotion classification models
- **Confidence Threshold**: Adjust detection sensitivity (0.1 - 1.0)
- **Device Info**: Shows whether CUDA or CPU is being used
- **Image Name Display**: Shows the currently loaded image filename

### 3. **Processing Pipeline**
1. **Face Detection**: YOLO model detects faces and draws bounding boxes
2. **Emotion Classification**: Each detected face is classified for emotion
3. **Results Display**: Shows both original and processed images with detailed results

### 4. **Results**
- **Visual Output**: Bounding boxes around faces with emotion labels
- **Detailed Information**: Position coordinates, confidence scores, and emotion predictions
- **Save Functionality**: Export processed images with annotations

## Expected Emotions
The model classifies faces into these categories:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Model Selection Guide

### YOLO Models
- **yolov12n-face.pt** (Recommended): Specialized face detection model for best accuracy
- **yolov8n.pt**: Fastest general object detection, may detect non-face objects
- **yolov8s.pt**: Balanced speed and accuracy
- **yolov8m.pt**: Higher accuracy, slower processing

### Emotion Models
- **efficientnet_b4_Tuned2_best.pth**: Default emotion classification model (75MB)
- **best_emotion_model.pth**: Alternative emotion model (297MB)

## Troubleshooting

### Common Issues:
1. **"Emotion model not found"**: Ensure model files exist in the `models/` directory
2. **"Drag & Drop not available"**: Install tkinterdnd2: `pip install tkinterdnd2`
3. **CUDA issues**: The app will automatically fall back to CPU if CUDA is unavailable
4. **Detecting objects instead of faces**: Switch to `yolov12n-face.pt` model

### Performance Tips:
- Use `yolov12n-face.pt` for best face detection accuracy
- Use smaller models (yolov8n, efficientnet_b4) for faster processing
- Lower confidence threshold may detect more faces but increase false positives
- GPU acceleration will significantly improve processing speed

## File Structure
```
CV/
├── main.py                             # Main GUI application
├── models/
│   ├── yolov12n-face.pt                # YOLOv12 face detection model
│   ├── yolov8n.pt, yolov8s.pt, yolov8m.pt  # YOLO models
│   ├── efficientnet_b4_Tuned2_best.pth # Emotion model (default)
│   └── best_emotion_model.pth          # Alternative emotion model
├── requirements.txt                    # Dependencies
├── README.md                           # Project overview
├── USAGE.md                            # This file (detailed usage guide)
├── yolo_detection.ipynb                # YOLO detection notebook
└── EfficientNet4_Tuned2.ipynb          # Emotion model training notebook
```