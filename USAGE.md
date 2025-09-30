# Face Detection & Emotion Classification GUI

A desktop application that combines YOLO face detection with EfficientNet emotion classification.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the emotion model file in the correct location:
```
models/efficientnet_b4_Tuned2_best.pth
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
- **YOLO Model Selection**: Choose from different YOLO model sizes (n, s, m, l)
- **Confidence Threshold**: Adjust detection sensitivity (0.1 - 1.0)
- **Device Info**: Shows whether CUDA or CPU is being used

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

## Troubleshooting

### Common Issues:
1. **"Emotion model not found"**: Ensure `models/efficientnet_b4_Tuned2_best.pth` exists
2. **"Drag & Drop not available"**: Install tkinterdnd2: `pip install tkinterdnd2`
3. **CUDA issues**: The app will automatically fall back to CPU if CUDA is unavailable

### Performance Tips:
- Use smaller YOLO models (yolov8n.pt) for faster processing
- Lower confidence threshold may detect more faces but increase false positives
- GPU acceleration will significantly improve processing speed

## File Structure
```
CV/
├── main.py          # Main GUI application
├── models/
│   └── efficientnet_b4_Tuned2_best.pth  # Emotion classification model
├── requirements.txt            # Dependencies
├── USAGE.md                    # This file
├── yolo_detection.ipynb        # Original notebook reference
└── EfficientNet4_Tuned2.ipynb  # Model training notebook
```