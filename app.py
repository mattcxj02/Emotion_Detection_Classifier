import gradio as gr
import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels (7 emotions)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global models (lazy-loaded)
yolo_current = None
emotion_current = None
models_loaded = False  # Flag to load once

def ensure_models_loaded(yolo_model_name, emotion_model_name):
    global yolo_current, emotion_current, models_loaded
    if models_loaded:
        return
    try:
        yolo_current = YOLO(f"models/{yolo_model_name}")
        logging.info(f"YOLO model {yolo_model_name} loaded.")
        emotion_current = load_emotion_model(emotion_model_name)
        logging.info(f"Emotion model {emotion_model_name} loaded.")
        models_loaded = True
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        raise e  # Re-raise to handle in caller

def load_emotion_model(model_name):
    path = f"models/{model_name}"
    num_classes = 7  # Default for most models
    model = None

    if "resnet" in model_name.lower():
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif "efficientnet" in model_name.lower():
        num_classes = 8  # Adjust if needed for your specific model
        model = models.efficientnet_b4(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    else:
        # Default to efficientnet
        num_classes = 8
        model = models.efficientnet_b4(weights=None)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().detach().cpu().numpy().transpose((1, 2, 0))
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def update_models(yolo_model_name, emotion_model_name):
    global yolo_current, emotion_current, models_loaded
    models_loaded = False  # Reset to reload on next process
    status = "Models will reload on next request."
    return status

def process_image(image, yolo_model_name, emotion_model_name, confidence):
    global yolo_current, emotion_current
    
    if image is None:
        return None, []
    
    try:
        ensure_models_loaded(yolo_model_name, emotion_model_name)  # Lazy load here
        
        # Convert Gradio image (PIL) to OpenCV
        img_array = np.array(image)
        if img_array.size == 0:
            return None, []
        
        frame_rgb = img_array
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        processed_bgr = frame_bgr.copy()

        # Face detection (YOLO expects RGB)
        results = yolo_current(frame_rgb, conf=confidence)
        face_previews = []
        face_count = 0

        for r in results:
            for box in r.boxes:
                face_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame_rgb[y1:y2, x1:x2]

                # Emotion classification
                pil_face = Image.fromarray(face)
                input_tensor = transform(pil_face)
                with torch.no_grad():
                    output = emotion_current(input_tensor.unsqueeze(0).to(device))
                    _, predicted = torch.max(output, 1)
                    emotion = emotion_labels[predicted.item() % len(emotion_labels)]  # Adjust index if num_classes > 7

                # Create preview
                preview_img = unnormalize(input_tensor).resize((112, 112))
                caption = f"Face {face_count}: {emotion} (Confidence: {box.conf[0]:.2f})"
                face_previews.append((preview_img, caption))

                # Draw on BGR image
                cv2.rectangle(processed_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_bgr, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert back to RGB for output
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(processed_rgb)

        if face_count == 0:
            return processed_pil, []

        return processed_pil, face_previews
    
    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        return Image.fromarray(np.array(image)), []

def process_webcam(frame, yolo_model_name, emotion_model_name, confidence):
    global yolo_current, emotion_current
    
    if frame is None or frame.size == 0:
        logging.info("Empty frame received")
        return frame
    
    try:
        ensure_models_loaded(yolo_model_name, emotion_model_name)  # Lazy load here
        
        frame_rgb = frame  # Already numpy RGB
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        processed_bgr = frame_bgr.copy()

        # Face detection
        results = yolo_current(frame_rgb, conf=confidence)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame_rgb[y1:y2, x1:x2]

                # Emotion classification
                pil_face = Image.fromarray(face)
                input_tensor = transform(pil_face)
                with torch.no_grad():
                    output = emotion_current(input_tensor.unsqueeze(0).to(device))
                    _, predicted = torch.max(output, 1)
                emotion = emotion_labels[predicted.item() % len(emotion_labels)]  # Adjust index if num_classes > 7

                # Draw on BGR
                cv2.rectangle(processed_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(processed_bgr, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Convert back to RGB
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        logging.info("Webcam frame processed successfully")
        return processed_rgb
    
    except Exception as e:
        logging.error(f"Error during webcam processing: {e}")
        return frame_rgb  # Return original on error

# Gradio interface (unchanged)
with gr.Blocks(title="Face Detection & Emotion Classification") as demo:
    gr.Markdown("Upload an image or use webcam to detect faces and classify emotions.")
    
    with gr.Row():
        yolo_model = gr.Dropdown(choices=["yolov12n-face.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], label="YOLO Model", value="yolov12n-face.pt")
        emotion_model = gr.Dropdown(choices=["resnet18_emotion_classifier.pth", "efficientnet_b4_Tuned2_best.pth", "best_emotion_model.pth"], label="Emotion Model", value="resnet18_emotion_classifier.pth")
        confidence = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, label="Confidence Threshold")
    
    status = gr.Textbox(label="Model Status", value="Models load on first request.")
    
    yolo_model.change(update_models, inputs=[yolo_model, emotion_model], outputs=status)
    emotion_model.change(update_models, inputs=[yolo_model, emotion_model], outputs=status)
    
    with gr.Tabs():
        with gr.Tab("Upload Image"):
            image_input = gr.Image(sources=["upload"], type="pil", label="Upload Image")
            process_btn = gr.Button("Process Image")
            with gr.Row():
                image_output = gr.Image(label="Processed Image")
            gallery_output = gr.Gallery(label="Face Previews", show_label=True, columns=3, height="auto")
            process_btn.click(process_image, inputs=[image_input, yolo_model, emotion_model, confidence], outputs=[image_output, gallery_output])
        
        with gr.Tab("Webcam"):
            webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam Feed")
            webcam_input.stream(
                process_webcam, 
                inputs=[webcam_input, yolo_model, emotion_model, confidence], 
                outputs=[webcam_input],
                stream_every=0.02, 
                concurrency_limit=10
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,  # Disable public sharing in cloud
        quiet=False,  # Enable logs for debugging
        show_error=True  # Show errors in browser
    )
    print(f"Gradio server started on 0.0.0.0:{port}")  # Log bind success