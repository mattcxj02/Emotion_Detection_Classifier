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
import ssl
import certifi
from torchvision.models import regnet_y_1_6gf, RegNet_Y_1_6GF_Weights, ResNet18_Weights, EfficientNet_B4_Weights
from transformers import pipeline
import torch.nn.functional as F

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context

# Set up logging
logging.basicConfig(level=logging.INFO)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Emotion labels (7 emotions)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Emoji mapping (dict of emotion to PNG filename)
emoji_map = {
    'Angry': 'angry.png',
    'Disgust': 'disgust.png',
    'Fear': 'fear.png',
    'Happy': 'happy.png',
    'Sad': 'sad.png',
    'Surprise': 'surprise.png',
    'Neutral': 'neutral.png'
}

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global models (lazy-loaded)
yolo_current = None
emotion_current = None
vlm_pipeline = None
models_loaded = False  # Flag to load once
vlm_loaded = False  # Flag for VLM loading
VLM_CONFIDENCE_THRESHOLD = 0.7  # Threshold for invoking VLM validation

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
    num_classes = 7
    model = None

    if "resnet" in model_name.lower():
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    elif "efficientnet" in model_name.lower():
        model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    elif "regnet" in model_name.lower():
        model = models.regnet_y_1_6gf(weights=RegNet_Y_1_6GF_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        # Default to efficientnet
        model = models.efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
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

def load_vlm_model():
    """Load Qwen3-VL pipeline for emotion validation"""
    global vlm_pipeline, vlm_loaded

    if vlm_loaded:
        return

    try:
        logging.info("Loading Qwen3-VL pipeline for validation...")
        vlm_pipeline = pipeline(
            "image-text-to-text",
            model="Qwen/Qwen3-VL-2B-Instruct",
            device=0 if torch.cuda.is_available() else -1,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        vlm_loaded = True
        logging.info("VLM pipeline loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load VLM pipeline: {e}")
        vlm_loaded = False

def validate_emotion_with_vlm(face_image_pil, predicted_emotion, confidence, threshold=0.7, enabled=True):
    """
    Use Qwen3-VL to validate emotion prediction when confidence is low.

    Args:
        face_image_pil: PIL Image of the face
        predicted_emotion: The emotion predicted by the classifier
        confidence: Confidence score of the prediction
        threshold: Confidence threshold for VLM validation
        enabled: Whether VLM validation is enabled

    Returns:
        tuple: (validated_emotion, is_vlm_used)
    """
    global vlm_pipeline, vlm_loaded

    # Log all parameters for debugging
    logging.info(f"VLM function called: enabled={enabled}, confidence={confidence:.2%}, threshold={threshold:.2%}, predicted={predicted_emotion}")

    # Check if VLM is enabled
    if not enabled:
        logging.info("VLM disabled by checkbox, skipping validation")
        return predicted_emotion, False

    logging.info(f"VLM enabled, checking threshold: will_use_vlm={confidence < threshold}")

    # Only use VLM if confidence is below threshold
    if confidence >= threshold:
        return predicted_emotion, False

    # Load VLM if not already loaded
    if not vlm_loaded:
        load_vlm_model()

    # If VLM failed to load, return original prediction
    if not vlm_loaded or vlm_pipeline is None:
        logging.warning("VLM not available, using original prediction")
        return predicted_emotion, False

    try:
        # Prepare the prompt for emotion classification
        prompt = f"""Analyze this facial expression and classify the emotion.

The current model predicted: {predicted_emotion} with {confidence:.2%} confidence.

Please classify the emotion into ONE of these categories ONLY:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

Respond with ONLY the emotion label, nothing else."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": face_image_pil},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Use pipeline for inference
        result = vlm_pipeline(
            text=messages,
            max_new_tokens=20,
            temperature=0.7,
            top_p=0.8
        )

        # Extract output text from pipeline result
        # Debug: Log the raw result structure
        logging.info(f"VLM raw result type: {type(result)}, content: {result}")

        # Handle different possible output formats
        if isinstance(result, list) and len(result) > 0:
            first_result = result[0]
            if isinstance(first_result, dict):
                # Format: [{'generated_text': '...'}]
                output_text = first_result.get('generated_text', '')
                if isinstance(output_text, list):
                    # If generated_text is a list, join or take last message
                    output_text = output_text[-1]['content'] if output_text else ''
            elif isinstance(first_result, str):
                # Format: ['text']
                output_text = first_result
            else:
                output_text = str(first_result)
        elif isinstance(result, dict):
            # Format: {'generated_text': '...'}
            output_text = result.get('generated_text', '')
            if isinstance(output_text, list):
                output_text = output_text[-1]['content'] if output_text else ''
        else:
            output_text = str(result)

        # Clean up the output
        if isinstance(output_text, str):
            vlm_emotion = output_text.strip()
        else:
            vlm_emotion = str(output_text).strip()

        logging.info(f"VLM extracted emotion text: '{vlm_emotion}'")

        # Validate that VLM output is a valid emotion
        if vlm_emotion in emotion_labels:
            logging.info(f"VLM validation: {predicted_emotion} -> {vlm_emotion} (original confidence: {confidence:.2%})")
            return vlm_emotion, True
        else:
            # Try to find emotion in the output text
            for emotion in emotion_labels:
                if emotion.lower() in vlm_emotion.lower():
                    logging.info(f"VLM validation (extracted): {predicted_emotion} -> {emotion} (original confidence: {confidence:.2%})")
                    return emotion, True

            # If no valid emotion found, use original
            logging.warning(f"VLM output invalid: '{vlm_emotion}', using original prediction")
            return predicted_emotion, False

    except Exception as e:
        logging.error(f"VLM validation error: {e}")
        return predicted_emotion, False

def unnormalize(tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().detach().cpu().numpy().transpose((1, 2, 0))
    tensor = std * tensor + mean
    tensor = np.clip(tensor, 0, 1)
    return Image.fromarray((tensor * 255).astype(np.uint8))

def blend_emoji_on_face(image_bgr, x1, y1, x2, y2, emoji_resized):
    """
    Blend transparent emoji over the face region in the BGR image.
    Uses simple alpha blending with the bounding box as mask.
    """
    if emoji_resized is None:
        return image_bgr
    
    # Create mask from bounding box (simple rectangular mask)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    mask.fill(255)  # Full mask for the box
    
    # Alpha channel handling
    alpha = emoji_resized[:, :, 3] / 255.0 if emoji_resized.shape[2] == 4 else np.ones((y2 - y1, x2 - x1), dtype=np.float32)
    
    # Blend each channel (BGR)
    for c in range(3):
        roi = image_bgr[y1:y2, x1:x2, c]
        emoji_channel = emoji_resized[:, :, c] * alpha
        blended = (roi * (1 - alpha) + emoji_channel).astype(np.uint8)
        image_bgr[y1:y2, x1:x2, c] = blended
    
    return image_bgr

def update_models(yolo_model_name, emotion_model_name):
    global yolo_current, emotion_current, models_loaded
    models_loaded = False  # Reset to reload on next process
    status = "Models will reload on next request."
    return status

def update_vlm_status(enabled, threshold):
    """Update VLM status display when checkbox or slider changes"""
    if enabled:
        return f"✓ VLM Enabled - Will validate predictions with confidence < {threshold:.0%}"
    else:
        return "✗ VLM Disabled - Using only base emotion classifier"

def process_image(image, yolo_model_name, emotion_model_name, confidence, vlm_enabled, vlm_threshold):
    global yolo_current, emotion_current

    logging.info(f"process_image called: vlm_enabled={vlm_enabled}, vlm_threshold={vlm_threshold}")

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
                    probabilities = F.softmax(output, dim=1)
                    confidence_score, predicted = torch.max(probabilities, 1)
                    confidence_value = confidence_score.item()
                    emotion = emotion_labels[predicted.item() % len(emotion_labels)]  # Adjust index if num_classes > 7

                # VLM validation for low-confidence predictions
                logging.info(f"About to call VLM validation: emotion={emotion}, confidence={confidence_value:.2%}, threshold={vlm_threshold:.2%}, enabled={vlm_enabled}")
                vlm_used = False
                emotion, vlm_used = validate_emotion_with_vlm(pil_face, emotion, confidence_value, vlm_threshold, vlm_enabled)
                logging.info(f"VLM validation result: emotion={emotion}, vlm_used={vlm_used}")

                # Load and resize emoji to face size
                emoji_path = f"emojis/{emoji_map.get(emotion, 'neutral.png')}"  # Fallback to neutral
                if os.path.exists(emoji_path):
                    emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)  # Preserve alpha
                    h, w = y2 - y1, x2 - x1
                    emoji_resized = cv2.resize(emoji_img, (w, h))
                else:
                    emoji_resized = None
                    logging.warning(f"Emoji not found: {emoji_path}")

                # Blend emoji on the face
                processed_bgr = blend_emoji_on_face(processed_bgr, x1, y1, x2, y2, emoji_resized)

                # Create preview
                preview_img = unnormalize(input_tensor).resize((112, 112))
                vlm_indicator = " [VLM]" if vlm_used else ""
                caption = f"Face {face_count}: {emotion}{vlm_indicator} (Conf: {confidence_value:.2f}, Det: {box.conf[0]:.2f})"
                face_previews.append((preview_img, caption))

                # Draw on BGR image
                box_color = (255, 165, 0) if vlm_used else (0, 255, 0)  # Orange if VLM used, green otherwise
                cv2.rectangle(processed_bgr, (x1, y1), (x2, y2), box_color, 2)
                label = f"{emotion}{vlm_indicator}"
                cv2.putText(processed_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Convert back to RGB for output
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(processed_rgb)

        if face_count == 0:
            return processed_pil, []

        return processed_pil, face_previews
    
    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        return Image.fromarray(np.array(image)), []

def process_webcam(frame, yolo_model_name, emotion_model_name, confidence, vlm_enabled, vlm_threshold):
    global yolo_current, emotion_current

    # Log parameters (only first time to avoid spam)
    if not hasattr(process_webcam, "logged"):
        logging.info(f"process_webcam called: vlm_enabled={vlm_enabled}, vlm_threshold={vlm_threshold}")
        process_webcam.logged = True

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
                    probabilities = F.softmax(output, dim=1)
                    confidence_score, predicted = torch.max(probabilities, 1)
                    confidence_value = confidence_score.item()
                emotion = emotion_labels[predicted.item() % len(emotion_labels)]  # Adjust index if num_classes > 7

                # VLM validation for low-confidence predictions
                # Note: VLM may slow down real-time processing, but checkbox allows toggling
                vlm_used = False
                emotion, vlm_used = validate_emotion_with_vlm(pil_face, emotion, confidence_value, vlm_threshold, vlm_enabled)

                # Load and resize emoji to face size
                emoji_path = f"emojis/{emoji_map.get(emotion, 'neutral.png')}"
                if os.path.exists(emoji_path):
                    emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                    h, w = y2 - y1, x2 - x1
                    emoji_resized = cv2.resize(emoji_img, (w, h))
                else:
                    emoji_resized = None

                # Blend emoji on the face
                processed_bgr = blend_emoji_on_face(processed_bgr, x1, y1, x2, y2, emoji_resized)

                # Draw on BGR
                box_color = (255, 165, 0) if vlm_used else (0, 255, 0)  # Orange if VLM used, green otherwise
                label = f"{emotion} ({confidence_value:.2f})"
                cv2.rectangle(processed_bgr, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(processed_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Convert back to RGB
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

        logging.info("Webcam frame processed successfully")
        return processed_rgb
    
    except Exception as e:
        logging.error(f"Error during webcam processing: {e}")
        return frame  # Return original on error

# Gradio interface
with gr.Blocks(title="Face Detection & Emotion Classification") as demo:
    gr.Markdown("# Face Detection & Emotion Classification")
    gr.Markdown("Upload an image or use webcam to detect faces and classify emotions.")
    gr.Markdown("**VLM Validation:** When emotion confidence is low, Qwen3-VL (Vision-Language Model) validates predictions. Orange boxes indicate VLM-validated results.")
    
    with gr.Row():
        yolo_model = gr.Dropdown(choices=["yolov12n-face.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], label="YOLO Model", value="yolov12n-face.pt")
        emotion_model = gr.Dropdown(choices=["resnet18_emotion_classifier.pth", "combined_resnet18_emotion_classifier.pth", "combine_regnetY16GF_emotion_classifier.pth"], label="Emotion Model", value="combine_regnetY16GF_emotion_classifier.pth")
        confidence = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, label="YOLO Confidence Threshold")

    with gr.Row():
        vlm_enabled = gr.Checkbox(label="Enable VLM Validation", value=True, info="Use Qwen3-VL to validate low-confidence predictions (updates in real-time)")
        vlm_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="VLM Validation Threshold (use VLM when emotion confidence < threshold)")

    with gr.Row():
        status = gr.Textbox(label="Model Status", value="Models load on first request.")
        vlm_status = gr.Textbox(label="VLM Status", value="✓ VLM Enabled - Will validate predictions with confidence < 70%")
    
    yolo_model.change(update_models, inputs=[yolo_model, emotion_model], outputs=status)
    emotion_model.change(update_models, inputs=[yolo_model, emotion_model], outputs=status)

    # Update VLM status when checkbox or threshold changes
    vlm_enabled.change(update_vlm_status, inputs=[vlm_enabled, vlm_threshold], outputs=vlm_status)
    vlm_threshold.change(update_vlm_status, inputs=[vlm_enabled, vlm_threshold], outputs=vlm_status)
    
    with gr.Tabs():
        with gr.Tab("Upload Image"):
            image_input = gr.Image(sources=["upload"], type="pil", label="Upload Image")
            process_btn = gr.Button("Process Image")
            with gr.Row():
                image_output = gr.Image(label="Processed Image")
            gallery_output = gr.Gallery(label="Face Previews", show_label=True, columns=3, height="auto")
            process_btn.click(process_image, inputs=[image_input, yolo_model, emotion_model, confidence, vlm_enabled, vlm_threshold], outputs=[image_output, gallery_output])
        
        with gr.Tab("Webcam"):
            webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam Feed")
            webcam_input.stream(
                process_webcam,
                inputs=[webcam_input, yolo_model, emotion_model, confidence, vlm_enabled, vlm_threshold],
                outputs=[webcam_input],
                stream_every=0.05, # Adjusted for smoother streaming
                concurrency_limit=10
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    demo.launch(
        server_name="localhost",
        server_port=port,
        share=False,  # Disable public sharing in cloud
        quiet=False,  # Enable logs for debugging
        show_error=True  # Show errors in browser
    )
    print(f"Gradio server started on 0.0.0.0:{port}")  # Log bind success