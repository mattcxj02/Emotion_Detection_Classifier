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

def get_mask_bbox(mask_full, padding=5):
    """
    Compute a tight bounding box from the segmentation mask.
    
    Args:
        mask_full: Full-size segmentation mask (H x W)
        padding: Optional padding to add around the mask bbox
        
    Returns:
        tuple: (x1, y1, x2, y2) or None if mask is invalid
    """
    try:
        rows, cols = np.where(mask_full > 0.5)
        if len(rows) == 0 or len(cols) == 0:
            return None
            
        y1 = max(0, int(rows.min()) - padding)
        y2 = min(mask_full.shape[0], int(rows.max()) + padding)
        x1 = max(0, int(cols.min()) - padding)
        x2 = min(mask_full.shape[1], int(cols.max()) + padding)
        
        # Validate bbox
        if y2 <= y1 or x2 <= x1:
            return None
            
        return (x1, y1, x2, y2)
    except Exception as e:
        logging.warning(f"Failed to compute mask bbox: {e}")
        return None

def blend_emoji_on_face(image_bgr, x1, y1, x2, y2, emoji_img, mask_full=None, use_black=False):
    """
    Replace segmented face pixels with black color if use_black=True, otherwise blend emoji.
    """
    if use_black and mask_full is not None:
        # Simple approach: set all pixels where mask > 0.5 to black
        try:
            black_mask = (mask_full > 0.5).astype(np.uint8)
            image_bgr[black_mask == 1] = [0, 0, 0]  # Set to black (BGR)
            return image_bgr
        except Exception as e:
            logging.warning(f"Failed to apply black mask: {e}")
            return image_bgr
    
    # Original emoji blending logic
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0 or emoji_img is None:
        return image_bgr

    # Resize emoji to fit the bounding box
    emoji_resized = cv2.resize(emoji_img, (w, h))

    # Get alpha from emoji if present
    if emoji_resized.shape[2] == 4:
        emoji_alpha = emoji_resized[:, :, 3] / 255.0
        emoji_resized = emoji_resized[:, :, :3]
    else:
        emoji_alpha = np.ones((h, w), dtype=np.float32)

    # Get mask for blending
    if mask_full is not None:
        # Ensure mask_crop dimensions match the region dimensions
        try:
            mask_crop = mask_full[y1:y2, x1:x2].astype(np.float32)
            # Resize mask crop if dimensions don't match (due to rounding errors)
            if mask_crop.shape[0] != h or mask_crop.shape[1] != w:
                mask_crop = cv2.resize(mask_crop, (w, h))
        except Exception as e:
            logging.warning(f"Mask crop failed: {e}, using full region")
            mask_crop = np.ones((h, w), dtype=np.float32)
    else:
        mask_crop = np.ones((h, w), dtype=np.float32)

    # Combine emoji alpha with segmentation mask
    alpha = emoji_alpha * mask_crop
    alpha = np.expand_dims(alpha, axis=2)  # Add channel dimension for broadcasting

    # Blend all channels at once using vectorized operations
    roi = image_bgr[y1:y2, x1:x2]
    blended = (roi * (1 - alpha) + emoji_resized * alpha).astype(np.uint8)
    image_bgr[y1:y2, x1:x2] = blended

    return image_bgr

def blur_face_on_image(image_bgr, x1, y1, x2, y2, strength=23, mask_full=None):
    """
    Blur the face region in-place on the BGR image, using the segmentation mask if available.
    """
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return image_bgr

    # Make kernel size odd and at least 1
    k = max(1, int(strength))
    if k % 2 == 0:
        k += 1
    k = min(k, min(w // 2 * 2 + 1, h // 2 * 2 + 1))

    # Extract ROI and blur
    roi = image_bgr[y1:y2, x1:x2]
    try:
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
    except Exception:
        # Fallback to resize-based blur
        small = cv2.resize(roi, (max(1, w // 10), max(1, h // 10)), interpolation=cv2.INTER_LINEAR)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    if mask_full is None:
        image_bgr[y1:y2, x1:x2] = blurred
    else:
        try:
            mask_crop = mask_full[y1:y2, x1:x2].astype(np.float32)
            # Resize mask crop if dimensions don't match
            if mask_crop.shape[0] != h or mask_crop.shape[1] != w:
                mask_crop = cv2.resize(mask_crop, (w, h))
            mask_crop = np.expand_dims(mask_crop, axis=2)  # Add channel dimension
            blended = (roi * (1 - mask_crop) + blurred * mask_crop).astype(np.uint8)
            image_bgr[y1:y2, x1:x2] = blended
        except Exception as e:
            logging.warning(f"Mask-based blur failed: {e}, using full region")
            image_bgr[y1:y2, x1:x2] = blurred

    return image_bgr

def update_models(yolo_model_name, emotion_model_name):
    global yolo_current, emotion_current, models_loaded
    models_loaded = False  # Reset to reload on next request
    status = "Models will reload on next request."
    return status

def update_vlm_status(enabled, threshold):
    """Update VLM status display when checkbox or slider changes"""
    if enabled:
        return f"✓ VLM Enabled - Will validate predictions with confidence < {threshold:.0%}"
    else:
        return "✗ VLM Disabled - Using only base emotion classifier"

def process_image(image, yolo_model_name, emotion_model_name, confidence, vlm_enabled, vlm_threshold, blur_enabled=False, blur_strength=23):
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
            has_masks = r.masks is not None
            for i, box in enumerate(r.boxes):
                face_count += 1
                # Original YOLO bounding box for detection
                x1_det, y1_det, x2_det, y2_det = map(int, box.xyxy[0])
                
                # Get mask if available and resize to image dimensions
                mask_full = None
                mask_bbox = None
                if has_masks:
                    mask_data = r.masks.data[i].cpu().numpy()
                    # Resize mask to match image dimensions
                    mask_full = cv2.resize(mask_data, (frame_rgb.shape[1], frame_rgb.shape[0]))
                    # Compute tight bounding box from mask
                    mask_bbox = get_mask_bbox(mask_full, padding=5)
                    logging.info(f"Face {face_count}: YOLO box=({x1_det},{y1_det},{x2_det},{y2_det}), Mask box={mask_bbox}")

                # STEP 1: Emotion Classification (use original YOLO box for cropping)
                face = frame_rgb[y1_det:y2_det, x1_det:x2_det]

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

                # STEP 2: Emoji/Blur Overlay (use mask-based box if available, otherwise fall back to YOLO box)
                overlay_x1, overlay_y1, overlay_x2, overlay_y2 = (x1_det, y1_det, x2_det, y2_det)
                if mask_bbox is not None:
                    overlay_x1, overlay_y1, overlay_x2, overlay_y2 = mask_bbox
                    logging.info(f"Using mask-based box for overlay: {mask_bbox}")
                else:
                    logging.info(f"No valid mask bbox, using YOLO box for overlay")

                # Check if using segmentation model
                use_segmentation_black = "seg" in yolo_model_name.lower()

                # Either blur the face or apply emoji/black overlay
                if blur_enabled:
                    processed_bgr = blur_face_on_image(processed_bgr, overlay_x1, overlay_y1, overlay_x2, overlay_y2, blur_strength, mask_full)
                else:
                    if use_segmentation_black:
                        # Black out the segmented face pixels (no emoji needed)
                        processed_bgr = blend_emoji_on_face(processed_bgr, overlay_x1, overlay_y1, overlay_x2, overlay_y2, None, mask_full, use_black=True)
                    else:
                        # Load emoji and blend normally
                        emoji_path = f"emojis/{emoji_map.get(emotion, 'neutral.png')}"
                        if os.path.exists(emoji_path):
                            emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                        else:
                            emoji_img = None
                            logging.warning(f"Emoji not found: {emoji_path}")
                        processed_bgr = blend_emoji_on_face(processed_bgr, overlay_x1, overlay_y1, overlay_x2, overlay_y2, emoji_img, mask_full, use_black=False)

                # STEP 3: Create preview and draw bounding box (use original YOLO box for consistency)
                preview_img = unnormalize(input_tensor).resize((112, 112))
                vlm_indicator = " [VLM]" if vlm_used else ""
                caption = f"Face {face_count}: {emotion}{vlm_indicator} (Conf: {confidence_value:.2f}, Det: {box.conf[0]:.2f})"
                face_previews.append((preview_img, caption))

                # Draw on BGR image (use YOLO box for detection visualization)
                box_color = (255, 165, 0) if vlm_used else (0, 255, 0)  # Orange if VLM used, green otherwise
                cv2.rectangle(processed_bgr, (x1_det, y1_det), (x2_det, y2_det), box_color, 2)
                label = f"{emotion}{vlm_indicator}"
                cv2.putText(processed_bgr, label, (x1_det, y1_det - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        # Convert back to RGB for output
        processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(processed_rgb)

        if face_count == 0:
            return processed_pil, []

        return processed_pil, face_previews
    
    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        return Image.fromarray(np.array(image)), []

def process_webcam(frame, yolo_model_name, emotion_model_name, confidence, vlm_enabled, vlm_threshold, blur_enabled=False, blur_strength=23):
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
            has_masks = r.masks is not None
            for i, box in enumerate(r.boxes):
                # Original YOLO bounding box for detection
                x1_det, y1_det, x2_det, y2_det = map(int, box.xyxy[0])
                
                # Get mask if available and resize to image dimensions
                mask_full = None
                mask_bbox = None
                if has_masks:
                    mask_data = r.masks.data[i].cpu().numpy()
                    mask_full = cv2.resize(mask_data, (frame_rgb.shape[1], frame_rgb.shape[0]))
                    mask_bbox = get_mask_bbox(mask_full, padding=5)

                # STEP 1: Emotion Classification (use original YOLO box)
                face = frame_rgb[y1_det:y2_det, x1_det:x2_det]

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
                vlm_indicator = " [VLM]" if vlm_used else ""

                # STEP 2: Emoji/Blur Overlay (use mask-based box if available)
                overlay_x1, overlay_y1, overlay_x2, overlay_y2 = (x1_det, y1_det, x2_det, y2_det)
                if mask_bbox is not None:
                    overlay_x1, overlay_y1, overlay_x2, overlay_y2 = mask_bbox

                # Check if using segmentation model
                use_segmentation_black = "seg" in yolo_model_name.lower()

                # Either blur the face or apply emoji/black overlay
                if blur_enabled:
                    processed_bgr = blur_face_on_image(processed_bgr, overlay_x1, overlay_y1, overlay_x2, overlay_y2, blur_strength, mask_full)
                else:
                    if use_segmentation_black:
                        # Black out the segmented face pixels (no emoji needed)
                        processed_bgr = blend_emoji_on_face(processed_bgr, overlay_x1, overlay_y1, overlay_x2, overlay_y2, None, mask_full, use_black=True)
                    else:
                        # Load emoji and blend normally
                        emoji_path = f"emojis/{emoji_map.get(emotion, 'neutral.png')}"
                        if os.path.exists(emoji_path):
                            emoji_img = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                        else:
                            emoji_img = None
                            logging.warning(f"Emoji not found: {emoji_path}")
                        processed_bgr = blend_emoji_on_face(processed_bgr, overlay_x1, overlay_y1, overlay_x2, overlay_y2, emoji_img, mask_full, use_black=False)

                # STEP 3: Draw bounding box (use YOLO box)
                box_color = (255, 165, 0) if vlm_used else (0, 255, 0)  # Orange if VLM used, green otherwise
                label = f"{emotion}{vlm_indicator} ({confidence_value:.2f})"
                cv2.rectangle(processed_bgr, (x1_det, y1_det), (x2_det, y2_det), box_color, 2)
                cv2.putText(processed_bgr, label, (x1_det, y1_det - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

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
        yolo_model = gr.Dropdown(choices=["yolov12n-face.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolo11n-seg.pt"], label="YOLO Model", value="yolov12n-face.pt")
        emotion_model = gr.Dropdown(choices=["resnet18_emotion_classifier.pth", "combined_resnet18_emotion_classifier.pth", "combine_regnetY16GF_emotion_classifier.pth"], label="Emotion Model", value="combine_regnetY16GF_emotion_classifier.pth")
        confidence = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, label="YOLO Confidence Threshold")

    with gr.Row():
        vlm_enabled = gr.Checkbox(label="Enable VLM Validation", value=True, info="Use Qwen3-VL to validate low-confidence predictions (updates in real-time)")
        vlm_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.05, label="VLM Validation Threshold (use VLM when emotion confidence < threshold)")
        # Blurring controls
        blur_enabled = gr.Checkbox(label="Blur Faces instead of Emojis", value=False, info="When enabled, detected faces will be blurred instead of showing emojis")
        blur_strength = gr.Slider(minimum=1, maximum=101, step=2, value=23, label="Blur Strength (odd kernel size)")

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
            process_btn.click(process_image, inputs=[image_input, yolo_model, emotion_model, confidence, vlm_enabled, vlm_threshold, blur_enabled, blur_strength], outputs=[image_output, gallery_output])
        
        with gr.Tab("Webcam"):
            webcam_input = gr.Image(sources=["webcam"], type="numpy", streaming=True, label="Webcam Feed")
            webcam_input.stream(
                process_webcam,
                inputs=[webcam_input, yolo_model, emotion_model, confidence, vlm_enabled, vlm_threshold, blur_enabled, blur_strength],
                outputs=[webcam_input],
                stream_every=0.1, # Adjusted for smoother streaming
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