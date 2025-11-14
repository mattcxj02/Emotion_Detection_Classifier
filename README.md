# Face Detection & Emotion Classification

This project provides two applications for face detection and emotion classification: a Gradio-based web application and a Tkinter-based desktop application.


## 📂 Model Architecture

![Model Architecture](asset/new_pipeline.png)


## 🚀 Gradio Web Application (`app.py`)

A web-based application offering advanced features for face detection and emotion classification, accessible via a browser.

Link: 
```angular2html

```

### Features

- **Face Detection**: Utilizes various YOLO models (e.g., `yolov12n-face.pt`, `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolo11n-seg.pt`) for accurate face detection, including segmentation capabilities.
- **Emotion Classification**: Employs multiple emotion classification models (e.g., `resnet18_emotion_classifier.pth`, `combine_regnetY16GF_emotion_classifier.pth`) to classify emotions into 7 categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).
- **Vision-Language Model (VLM) Validation**: Integrates Qwen3-VL to validate low-confidence emotion predictions, improving accuracy.
- **Interactive Interface**: Gradio-based UI with support for image uploads and real-time webcam processing.
- **Customizable Processing**: Adjustable confidence threshold for detection, option to blur faces or overlay emojis, and visualization of activation feature maps.
- **Model Selection**: Easily switch between different YOLO and emotion classification models.

### Usage

To run the Gradio web application:

```bash
python app.py
```

The application will typically be available at `http://localhost:8080` or a similar address.



## ⚙️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the necessary models into the `models/` directory. The application expects specific `.pt` and `.pth` files.




## 🖥️ Tkinter Desktop Application (`main.py`) [depreciated]

A desktop application providing basic face detection and emotion classification functionalities.

### Features

- **Live Face Detection**: Uses various YOLO models (e.g., `yolov12n-face.pt`, `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) for face detection.
- **Live Emotion Classification**: Employs emotion classification models (e.g., `resnet18_emotion_classifier.pth`, `efficientnet_b4_Tuned2_best.pth`) to classify emotions.
- **Interactive GUI**: Tkinter-based interface with drag-and-drop support for images.
- **Adjustable Confidence**: Set a confidence threshold for face detection.


### Usage

To run the Tkinter desktop application:

```bash
python main.py
```





