
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from tkinterdnd2 import DND_FILES, TkinterDnD

class FaceEmotionGUI(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Detection & Emotion Classification")
        self.geometry("1200x800")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO("models/yolov12n-face.pt")
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'] # As per USAGE.md, 7 emotions

        # Initialize emotion model (will be loaded after widgets setup)
        self.emotion_model = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.setup_widgets()
        self.load_initial_emotion_model()
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.drop_image)

    def setup_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # --- Row 0: Buttons ---
        self.load_btn = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        self.load_btn.grid(row=0, column=0, padx=5, pady=5)

        self.remove_btn = ttk.Button(control_frame, text="Remove Image", command=self.remove_image)
        self.remove_btn.grid(row=0, column=1, padx=5, pady=5)

        self.process_btn = ttk.Button(control_frame, text="Process Image", command=self.process_image, state=tk.DISABLED)
        self.process_btn.grid(row=0, column=2, padx=5, pady=5)

        # --- Row 1: Model and Confidence ---
        ttk.Label(control_frame, text="YOLO Model:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.yolo_model_var = tk.StringVar(value="yolov12n-face.pt")
        self.yolo_model_menu = ttk.Combobox(control_frame, textvariable=self.yolo_model_var,
                                            values=["yolov12n-face.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                                            width=20)
        self.yolo_model_menu.grid(row=1, column=1, padx=5, pady=5)
        self.yolo_model_menu.bind("<<ComboboxSelected>>", self.update_yolo_model)

        ttk.Label(control_frame, text="Confidence:").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.confidence_var = tk.DoubleVar(value=0.5)
        self.confidence_scale = ttk.Scale(control_frame, from_=0.1, to=1.0, variable=self.confidence_var, orient=tk.HORIZONTAL, command=self.update_confidence_label)
        self.confidence_scale.grid(row=1, column=3, padx=5, pady=5)
        self.confidence_label = ttk.Label(control_frame, text=f"{self.confidence_var.get():.2f}")
        self.confidence_label.grid(row=1, column=4, padx=5, pady=5)

        # --- Row 2: Emotion Model ---
        ttk.Label(control_frame, text="Emotion Model:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.emotion_model_var = tk.StringVar(value="efficientnet_b4_Tuned2_best.pth")
        self.emotion_model_menu = ttk.Combobox(control_frame, textvariable=self.emotion_model_var,
                                                values=["efficientnet_b4_Tuned2_best.pth", "resnet18_emotion_classifier.pth", "best_emotion_model.pth"],
                                                width=30)
        self.emotion_model_menu.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky=tk.W)
        self.emotion_model_menu.bind("<<ComboboxSelected>>", self.update_emotion_model)

        # --- Row 3: Device Info and Image Name ---
        ttk.Label(control_frame, text=f"Device: {self.device}").grid(row=3, column=0, padx=10, pady=5, sticky=tk.W)

        ttk.Label(control_frame, text="Image:").grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        self.image_name_label = ttk.Label(control_frame, text="No image loaded", foreground="gray")
        self.image_name_label.grid(row=3, column=2, columnspan=3, padx=5, pady=5, sticky=tk.W)

        # Image display frame
        image_frame = ttk.Frame(main_frame, padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        self.canvas = tk.Canvas(image_frame, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Results frame
        results_outer_frame = ttk.LabelFrame(main_frame, text="Classifier Previews", padding="10")
        results_outer_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(1, weight=1)

        self.results_canvas = tk.Canvas(results_outer_frame)
        self.results_scrollbar = ttk.Scrollbar(results_outer_frame, orient="vertical", command=self.results_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.results_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(
                scrollregion=self.results_canvas.bbox("all")
            )
        )

        self.results_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=self.results_scrollbar.set)

        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scrollbar.pack(side="right", fill="y")


        # Save button
        self.save_btn = ttk.Button(main_frame, text="Save Processed Image", command=self.save_image, state=tk.DISABLED)
        self.save_btn.grid(row=2, column=1, sticky=(tk.E), padx=10, pady=10)


    def load_emotion_model(self, model_name, path):
        try:
            num_classes = 7
            model = None

            # Decide which model architecture to build
            if "resnet" in model_name:
                model = models.resnet18(weights=None)
                num_ftrs = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(num_ftrs, num_classes)
                )
            elif "efficientnet" in model_name:
                model = models.efficientnet_b4(weights=None)
                model.classifier = nn.Sequential(
                    nn.Linear(model.classifier[1].in_features, 512),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes)
                )
            else:
                raise ValueError("Unsupported model type in filename. Use 'resnet' or 'efficientnet'.")

            # Load the state dictionary
            state_dict = torch.load(path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            model.to(self.device)
            model.eval()
            return model

        except FileNotFoundError:
            messagebox.showerror("Error", f"Emotion model not found at {path}")
            self.quit()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load emotion model: {e}")
            self.quit()

    def unnormalize(self, tensor):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        tensor = tensor.clone().detach().cpu().numpy().transpose((1, 2, 0))
        tensor = std * tensor + mean
        tensor = np.clip(tensor, 0, 1)
        return Image.fromarray((tensor * 255).astype(np.uint8))

    def update_confidence_label(self, value):
        self.confidence_label.config(text=f"{float(value):.2f}")

    def load_initial_emotion_model(self):
        """Load the default emotion model on startup"""
        model_name = self.emotion_model_var.get()
        model_path = f"models/{model_name}"
        self.emotion_model = self.load_emotion_model(model_name, model_path)

    def update_yolo_model(self, event):
        model_name = self.yolo_model_var.get()
        model_path = f"models/{model_name}"
        try:
            self.yolo_model = YOLO(model_path)
            messagebox.showinfo("Info", f"YOLO model updated to {model_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")

    def update_emotion_model(self, event):
        """Load selected emotion model"""
        model_name = self.emotion_model_var.get()
        model_path = f"models/{model_name}"
        try:
            self.emotion_model = self.load_emotion_model(model_name, model_path)
            messagebox.showinfo("Info", f"Emotion model updated to {model_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load emotion model: {e}")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")])
        if file_path:
            self.original_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image)
            self.process_btn['state'] = tk.NORMAL
            self.save_btn['state'] = tk.DISABLED
            self.clear_results()
            # Update image name label
            import os
            self.image_name_label.config(text=os.path.basename(file_path), foreground="black")

    def drop_image(self, event):
        file_path = event.data
        if file_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            self.original_image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            self.display_image(self.original_image)
            self.process_btn['state'] = tk.NORMAL
            self.save_btn['state'] = tk.DISABLED
            self.clear_results()
            # Update image name label
            import os
            self.image_name_label.config(text=os.path.basename(file_path), foreground="black")

    def remove_image(self):
        self.canvas.delete("all")
        self.clear_results()
        self.save_btn['state'] = tk.DISABLED
        self.process_btn['state'] = tk.DISABLED
        self.image_name_label.config(text="No image loaded", foreground="gray")
        if hasattr(self, 'original_image'):
            del self.original_image
        if hasattr(self, 'processed_image'):
            del self.processed_image

    def clear_results(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

    def process_image(self):
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        self.processed_image = self.original_image.copy()
        self.clear_results()

        # Face detection
        results = self.yolo_model(self.original_image, conf=self.confidence_var.get())

        face_count = 0
        for i, r in enumerate(results):
            for box in r.boxes:
                face_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = self.original_image[y1:y2, x1:x2]

                # Emotion classification
                pil_face = Image.fromarray(face)
                input_tensor = self.transform(pil_face)
                
                with torch.no_grad():
                    output = self.emotion_model(input_tensor.unsqueeze(0).to(self.device))
                    _, predicted = torch.max(output, 1)
                    emotion = self.emotion_labels[predicted.item()]

                # Create preview image
                preview_img = self.unnormalize(input_tensor)
                preview_photo = ImageTk.PhotoImage(preview_img.resize((112, 112))) # Half size of 224

                # Create a frame for this result
                result_entry_frame = ttk.Frame(self.scrollable_frame, padding=5)
                result_entry_frame.pack(fill='x', expand=True, pady=5)

                preview_label = ttk.Label(result_entry_frame, image=preview_photo)
                preview_label.image = preview_photo # Keep a reference!
                preview_label.pack(side='left', padx=5)

                details_text = f"Face {face_count}: {emotion}\n" \
                               f"Confidence: {box.conf[0]:.2f}"
                details_label = ttk.Label(result_entry_frame, text=details_text)
                details_label.pack(side='left', padx=5, fill='x', expand=True)

                # Draw bounding box and label on main image
                cv2.rectangle(self.processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.processed_image, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if face_count == 0:
            no_faces_label = ttk.Label(self.scrollable_frame, text="No faces detected.")
            no_faces_label.pack()

        self.display_image(self.processed_image)
        self.save_btn['state'] = tk.NORMAL


    def display_image(self, img):
        img_pil = Image.fromarray(img)
        img_pil.thumbnail((self.canvas.winfo_width(), self.canvas.winfo_height()))
        self.photo = ImageTk.PhotoImage(image=img_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def save_image(self):
        if hasattr(self, 'processed_image'):
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(self.processed_image, cv2.COLOR_RGB2BGR))
                messagebox.showinfo("Success", f"Image saved to {save_path}")

if __name__ == "__main__":
    app = FaceEmotionGUI()
    app.mainloop()
