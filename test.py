import cv2
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, AutoModelForCausalLM
import onnxruntime as ort
import warnings
import torch

# Suppress warnings
warnings.filterwarnings('ignore')

class SnapdragonObjectAnalyzer:
    def __init__(self):
        # Initialize ONNX runtime for NPU acceleration
        self.providers = ['DmlExecutionProvider']  # For Qualcomm NPU
        self.session_options = ort.SessionOptions()
        
        # YOLO model configuration
        self.model_path = "yolov8n.onnx"
        self._download_yolo_model()
        
        # Create ONNX runtime session
        self.ort_session = ort.InferenceSession(
            self.model_path,
            providers=self.providers,
            sess_options=self.session_options
        )
        
        # Load CPU-optimized Gemma-2B
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
        self.llm = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2b",
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",  # Force CPU usage
            use_cache=True
        )
        
        # Complete YOLO class names (80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Font setup
        try:
            self.font = ImageFont.truetype("arial.ttf", 14)
        except:
            self.font = ImageFont.load_default()
        
        # Camera setup with DirectShow
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Timing control
        self.last_processed_time = 0
        self.process_interval = 3  # seconds
        self.previous_explanations = {}

    def _download_yolo_model(self):
        """Download YOLO ONNX model if not already present"""
        import os
        import urllib.request
        
        if not os.path.exists(self.model_path):
            print("Downloading YOLOv8n ONNX model...")
            url = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.onnx"
            urllib.request.urlretrieve(url, self.model_path)
            print("Download complete")

    def generate_explanation(self, object_name):
        """Generate explanation with caching (CPU-only version)"""
        if object_name in self.previous_explanations:
            return self.previous_explanations[object_name]
        
        prompt = f"Explain {object_name} simply in 1 sentence:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.llm.generate(**inputs, max_new_tokens=50)
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        explanation = explanation.replace(prompt, "").strip()
        
        self.previous_explanations[object_name] = explanation
        return explanation

    def process_frame(self, frame):
        """Process frame with NPU-accelerated YOLO"""
        # Preprocess
        img = cv2.resize(frame, (640, 640))
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        
        # NPU-accelerated inference
        outputs = self.ort_session.run(None, {'images': img})
        
        # Get predictions
        predictions = np.squeeze(outputs[0])
        return self._draw_results(frame, predictions)

    def _draw_results(self, frame, predictions):
        """Draw bounding boxes and labels with proper class handling"""
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        explanations = []
        
        # Check if predictions is 2D array
        if len(predictions.shape) == 1:
            predictions = np.expand_dims(predictions, axis=0)
        
        for pred in predictions:
            if pred[4] < 0.5:  # Confidence threshold
                continue
                
            # Get class ID safely
            class_id = int(np.argmax(pred[5:85]))  # Only first 80 classes
            if class_id >= len(self.class_names):
                continue
                
            label = self.class_names[class_id]
            
            # Convert box coordinates to image size
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = pred[:4]
            x1 = int(x1 * w / 640)
            y1 = int(y1 * h / 640)
            x2 = int(x2 * w / 640)
            y2 = int(y2 * h / 640)
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1 - 20), 
                     f"{label} {pred[4]:.2f}", 
                     fill="red", 
                     font=self.font)
            
            # Generate explanation
            explanation = self.generate_explanation(label)
            explanations.append((label, explanation, [x1, y1, x2, y2], pred[4]))
            
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), explanations

    def run(self):
        """Main processing loop"""
        print("Live NPU-accelerated object detection running. Press 'q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            current_time = time.time()
            
            # Process at intervals
            if current_time - self.last_processed_time >= self.process_interval:
                processed_frame, explanations = self.process_frame(frame)
                self.last_processed_time = current_time
                
                if explanations:
                    print("\n=== Object Explanations ===")
                    for obj, exp, _, _ in explanations:
                        print(f"{obj}: {exp}")
            else:
                processed_frame = frame
                
            # Display
            cv2.imshow('Snapdragon Object Analyzer', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    analyzer = SnapdragonObjectAnalyzer()
    analyzer.run()