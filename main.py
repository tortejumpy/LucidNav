import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import pyttsx3
import threading
import queue
import time
from collections import defaultdict
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import easyocr
import geocoder
from twilio.rest import Client

# --- Text-to-Speech Engine ---
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Speech Error: {e}")

# --- Vision Assistant Logic --- #
class VisionAssistant:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_gesture_time = 0
        self.gesture_cooldown = 3
        self.last_face_analysis_time = 0
        self.face_analysis_cooldown = 5

        # Initialize EasyOCR reader
        print("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(['en']) # You can add other languages if needed, e.g., ['en', 'fr']

    def recognize_gesture(self, hand_landmarks):
        if not hand_landmarks: return None
        landmarks = hand_landmarks.landmark
        fingers_up = [
            (landmarks[4].x < landmarks[3].x) if (landmarks[4].x < landmarks[17].x) else (landmarks[4].x > landmarks[3].x),
            landmarks[8].y < landmarks[6].y, landmarks[12].y < landmarks[10].y,
            landmarks[16].y < landmarks[14].y, landmarks[20].y < landmarks[18].y
        ]
        num_fingers_up = sum(fingers_up)
        if num_fingers_up == 0: return "fist"
        if num_fingers_up == 1 and fingers_up[1]: return "pointing"
        if num_fingers_up == 2 and fingers_up[1] and fingers_up[2]: return "victory or peace"
        if num_fingers_up == 5: return "open hand"
        if num_fingers_up == 1 and fingers_up[0]: return "thumbs up"
        return None

    def process_frame_for_gestures(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        gestures = []
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                gesture = self.recognize_gesture(hand_landmarks)
                if gesture and (time.time() - self.last_gesture_time > self.gesture_cooldown):
                    gestures.append(gesture)
                    self.last_gesture_time = time.time()
        return frame, gestures

    def analyze_facial_expressions(self, frame, detections):
        emotions = {}
        if time.time() - self.last_face_analysis_time < self.face_analysis_cooldown: return emotions
        for i, (x1, y1, x2, y2, name) in enumerate(detections):
            if name == 'person':
                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0: continue
                try:
                    analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    if isinstance(analysis, list): analysis = analysis[0]
                    dominant_emotion = analysis.get('dominant_emotion')
                    if dominant_emotion: emotions[i] = dominant_emotion
                except Exception: pass
        if emotions: self.last_face_analysis_time = time.time()
        return emotions

    def describe_environment(self, all_detected_names, objects_with_distance, emotions, gestures):
        if not all_detected_names and not gestures: return None
        description_parts = []
        if all_detected_names:
            dist_pool = list(objects_with_distance)
            obj_counts = defaultdict(int)
            for name in all_detected_names: obj_counts[name] += 1
            items = []
            for name, count in obj_counts.items():
                dist_entries = [d for d in dist_pool if d[0] == name]
                for entry in dist_entries:
                    items.append(f"a {name} {entry[1]} at {entry[2]:.1f} meters")
                    dist_pool.remove(entry)
                remaining_count = count - len(dist_entries)
                if remaining_count > 0: items.append(f"{remaining_count} {name}{'s' if remaining_count > 1 else ''}")
            if items: description_parts.append(f"I see {', '.join(items)}")
        if emotions:
            emotion_list = [f"a person who appears {emo}" for emo in emotions.values()]
            description_parts.append(f"I can also see {', '.join(emotion_list)}")
        if gestures: description_parts.append(f"I also detect a {', '.join(gestures)} gesture")
        return ". ".join(description_parts) + "." if description_parts else ""

# --- Main Application --- #
class VisionApp:
    def __init__(self, root):
        print("Initializing VisionApp...")
        self.root = root
        self.root.title("Vision IRL - Advanced Assistant")
        self.root.geometry("1280x720")
        print("Creating VisionAssistant...")
        self.vision_assistant = VisionAssistant()
        print("Setting up models...")
        try:
            self.setup_models()
            print("Models setup complete.")
        except Exception as e:
            print(f"Error during model setup: {e}")
            messagebox.showerror("Initialization Error", f"Failed to setup models: {e}")
            self.root.destroy()
            return
        print("Setting up video capture...")
        try:
            self.setup_video_capture()
            print("Video capture setup complete.")
        except Exception as e:
            print(f"Error during video capture setup: {e}")
            messagebox.showerror("Initialization Error", f"Failed to setup video capture: {e}")
            self.root.destroy()
            return
        print("Setting up speech thread...")
        self.setup_speech_thread()

        # Initialize EasyOCR reader
        print("Initializing EasyOCR reader...")
        self.ocr_reader = easyocr.Reader(['en'])

        print("Setting up UI...")
        self.setup_ui()
        self.KNOWN_WIDTHS = {"person": 0.5, "car": 1.8, "bus": 3.0, "truck": 3.5, "stop sign": 0.75}
        self.FOCAL_LENGTH = 840
        self.last_speech_time = time.time()
        self.speech_cooldown = 5
        self.detected_emotions = {}
        print("Starting frame updates...")
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        print("VisionApp Initialized.")

        # Initialize last click time for double-tap detection
        self.last_click_time = 0
        self.double_tap_interval = 300 # milliseconds
        self.root.bind("<Button-1>", self.on_click) # Bind left-click event

    def setup_models(self):
        try:
            model_path = 'yolov8n.pt'
            if not os.path.exists(model_path): raise FileNotFoundError(f"YOLO model not found at {model_path}.")
            self.model = YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load YOLO model: {e}")
            self.root.destroy()

    def setup_video_capture(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): raise IOError("Cannot open webcam")
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.root.geometry(f"{self.video_width}x{self.video_height + 80}") # Increased height for new button
        except Exception as e:
            messagebox.showerror("Webcam Error", f"Failed to initialize webcam: {e}")
            self.root.destroy()

    def setup_ui(self):
        self.main_frame = tk.Frame(self.root, bg="#2c3e50")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.video_label = tk.Label(self.main_frame)
        self.video_label.pack(pady=10, padx=10)
        
        self.button_frame = tk.Frame(self.main_frame, bg="#2c3e50")
        self.button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.status_bar = tk.Label(self.main_frame, text="Initializing...", bd=1, relief=tk.SUNKEN, anchor=tk.W, fg="white", bg="#34495e")
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_speech_thread(self):
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self.speech_worker, daemon=True)
        self.speech_thread.start()

    def speech_worker(self):
        while True:
            text = self.speech_queue.get()
            if text is None: break
            speak(text)
            self.speech_queue.task_done()

    def update_status_bar(self, text):
        self.status_bar.config(text=text)

    def estimate_distance(self, object_name, pixel_width):
        if object_name in self.KNOWN_WIDTHS: return (self.KNOWN_WIDTHS[object_name] * self.FOCAL_LENGTH) / pixel_width
        return None

    def get_direction(self, x_center):
        segment = self.video_width / 3
        if x_center < segment: return "to your left"
        elif x_center > 2 * segment: return "to your right"
        return "ahead of you"

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        annotated_frame, gestures = self.vision_assistant.process_frame_for_gestures(frame.copy())

        results = self.model(frame, verbose=False)
        
        CONFIDENCE_THRESHOLD = 0.5
        detections_for_processing = []
        all_detected_names = []

        for box in results[0].boxes:
            if box.conf[0] > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = [int(val) for val in box.xyxy[0]]
                object_name = self.model.names[int(box.cls[0])]
                detections_for_processing.append((x1, y1, x2, y2, object_name))
                all_detected_names.append(object_name)

        emotions = self.vision_assistant.analyze_facial_expressions(frame, detections_for_processing)
        self.detected_emotions.update(emotions)

        objects_with_distance = []
        for i, (x1, y1, x2, y2, name) in enumerate(detections_for_processing):
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = name
            distance = self.estimate_distance(name, x2 - x1)
            if distance:
                direction = self.get_direction((x1 + x2) / 2)
                objects_with_distance.append((name, direction, distance))
                label += f" ({distance:.1f}m)"
            
            if name == 'person' and i in self.detected_emotions:
                label += f" ({self.detected_emotions[i]})"
            
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if time.time() - self.last_speech_time > self.speech_cooldown:
            description = self.vision_assistant.describe_environment(all_detected_names, objects_with_distance, self.detected_emotions, gestures)

            # --- OCR Integration ---
            print("Performing continuous OCR...")
            ocr_text = self.perform_ocr_on_frame(frame) # New method to encapsulate OCR
            if ocr_text:
                if description: description += f". I also see the text: {ocr_text}"
                else: description = f"I see the text: {ocr_text}"
            # --- End OCR Integration ---

            if description:
                self.speech_queue.put(description)
            self.last_speech_time = time.time()
        
        # Update status bar with detection count
        self.update_status_bar(f"Detections: {len(detections_for_processing)}")

        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def perform_ocr_on_frame(self, frame):
        """Performs OCR on a given frame and returns the detected text."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.ocr_reader.readtext(rgb_frame)
            
            detected_texts = [text for (bbox, text, prob) in results]
            if detected_texts:
                return " ".join(detected_texts)
            return None
        except Exception as e:
            print(f"Error during continuous OCR: {e}")
            return None

    def on_closing(self):
        self.speech_queue.put(None)
        if hasattr(self, 'cap') and self.cap.isOpened(): self.cap.release()
        self.root.destroy()

    def on_click(self, event):
        current_time = event.time
        if current_time - self.last_click_time < self.double_tap_interval:
            print("Double-tap detected!")
            self.speech_queue.put("Double tap detected. Initiating emergency SOS. Please wait.")
            threading.Thread(target=self.send_sos_message, daemon=True).start()
        self.last_click_time = current_time

    def send_sos_message(self):
        # --- Twilio Configuration (REPLACE WITH YOUR ACTUAL CREDENTIALS) ---
        # You can get these from your Twilio console after signing up for a free account
        # https://www.twilio.com/try-twilio
        ACCOUNT_SID = 'AC557f8086eff037dcf66838d884c96b54' # Your Account SID
        AUTH_TOKEN = '49fb59789186b1159da2502e09caca65' # Your Auth Token
        TWILIO_PHONE_NUMBER = '+16812339135' # Your Twilio Phone Number (e.g., +1XXXXXXXXXX)
        FAMILY_PHONE_NUMBER = '+918660182267' # The phone number of your family member (e.g., +1XXXXXXXXXX)
        # ------------------------------------------------------------------

        client = Client(ACCOUNT_SID, AUTH_TOKEN)

        try:
            self.update_status_bar("Getting current location...")
            g = geocoder.ip('me')
            if g.ok and g.latlng:
                latitude, longitude = g.latlng
                location_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
                message_body = f"Emergency! My current location is: {location_url} (Latitude: {latitude}, Longitude: {longitude})"
                
                self.update_status_bar("Sending SOS message...")
                message = client.messages.create(
                    to=FAMILY_PHONE_NUMBER,
                    from_=TWILIO_PHONE_NUMBER,
                    body=message_body
                )
                print(f"SOS message sent! SID: {message.sid}")
                self.speech_queue.put("Emergency SOS message sent successfully.")
                self.update_status_bar("SOS message sent!")
            else:
                error_msg = "Could not determine current location. SOS message not sent."
                print(error_msg)
                self.speech_queue.put(error_msg)
                self.update_status_bar("SOS Failed: No Location.")
        except Exception as e:
            error_msg = f"Failed to send SOS message: {e}"
            print(error_msg)
            self.speech_queue.put(error_msg)
            self.update_status_bar("SOS Failed.")

def main():
    root = tk.Tk()
    try:
        app = VisionApp(root)
        root.mainloop()
    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        messagebox.showerror("Application Error", f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()