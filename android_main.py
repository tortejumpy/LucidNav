import cv2
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics (YOLO) library not found. Object detection will be disabled.")
import os
from plyer import tts
import threading
import queue
import time
from collections import defaultdict
import mediapipe as mp
import numpy as np
from kivy.app import App

# --- Optional Imports with Graceful Fallbacks ---
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace library not found. Facial analysis will be disabled.")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Warning: EasyOCR library not found. Text recognition (OCR) will be disabled.")
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# --- Vision Assistant Logic (adapted from main.py) ---
class VisionAssistant:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_gesture_time = 0
        self.gesture_cooldown = 3
        self.last_face_analysis_time = 0
        self.face_analysis_cooldown = 5
        if EASYOCR_AVAILABLE:
            print("Initializing EasyOCR reader...")
            self.ocr_reader = easyocr.Reader(['en'])
        else:
            self.ocr_reader = None

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
        if not DEEPFACE_AVAILABLE:
            return {}
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
                except Exception as e:
                    print(f"DeepFace analysis failed: {e}")
                    pass
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
    
    def perform_ocr_on_frame(self, frame):
        if not self.ocr_reader:
            return None
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

# Load the kv file
Builder.load_file('ui.kv')

class VoiceWaitScreen(Screen):
    pass

class MainMenuScreen(Screen):
    def start_system(self):
        print("System starting...")
        self.manager.current = 'vision_view'

class VisionScreen(Screen):
    def on_enter(self, *args):
        self.app = App.get_running_app()
        self.vision_assistant = self.app.vision_assistant
        if self.app.model:
            self.model = self.app.model
        else:
            self.model = None
        self.KNOWN_WIDTHS = {"person": 0.5, "car": 1.8, "bus": 3.0, "truck": 3.5, "stop sign": 0.75}
        self.FOCAL_LENGTH = 840
        self.last_speech_time = time.time()
        self.speech_cooldown = 5
        self.detected_emotions = {}
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Cannot open camera")
            return
        self.video_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def on_leave(self, *args):
        Clock.unschedule(self.update)
        if self.capture.isOpened():
            self.capture.release()

    def estimate_distance(self, object_name, pixel_width):
        if object_name in self.KNOWN_WIDTHS: return (self.KNOWN_WIDTHS[object_name] * self.FOCAL_LENGTH) / pixel_width
        return None

    def get_direction(self, x_center):
        segment = self.video_width / 3
        if x_center < segment: return "to your left"
        elif x_center > 2 * segment: return "to your right"
        return "ahead of you"

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret: return

        annotated_frame, gestures = self.vision_assistant.process_frame_for_gestures(frame.copy())
        if self.model:
            results = self.model(frame, verbose=False)
        else:
            results = []
        
        CONFIDENCE_THRESHOLD = 0.5
        detections_for_processing = []
        all_detected_names = []

        if self.model and results:
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
            ocr_text = self.vision_assistant.perform_ocr_on_frame(frame)
            if ocr_text:
                if description: description += f". I also see the text: {ocr_text}"
                else: description = f"I see the text: {ocr_text}"

            if description:
                self.app.speech_queue.put(description)
            self.last_speech_time = time.time()
        
        buf1 = cv2.flip(annotated_frame, 0)
        buf = buf1.tostring()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_view.texture = image_texture

class ChalChitraApp(App):
    def build(self):
        print("Initializing VisionAssistant...")
        self.vision_assistant = VisionAssistant()
        if ULTRALYTICS_AVAILABLE:
            print("Setting up models...")
            model_path = 'yolov8n.pt'
            if not os.path.exists(model_path):
                print(f"YOLO model not found at {model_path}. Object detection will be disabled.")
                self.model = None
            else:
                self.model = YOLO(model_path)
        else:
            self.model = None
        self.speech_queue = queue.Queue()
        sm = ScreenManager()
        sm.add_widget(VoiceWaitScreen(name='wait'))
        sm.add_widget(MainMenuScreen(name='main'))
        sm.add_widget(VisionScreen(name='vision_view'))
        sm.current = 'main'
        return sm
    
    def on_start(self):
        self.speech_thread = threading.Thread(target=self.speech_worker)
        self.speech_thread.daemon = True
        self.speech_thread.start()

    def speech_worker(self):
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                if text is None:
                    self.speech_queue.task_done()
                    break
                tts.speak(text)
                self.speech_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in speech worker: {e}")

    def on_stop(self):
        if hasattr(self, 'speech_queue'):
            self.speech_queue.put(None)
        if hasattr(self, 'speech_thread') and self.speech_thread.is_alive():
            self.speech_thread.join()

if __name__ == '__main__':
    ChalChitraApp().run()
