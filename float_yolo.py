import torch
import cv2
import pyttsx3
from ultralytics import YOLO
import time
import numpy as np
import os
import speech_recognition as sr
import threading
import queue
from datetime import datetime
from collections import defaultdict
import mediapipe as mp
from scipy.spatial import distance as dist

# Voice setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('voice', engine.getProperty('voices')[1].id)

def speak(text, print_text=True):
    """Speak text and optionally print to terminal"""
    if print_text:
        print(f"[Assistant]: {text}")
    engine.say(text)
    engine.runAndWait()

class VisionAssistant:
    def __init__(self):
        self.last_announce = time.time()
        self.last_objects = []
        self.cooldown = 15  # Seconds between auto announcements
        self.gesture_history = []
        self.gesture_cooldown = 2  # Seconds between gesture recognition
        self.suspicious_activity_start = None
        self.suspicious_threshold = 3  # Seconds to consider activity suspicious
        self.face_cover_duration = 0
        self.hand_movement_history = []
        self.movement_threshold = 0.1  # Threshold for rapid movement detection
        self.last_gesture_time = 0
        self.last_suspicious_alert = 0
        self.suspicious_alert_cooldown = 10  # Seconds between suspicious activity alerts

        # Initialize MediaPipe solutions
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def describe_environment(self, objects):
        """Generate concise environment description"""
        if not objects:
            return "The space appears clear."

        obj_counts = defaultdict(int)
        for obj in objects:
            obj_counts[obj[0]] += 1

        closest = min(objects, key=lambda x: x[2])
        dir_map = {'left': 'to your left', 'right': 'to your right', 'center': 'ahead'}

        items = []
        for obj, count in obj_counts.items():
            items.append(f"{count} {obj}" + ("s" if count > 1 else ""))

        description = f"I see {', '.join(items)}. "
        description += f"Closest is {closest[0]} {dir_map.get(closest[1], 'nearby')}, {closest[2]:.1f}m away."

        return description

    def recognize_gesture(self, hand_landmarks):
        """Recognize hand gestures"""
        if not hand_landmarks:
            return None
            
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        thumb_index_dist = dist.euclidean((thumb_tip.x, thumb_tip.y), (index_tip.x, index_tip.y))
        thumb_middle_dist = dist.euclidean((thumb_tip.x, thumb_tip.y), (middle_tip.x, middle_tip.y))
        
        if thumb_index_dist < 0.05 and thumb_middle_dist > 0.1:
            return "pointing"
        elif thumb_index_dist < 0.05 and thumb_middle_dist < 0.05:
            return "ok"
        elif index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y:
            return "victory"
        elif index_tip.y < wrist.y and middle_tip.y < wrist.y and thumb_tip.y < wrist.y:
            return "raised_hand"
        return None

    def detect_suspicious_activity(self, hand_landmarks_list, face_landmarks=None):
        """Detect suspicious activities"""
        suspicious_activities = []
        current_time = time.time()
        
        # Track hand movements
        current_positions = []
        for hand in hand_landmarks_list:
            wrist = hand.landmark[self.mp_hands.HandLandmark.WRIST]
            current_positions.append((wrist.x, wrist.y))
        self.hand_movement_history.append((current_time, current_positions))
        self.hand_movement_history = [x for x in self.hand_movement_history if current_time - x[0] <= 1.0]
        
        if len(self.hand_movement_history) > 5:
            movement_distance = 0
            for i in range(1, len(self.hand_movement_history)):
                prev_pos = self.hand_movement_history[i-1][1]
                curr_pos = self.hand_movement_history[i][1]
                if len(prev_pos) == len(curr_pos):
                    for j in range(len(prev_pos)):
                        movement_distance += dist.euclidean(prev_pos[j], curr_pos[j])
            if movement_distance > self.movement_threshold:
                suspicious_activities.append("rapid_hand_movements")
        
        if face_landmarks:
            nose_tip = face_landmarks[0].landmark[4]
            for hand in hand_landmarks_list:
                hand_center_x = np.mean([lm.x for lm in hand.landmark])
                hand_center_y = np.mean([lm.y for lm in hand.landmark])
                if dist.euclidean((hand_center_x, hand_center_y), (nose_tip.x, nose_tip.y)) < 0.2:
                    self.face_cover_duration += 0.1
                    if self.face_cover_duration > 1.5:
                        suspicious_activities.append("face_covering")
                else:
                    self.face_cover_duration = max(0, self.face_cover_duration - 0.05)
        
        if len(hand_landmarks_list) == 2:
            wrist1 = hand_landmarks_list[0].landmark[self.mp_hands.HandLandmark.WRIST]
            wrist2 = hand_landmarks_list[1].landmark[self.mp_hands.HandLandmark.WRIST]
            if dist.euclidean((wrist1.x, wrist1.y), (wrist2.x, wrist2.y)) < 0.15:
                suspicious_activities.append("defensive_posture")
        
        if suspicious_activities:
            if self.suspicious_activity_start is None:
                self.suspicious_activity_start = current_time
            elif current_time - self.suspicious_activity_start > self.suspicious_threshold:
                if current_time - self.last_suspicious_alert > self.suspicious_alert_cooldown:
                    self.last_suspicious_alert = current_time
                    return suspicious_activities
        else:
            self.suspicious_activity_start = None
        
        return None

    def process_frame(self, frame):
        """Process frame for gestures and suspicious activities"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        current_gestures = []
        suspicious_activities = []
        
        if hand_results.multi_hand_landmarks:
            face_landmarks_for_detection = face_results.multi_face_landmarks if face_results.multi_face_landmarks else None
            suspicious_activities = self.detect_suspicious_activity(
                hand_results.multi_hand_landmarks, 
                face_landmarks_for_detection
            )
            
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))
                
                gesture = self.recognize_gesture(hand_landmarks)
                if gesture and time.time() - self.last_gesture_time > self.gesture_cooldown:
                    current_gestures.append(gesture)
                    self.last_gesture_time = time.time()
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(200, 200, 200), thickness=1, circle_radius=1)
                )
        
        return frame, current_gestures, suspicious_activities

def listen_loop(recognizer, mic, cmd_queue, stop_event):
    """Listen for voice commands"""
    while not stop_event.is_set():
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
                command = recognizer.recognize_google(audio).lower()
                print(f"[User]: {command}")
                cmd_queue.put(command)
            except Exception:
                pass

def check_hardware_support():
    """Check available hardware acceleration"""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

def vision_system():
    # Initialize with best available hardware
    device = check_hardware_support()
    
    try:
        model = YOLO("yolov8m.pt").to(device)
        speak(f"Model loaded on {device.upper()}")
    except Exception as e:
        speak(f"Failed to load model: {str(e)}")
        return
    
    assistant = VisionAssistant()
    focal_length = 910
    known_widths = {
        'person': 0.5, 'chair': 0.45, 'laptop': 0.3,
        'book': 0.2, 'phone': 0.075, 'cup': 0.09,
        'hand': 0.1
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        speak("Camera unavailable.")
        return

    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    cmd_queue = queue.Queue()
    stop_event = threading.Event()
    listener = threading.Thread(target=listen_loop, args=(recognizer, mic, cmd_queue, stop_event))
    listener.daemon = True
    listener.start()

    speak(f"Vision system ready (running on {device.upper()})")
    prev_frame = None
    last_motion_alert = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, current_gestures, suspicious_activities = assistant.process_frame(frame)
        
        for gesture in current_gestures:
            speak(f"Detected {gesture} gesture")
        
        if suspicious_activities:
            speak(f"Warning: Detected suspicious activity - {' and '.join(suspicious_activities)}")
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if prev_frame is not None:
            if np.sum(cv2.absdiff(prev_frame, gray)) > 10000:
                if time.time() - last_motion_alert > 30:
                    speak("Movement detected.")
                    last_motion_alert = time.time()
        prev_frame = gray

        results = model.track(frame, persist=True, verbose=False, device=device)
        detected_objects = []

        if results[0].boxes.id is not None:
            for box, track_id, cls, conf in zip(results[0].boxes.xywh.cpu(),
                                             results[0].boxes.id.int().cpu().tolist(),
                                             results[0].boxes.cls.int().cpu().tolist(),
                                             results[0].boxes.conf.cpu().tolist()):
                obj_name = model.names[cls]
                if obj_name not in known_widths or conf < 0.5:
                    continue

                x, y, w, h = box.tolist()
                distance = (known_widths[obj_name] * focal_length) / w
                direction = "left" if x < frame.shape[1]/3 else "right" if x > 2*frame.shape[1]/3 else "center"
                detected_objects.append((obj_name, direction, float(distance), float(conf)))

                cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{obj_name} {distance:.1f}m", (int(x-w/2), int(y-h/2 - 5)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        overlay_y = 20
        for obj in detected_objects:
            label = f"{obj[0]}: {obj[2]:.1f}m"
            cv2.putText(frame, label, (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (0, 255, 255), 2)
            overlay_y += 25
            
        if current_gestures:
            cv2.putText(frame, f"Gestures: {', '.join(current_gestures)}", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)
        
        if suspicious_activities:
            cv2.putText(frame, f"Suspicious: {', '.join(suspicious_activities)}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)

        current_time = time.time()
        if current_time - assistant.last_announce > assistant.cooldown and detected_objects:
            description = assistant.describe_environment(detected_objects)
            speak(description)
            assistant.last_announce = current_time
            assistant.last_objects = detected_objects

        while not cmd_queue.empty():
            cmd = cmd_queue.get().lower()
            if "exit" in cmd or "stop" in cmd:
                speak("Shutting down.")
                stop_event.set()
                cap.release()
                cv2.destroyAllWindows()
                return

            elif "what" in cmd and ("see" in cmd or "there" in cmd):
                speak(assistant.describe_environment(detected_objects or assistant.last_objects))

            elif "how many" in cmd:
                obj_counts = defaultdict(int)
                for obj in (detected_objects or assistant.last_objects):
                    obj_counts[obj[0]] += 1
                speak(f"Count: {', '.join(f'{v} {k}' for k,v in obj_counts.items())}")
                
            elif "gesture" in cmd or "hand" in cmd:
                if current_gestures:
                    speak(f"I see {', '.join(current_gestures)} gestures.")
                else:
                    speak("No recent gestures detected.")
            
            elif "suspicious" in cmd or "danger" in cmd:
                if suspicious_activities:
                    speak(f"Current suspicious activities: {', '.join(suspicious_activities)}")
                else:
                    speak("No suspicious activities detected currently.")

            else:
                speak("I can describe objects, count them, recognize gestures, or monitor for suspicious activities.")

        try:
            cv2.imshow("Enhanced Vision System", frame)
            if cv2.waitKey(1) == ord('q'):
                stop_event.set()
                break
        except:
            stop_event.set()
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    listener.join()

if __name__ == "__main__":
    vision_system()