# 👁️‍🗨️ Chal-Chitra: Agentic AI-Based Vision Assistant for the Blind

> "See the unseen, navigate the unknown."  
> A powerful agentic AI application that empowers visually impaired individuals to understand their surroundings using real-time computer vision, gesture recognition, OCR, and natural narration.

---

## 🔍 Project Overview

ChalChitra is an AI-powered assistive tool designed to interpret visual scenes for blind or visually impaired individuals. It provides auditory feedback for surrounding objects, text on signboards, human gestures, and emotional cues — all in real time.

Whether navigating indoors or outdoors, LucidNav acts as a virtual guide capable of:
- Recognizing gestures (like “thumbs up”, “peace”, etc.)
- Identifying objects with estimated distances and directions
- Reading live text (e.g., signs, warnings, door labels)
- Detecting human emotions
- Performing SOS alerts on double-tap gestures

---

## 🧠 Core Features

- 🎯 Real-Time Object Detection (YOLOv8)
- ✋ Gesture Recognition (MediaPipe Hands)
- 😐 Facial Emotion Analysis (DeepFace)
- 🪧 Live OCR from Camera Feed (EasyOCR)
- 🗣️ Natural Speech Feedback (pyttsx3)
- 🆘 Emergency SOS + Geolocation Sharing (Twilio + Geocoder)
- 🖱️ Double-Click Trigger for Safety Alerts
- 🔊 Works offline for most features (speech, vision, gesture)

---

## 🧰 Tech Stack

| Layer | Tools/Frameworks |
|------|------------------|
| 👁️ Vision | OpenCV, MediaPipe, YOLOv8 |
| 💬 Speech | pyttsx3 |
| 📖 OCR | EasyOCR |
| 😊 Emotion | DeepFace |
| 🌐 Location/SOS | Geocoder, Twilio API |
| 🧠 Agentic AI | Threaded logic + context fusion |
| 🎛️ GUI | Tkinter, PIL |
| 🐍 Language | Python 3.9 |

---

## 🖥️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/LucidNav.git
cd LucidNav/backend
