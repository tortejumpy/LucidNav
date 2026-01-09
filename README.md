# 👁️‍🗨️ LucidNav: Agentic AI Vision Assistant

> **"See through sound. Navigate with intelligence."**
> LucidNav is an agentic, multimodal AI-powered vision assistant designed to help visually impaired users understand and navigate their surroundings in real time using intelligent audio feedback.

---

## 🔍 Project Overview

LucidNav transforms live camera input into **context-aware spoken guidance** by combining computer vision, gesture recognition, facial emotion analysis, optical character recognition (OCR), and safety mechanisms. Unlike traditional object detection systems, LucidNav behaves as an **agentic AI system** — it perceives, reasons, prioritizes, and communicates meaningfully.

The system is designed to work on **consumer-grade hardware**, making it accessible, low-cost, and practical for real-world use.

---

## ✨ Key Features

* 🎯 **Real-Time Object Detection** using YOLOv8
* 📏 **Distance & Direction Estimation** (left / right / ahead)
* ✋ **Hand Gesture Recognition** (pointing, thumbs up, open hand, victory)
* 😐 **Facial Emotion Analysis** for detected persons
* 🪧 **Live OCR** to read signboards, labels, and printed text
* 🗣️ **Natural Audio Narration** with context-aware summaries
* 🧠 **Agentic Reasoning Layer** to fuse multi-modal signals
* 🆘 **Emergency SOS System** with geolocation and SMS alerts
* ⚡ **Asynchronous & Optimized Pipeline** for smooth real-time performance

---

## 🧠 How LucidNav Works (System Architecture)

1. **Vision Perception**

   * Camera feed processed in real time using OpenCV
   * YOLOv8 detects objects with confidence filtering

2. **Spatial Awareness**

   * Distance estimated using focal length and known object widths
   * Direction inferred based on object position in frame

3. **Human Interaction Understanding**

   * MediaPipe detects hand landmarks and gestures
   * DeepFace analyzes facial expressions to infer emotions

4. **Text Understanding**

   * EasyOCR extracts visible text from live frames

5. **Agentic Reasoning**

   * All signals (objects, distance, gestures, emotions, text) are fused
   * The system prioritizes relevance and safety before narration

6. **Audio Feedback & Safety**

   * Context-aware descriptions delivered via Text-to-Speech
   * Double-tap gesture triggers emergency SOS with location sharing

---

## 🧰 Tech Stack

| Category            | Technologies                 |
| ------------------- | ---------------------------- |
| Language            | Python 3.9+                  |
| Computer Vision     | OpenCV, YOLOv8 (Ultralytics) |
| Gesture Recognition | MediaPipe Hands              |
| Emotion Analysis    | DeepFace                     |
| OCR                 | EasyOCR                      |
| Speech              | pyttsx3 / plyer.tts          |
| Safety & SOS        | Twilio API, Geocoder         |
| UI                  | Tkinter / Kivy               |
| Concurrency         | Multithreading, Queues       |

---

## 📁 Project Structure

```
LucidNav/
│── main.py                  # Main application entry point
│── android_main.py          # Kivy-based Android-ready version
│── vision_assistant.py      # Core agentic logic (perception + reasoning)
│── float_yolo.py            # YOLO utilities
│── obj.py                   # Object configuration & helpers
│── ui.kv                    # Kivy UI layout
│── yolov8n.pt               # YOLOv8 model weights
│── README.md                # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/LucidNav.git
cd LucidNav
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ Note: Some features (DeepFace, EasyOCR) are optional and will gracefully disable if not installed.

### 3️⃣ Run the Application

```bash
python main.py
```

---

## 🚨 Emergency SOS Feature

* Triggered via **double-tap / double-click gesture**
* Automatically fetches user location
* Sends SOS SMS with Google Maps link to a predefined contact
* Provides spoken confirmation to the user

---

## 🎯 Use Cases

* Navigation assistance for visually impaired users
* Indoor & outdoor environment understanding
* Reading signboards and printed instructions
* Safety monitoring and emergency alerts
* Human interaction awareness (gestures & emotions)

---

## 🚀 Future Enhancements

* 📱 Mobile deployment (Android / iOS)
* 🕶️ Smart glasses or wearable integration
* 🌐 Offline-first edge AI optimization
* 🧭 Path planning and obstacle avoidance
* 🗺️ Indoor mapping and memory-based navigation

---

## 🏆 Why LucidNav Stands Out

* Goes beyond object detection into **agentic AI reasoning**
* Multimodal perception fused into meaningful narration
* Designed for **real-world accessibility and safety**
* Modular, scalable, and production-oriented architecture

---

## 👤 Author

**Harsh Pandey**
Aspiring AI / ML Engineer | Computer Vision & Agentic AI Enthusiast

---

## 📜 License

This project is licensed under the MIT License. Feel free to use, modify, and build upon it.

---

⭐ *If you found this project useful or inspiring, please consider giving it a star!*
