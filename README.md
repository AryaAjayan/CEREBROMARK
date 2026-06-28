
<div align="center">

# 🧠 CerbroMark

### Next-Generation AI-Powered Facial Recognition Attendance System

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-FF6B6B?style=for-the-badge)](https://github.com/deepinsight/insightface)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br/>

> **CerbroMark** eliminates manual roll-calls and badge swipes forever.  
> Just point a camera — let AI do the rest.

<br/>

![CerbroMark Banner](https://raw.githubusercontent.com/Ronin-117/Automated_attendance/main/GIF.gif)

</div>

---

## 📌 Table of Contents

- [What is CerbroMark?](#-what-is-cerbromark)
- [Core Idea & Motivation](#-core-idea--motivation)
- [System Architecture](#-system-architecture)
- [Sub-System 1: Web App (Flask + dlib)](#-sub-system-1-web-app-flask--dlib)
- [Sub-System 2: Enterprise InsightFace](#-sub-system-2-enterprise-insightface)
- [How the AI Works](#-how-the-ai-works)
- [Full Workflow Diagrams](#-full-workflow-diagrams)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Known Issues & Roadmap](#-known-issues--roadmap)

---

## 🎯 What is CerbroMark?

**CerbroMark** is an end-to-end automated attendance management platform built entirely on **facial recognition AI**. The system uses a live camera feed (USB webcam or professional Hikvision CCTV) to:

1. **Detect** every human face in the frame in real-time
2. **Recognize** who each face belongs to by comparing it against a database of known faces
3. **Log** the person's name and timestamp into an attendance record automatically
4. **Report** the attendance data via CSV export and email delivery

The project is built for **educational institutions, corporate offices, and events** where tracking people's presence is essential but manual methods are slow and error-prone.

---

## 💡 Core Idea & Motivation

Traditional attendance systems are broken:
- ❌ **Manual roll-call** is slow and easily gamed (proxy attendance)
- ❌ **ID cards / RFID** require physical hardware and can be forgotten or shared
- ❌ **Fingerprint scanners** create long queues and cannot operate contactlessly

**CerbroMark's Approach:**
- ✅ Completely **contactless** — no physical interaction needed
- ✅ **Impossible to fake** — a face is unique and always with the person
- ✅ Works in **real-time** and can handle multiple faces simultaneously
- ✅ Scales from a single USB webcam all the way to full **CCTV infrastructure**
- ✅ Sends **automated email reports** with zero manual effort

---

## 🏗️ System Architecture

CerbroMark is a **dual-engine architecture**. Both engines solve the same problem but target very different environments:

```
┌─────────────────────────────────────────────────────────────┐
│                        CerbroMark                           │
│                                                             │
│   ┌─────────────────────┐    ┌────────────────────────┐    │
│   │  Sub-System 1       │    │  Sub-System 2          │    │
│   │  Automated_         │    │  Insightface/          │    │
│   │  attendance-main/   │    │                        │    │
│   │                     │    │                        │    │
│   │  • Flask Web App    │    │  • CLI / CV Window     │    │
│   │  • dlib HOG+ResNet  │    │  • ArcFace + Retina    │    │
│   │  • USB Webcam       │    │  • USB + CCTV IP Cam   │    │
│   │  • CSV + Email      │    │  • .npy embedding DB   │    │
│   │  • CPU Only         │    │  • CPU or CUDA GPU     │    │
│   └─────────────────────┘    └────────────────────────┘    │
│         Lightweight                   Enterprise            │
│         Easy Setup                    High Accuracy         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🌐 Sub-System 1: Web App (Flask + dlib)

### Overview
The web application is the **simple, plug-and-play version** of CerbroMark. It runs entirely on CPU, needs no GPU, and is accessible from any browser on the network. A teacher or event manager opens the webpage, hits **START**, and the camera starts recognizing students automatically.

### AI Model
| Component | Model | Details |
|---|---|---|
| Face Detection | **HOG** (Histogram of Oriented Gradients) | Traditional ML, very fast on CPU |
| Face Recognition | **dlib ResNet** | 128-dimensional face embedding |
| Accuracy | **99.38%** on LFW Benchmark | Industry-standard dataset |
| Matching | **Euclidean Distance** | Tolerance threshold: `0.5` |

### Files
| File | Purpose |
|---|---|
| `get_encodings.py` | One-time setup: converts photos → `.pkl` embedding file |
| `app.py` | Main Flask application — camera, recognition, web server |
| `app_base.py` | Simplified base version (no email, no inference toggle) |
| `templates/main.html` | Admin web UI dashboard |
| `Training_images/` | Add one photo per person here (filename = their name) |
| `Attendance.csv` | Output file: Name + Timestamp per recognized person |

### Web UI Features
- 📹 **Live floating video preview** (bottom-right corner, always visible)
- 📋 **Real-time attendance table** (auto-refreshes every 5 seconds)
- ▶️ **START button** — activates face recognition inference
- ⏹️ **STOP button** — stops recognition and auto-emails the CSV report
- 🗑️ **CLEAR button** — wipes the attendance log for a new session
- 🎓 **Dropdowns** — select Academic Year and Department

### API Endpoints
| Route | Method | Action |
|---|---|---|
| `/` | GET | Renders the admin dashboard |
| `/video_feed` | GET | MJPEG live camera stream |
| `/get_csv` | GET | Returns current Attendance.csv |
| `/st_inf` | POST | Starts face recognition |
| `/ed_inf` | POST | Stops recognition + sends email |
| `/clear_csv` | POST | Resets the attendance log |

---

## 🏢 Sub-System 2: Enterprise InsightFace

### Overview
The enterprise module uses **state-of-the-art deep learning** models far beyond what Sub-System 1 uses. It is designed for situations where accuracy is critical, such as high-security environments or professional CCTV setups. It ships with three operational modes:

### AI Model
| Component | Model | Details |
|---|---|---|
| Face Detection | **RetinaFace** | Deep learning, handles tilted/partial faces |
| Face Recognition | **ArcFace (ResNet-100)** | 512-dimensional face embedding |
| Model Bundle | **`buffalo_l`** | InsightFace's large, highest accuracy pack |
| Accuracy | **99.8%+** on LFW | Significantly outperforms dlib |
| Matching | **Cosine Similarity** | Threshold: `0.6` |
| Runtime | **ONNX Runtime** | Supports CPU + CUDA GPU |

### Three Operational Modes

#### Mode 1 — `main.py`: Live Auto-Discovery
> No registration needed. The system tracks every new face it sees automatically.

- Detects every face → generates a 512-d ArcFace embedding
- Checks the embedding against in-memory face database
- If cosine similarity < 0.6 (new face) → assigns ID (`face1`, `face2`, ...)
- Saves cropped face images to `live_face_db/` for inspection
- **Use case**: Surveillance, anonymous event tracking

#### Mode 2 — `emb_storing.py`: Interactive Registration + Recognition
> Interactive CLI that lets you train the system on real people with names.

- **Registration phase**: Camera shows live feed. You type a person's name, press `s` to capture their embeddings, press `q` to skip
- Deduplicates embeddings to avoid storing the same pose/angle twice
- Saves embeddings as `live_face_db/<name>/<name>_0.npy`, `<name>_1.npy`, etc.
- **Recognition phase**: Loads all saved embeddings and runs live recognition on the camera
- Unknown faces display "Unknown" label
- **Use case**: Small team/classroom registration by admin

#### Mode 3 — `hik_rec.py`: Hikvision CCTV Integration
> Connect directly to an IP camera on the local network for full HD recognition.

- Connects to Hikvision camera via HTTP using `hikvisionapi`
- Pulls **full-resolution JPEG snapshots** from Channel 101 (main stream)
- Runs InsightFace recognition on the **full-res frame** for maximum accuracy
- Displays results in a resized preview window (1280px wide, aspect-preserved)
- Shows similarity score alongside name: e.g., `JOHN DOE (0.87)`
- Auto-reconnects if camera drops from network
- **Use case**: Enterprise environments, CCTV-equipped institutions

---

## 🤖 How the AI Works

### Step 1: Face Detection
The camera frame is passed to the detector:
- **HOG (Sub-System 1)**: Slides a detection window over the image, looking for gradient patterns that match a human face shape. Fast, works on CPU.
- **RetinaFace (Sub-System 2)**: A deep CNN that predicts bounding boxes, facial landmarks (eyes, nose, mouth corners), and face confidence scores simultaneously. Works in challenging lighting and angles.

### Step 2: Face Embedding
Once a face region is found, it's fed into the recognition model:
- **dlib ResNet-128 (Sub-System 1)**: Outputs a **128-number vector** describing the face. Two photos of the same person produce vectors that are very close together in 128-d space.
- **ArcFace ResNet-100 (Sub-System 2)**: Outputs a **512-number vector**. Trained with ArcFace loss which maximizes the angular margin between different identities — making it far more discriminative.

### Step 3: Matching
The live face embedding is compared to every stored embedding:
- **Euclidean Distance (Sub-System 1)**: `√(Σ(a-b)²)` — the geometric separation between two vectors. Below `0.5` = same person.
- **Cosine Similarity (Sub-System 2)**: `(a·b) / (|a| × |b|)` — measures the angle between vectors. Above `0.6` = same person.

### Step 4: Attendance Logging
When a match is confirmed:
- The person's name is extracted from the filename / database key
- `markAttendance()` checks if this name is already in today's CSV
- If not → writes `Name, HH:MM:SS` to `Attendance.csv`
- This prevents duplicate entries for the same person during a session

---

## 🔄 Full Workflow Diagrams

### Sub-System 1: Web App Flow

```
┌─── SETUP (Run Once) ──────────────────────────────────┐
│                                                        │
│  Training_images/                                      │
│  ├── ALICE.jpg   ──┐                                   │
│  ├── BOB.jpg     ──┤── get_encodings.py                │
│  └── CAROL.jpg   ──┘        │                          │
│                             ▼                          │
│                    dlib ResNet-128                     │
│                             │                          │
│                             ▼                          │
│                    encodings.pkl  ◄── saved to disk    │
└────────────────────────────────────────────────────────┘

┌─── RUNTIME (app.py) ──────────────────────────────────┐
│                                                        │
│  encodings.pkl ──► load into memory on startup         │
│                                                        │
│  Browser Admin                                         │
│  [START click] ──► POST /st_inf ──► do_inference=True  │
│                                          │             │
│  USB Webcam                              │             │
│  Frame ──► Resize 25% ──► RGB convert   │             │
│                 │                        │             │
│                 ▼                        ▼             │
│           HOG Detector ◄──────── if (do_inference)    │
│                 │                                      │
│                 ▼  (every 3rd frame)                   │
│         Face Locations                                 │
│                 │                                      │
│                 ▼                                      │
│         ResNet Encoder ──► 128-d embeddings            │
│                 │                                      │
│                 ▼                                      │
│    Compare vs all known encodings                      │
│    (Euclidean Distance < 0.5?)                         │
│                 │                                      │
│       YES ──────┤──── NO ──► skip frame                │
│        │                                               │
│        ▼                                               │
│  markAttendance(name)                                  │
│  Attendance.csv: "ALICE, 09:32:14"                     │
│                 │                                      │
│  Draw green box + name on frame                        │
│  Stream MJPEG ──► Browser Video Preview                │
│                                                        │
│  CSV Table auto-refreshes every 5s ──► Browser Table  │
│                                                        │
│  [STOP click] ──► do_inference=False                   │
│               ──► send_email(Attendance.csv)           │
└────────────────────────────────────────────────────────┘
```

### Sub-System 2: InsightFace Flow (`emb_storing.py`)

```
┌─── REGISTRATION PHASE ────────────────────────────────┐
│                                                        │
│  Admin runs: python emb_storing.py                     │
│                                                        │
│  Terminal: "Enter name: ALICE"                         │
│       │                                                │
│       ▼                                                │
│  Webcam Live Feed ──► RetinaFace Detector              │
│                              │                         │
│                              ▼                         │
│                    ArcFace ResNet-100                  │
│                              │                         │
│                              ▼                         │
│                    512-d Face Embedding                │
│                              │                         │
│         Deduplicate (cosine_sim check)                 │
│                              │                         │
│     Press 's' ──► save to live_face_db/ALICE/          │
│                   ALICE_0.npy, ALICE_1.npy ...         │
│                                                        │
│  Repeat for each person, then press Enter (blank name) │
└────────────────────────────────────────────────────────┘

┌─── RECOGNITION PHASE ─────────────────────────────────┐
│                                                        │
│  Load all .npy files from live_face_db/                │
│  Build in-memory DB: [(embedding, "ALICE"), ...]       │
│                                                        │
│  Webcam / IP Camera frame                              │
│        │                                               │
│        ▼                                               │
│  RetinaFace ──► Face bounding boxes                    │
│        │                                               │
│        ▼                                               │
│  ArcFace ──► 512-d embedding per face                  │
│        │                                               │
│        ▼                                               │
│  Cosine Similarity vs all stored embeddings            │
│        │                                               │
│  Max similarity > 0.6?                                 │
│        │                                               │
│   YES ─┤──► label = "ALICE (0.89)"                     │
│   NO  ─┤──► label = "Unknown"                          │
│        │                                               │
│  Draw box + label on frame ──► cv2.imshow()            │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| Web Framework | Flask | Latest |
| Computer Vision | OpenCV (`cv2`) | 4.x |
| Face Detection (S1) | dlib HOG | via `face_recognition` |
| Face Recognition (S1) | dlib ResNet-128 | via `face_recognition` |
| Face Detection (S2) | RetinaFace | via `insightface` |
| Face Recognition (S2) | ArcFace ResNet-100 | via `insightface` |
| Inference Runtime | ONNX Runtime | CPU + CUDA |
| Data Processing | NumPy, Pandas | Latest |
| Serialization | Pickle (S1), NumPy `.npy` (S2) | Built-in |
| Email | smtplib + Gmail SMTP | Built-in |
| IP Camera | hikvisionapi | Latest |
| Fonts | Google Fonts (Roboto, Silkscreen) | CDN |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Webcam or Hikvision IP Camera
- For Windows (Sub-System 1): **Visual Studio with C++ Desktop Development** workload + **CMake**
- For GPU acceleration (Sub-System 2): CUDA-compatible NVIDIA GPU + cuDNN

---

### 🔵 Sub-System 1: Web App Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/CerbroMark.git
cd CerbroMark/Automated_attendance-main

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add student photos to Training_images/
#    Name each file after the person: ALICE.jpg, BOB.png etc.

# 4. Generate face encodings (run once; re-run when you add new people)
python get_encodings.py

# 5. Launch the web application
python app.py

# 6. Open your browser and navigate to:
#    http://localhost:5000
```

**Using the Web App:**
1. Click **START** to begin face recognition
2. Watch names populate the live attendance table
3. Click **STOP** when the session ends — CSV is auto-emailed
4. Click **CLEAR** to reset for the next session

---

### 🟠 Sub-System 2: InsightFace Setup

```bash
# 1. Navigate to the InsightFace folder
cd CerbroMark/Insightface

# 2. Install dependencies
pip install insightface onnxruntime opencv-python numpy

# Note: InsightFace will automatically download the buffalo_l model
#       (~300MB) on the very first run. Internet connection required.

# --- MODE 1: Auto-Discovery (no registration needed) ---
python main.py
# A window pops up. Every new face gets auto-assigned face1, face2 etc.

# --- MODE 2: Interactive Registration + Recognition ---
python emb_storing.py
# Follow CLI prompts to register people by name, then run recognition

# --- MODE 3: Hikvision CCTV ---
# First, edit hik_rec.py and set your camera IP/credentials:
# CAMERA_IP = "192.168.x.x"
# USERNAME  = "admin"
# PASSWORD  = "your_password"
python hik_rec.py
```

---

## 📁 Project Structure

```
CerbroMark/
│
├── README.md                          ← You are here
│
├── Automated_attendance-main/         ← Sub-System 1: Flask Web App
│   ├── app.py                         ← Main server (inference + email)
│   ├── app_base.py                    ← Simplified base version
│   ├── get_encodings.py               ← One-time encoding generator
│   ├── requirements.txt               ← Python dependencies
│   ├── Attendance.csv                 ← Auto-generated attendance log
│   ├── encodings.pkl                  ← Auto-generated face database
│   ├── Training_images/               ← Add student photos here
│   │   ├── ALICE.jpg
│   │   └── BOB.jpeg
│   └── templates/
│       └── main.html                  ← Admin web dashboard UI
│
└── Insightface/                       ← Sub-System 2: Enterprise System
    ├── main.py                        ← Auto-discovery mode
    ├── emb_storing.py                 ← Interactive register + recognize
    ├── hik_rec.py                     ← Hikvision IP camera mode
    ├── requirements.txt               ← Python dependencies
    └── live_face_db/                  ← Auto-generated embeddings store
        ├── ALICE/
        │   ├── ALICE_0.npy
        │   └── ALICE_1.npy
        └── BOB/
            └── BOB_0.npy
```

---

## ⚙️ Configuration

### Email Setup (Sub-System 1)
> Edit `Automated_attendance-main/app.py`, function `send_email()`:

```python
from_email = "your_gmail@gmail.com"
password   = "your_app_password"     # Gmail App Password (not your login password)
to_email   = "admin@school.edu"
```

> ⚠️ Use Gmail **App Passwords** (enable 2FA on your Google account first, then generate an App Password from Security settings).

### Hikvision Camera Setup (Sub-System 2)
> Edit `Insightface/hik_rec.py`:

```python
CAMERA_IP = "192.168.1.100"   # Your camera's IP on the local network
USERNAME  = "admin"
PASSWORD  = "your_camera_password"
CHANNEL   = 101               # 101 = main HD stream
```

---

## 🗺️ Known Issues & Roadmap

### Known Issues
| Issue | Status | Workaround |
|---|---|---|
| Email credentials in plain code | ⚠️ Security Risk | Move to `.env` / environment variables |
| Year/Department dropdowns are UI-only | 🔧 Not Wired | Backend filtering not yet implemented |
| HOG struggles with non-frontal faces | ⚙️ By Design | Use Sub-System 2 for better angles |
| `cv2.waitKey` error on headless server | ✅ Fixed | Install `opencv-python` not `headless` |

### Roadmap
- [ ] Database integration (SQLite / PostgreSQL) instead of CSV
- [ ] Department/Year filtering wired to backend
- [ ] Web UI for InsightFace registration (replace CLI)
- [ ] Real-time attendance dashboard with charts
- [ ] Mobile app companion for admin monitoring
- [ ] Multi-camera support
- [ ] Environment variable configuration (`.env` file)
- [ ] Docker containerization for easy deployment

---

<div align="center">

**Made with ❤️ | CerbroMark — Redefining Smart Attendance**

*If this project helped you, please consider giving it a ⭐ on GitHub!*

</div>
