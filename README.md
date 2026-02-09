# MultiMedAI – Computer Vision Assisted Prakriti Assessment

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellow?logo=scikitlearn)
![Flask](https://img.shields.io/badge/Flask-Backend-black?logo=flask)
![React](https://img.shields.io/badge/React.js-Frontend-blue?logo=react)
![Status](https://img.shields.io/badge/Status-Active-success)

MultiMedAI is an **AI-powered, computer vision–driven system** that assists Ayurvedic practitioners in **Prakriti (Vata–Pitta–Kapha) assessment** by combining **facial analysis, teeth analysis, questionnaire intelligence, and Human-in-the-Loop (HITL) validation**.

The system is designed to be **doctor-assistive**, explainable, and scalable — not a black-box diagnosis tool.

---
## Overview
In Ayurvedic diagnostics, visual analysis (Darshana) plays a crucial role. MultiMedAI automates and standardizes this process by:

- Capturing **stable facial input** from a live camera.
- Inferring a **normalized 3D facial mesh** from regular RGB video.
- Enforcing **temporal stability** across frames (neutral pose, lighting, expression).
- Extracting reproducible facial measurements using **controlled 3D–2D geometry**.

---

##  Key Features

###  Multi-Modal Analysis
- **Face Mesh Capture (MediaPipe)**  
  High-fidelity 3D facial landmark extraction with pose, quality, and stability validation.
- **Teeth & Smile Analysis**  
  Teeth visibility, mouth openness, and ROI-based evidence extraction.
- **Nail Analysis**  
  Nail size and nail to finger ratio on real time web cam input.
- **Questionnaire Prefill Engine**  
  Auto-prefills Ayurvedic questionnaire answers using CV outputs + confidence scores.

### Intelligent Quality Gatekeeping
- **Pose Gatekeeper** – Ensures correct frontal/profile alignment  
- **Quality Gatekeeper** – Checks blur, lighting, distance, expression neutrality  
- **Stability Gatekeeper** – Captures only temporally stable frames

### Human-in-the-Loop (HITL)
- Doctors can **review, override, and correct AI predictions**
- All corrections are logged for transparency and future learning

### Evidence-Driven System
- Question-wise **image evidence linking**
- ROI snapshots for explainability
- Frontend-ready evidence APIs

---

## System Architecture
1.  Camera Feed
2.  Face Capture (MediaPipe)
3.  Pose + Quality + Stability Gatekeepers
4. Golden Mesh & Image Storage
5. Analysis Pipelines (Face / Teeth / Features)
6. Questionnaire Prefill Engine
7. Doctor Review (HITL)
8. Final Prakriti Scoring

---

## Tech Stack

| Layer | Technology |
|------|-----------|
| Computer Vision | MediaPipe, OpenCV |
| Backend API | FastAPI |
| Processing | NumPy, Python |
| Storage (Current) | File-based session storage |
| Storage (Planned) | PostgreSQL |
| Frontend | React |
| Security | Encrypted local file serving |

---

## API Overview

### Core APIs
- `POST /api/capture/start` – Launch face capture pipeline
- `GET /api/sessions` – List all captured sessions
- `GET /api/session/{id}/summary` – One-shot session summary
- `GET /api/session/{id}/evidence` – Question-wise image evidence
- `POST /api/submit` – Final submission + Prakriti scoring

### File Serving
- Secure encrypted image serving via `/api/file`

---

## ⚙️ Setup Instructions

### 1️. Clone Repository
```bash
git clone https://github.com/your-username/MultiMedAI.git
cd MultiMedAI
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run Backend
```bash
cd backend
uvicorn app:app --reload
```
---
## Future Scope
### 1. Beard & Turban Awareness
Detect facial obstructions (beard, turban, accessories) and dynamically adjust ROI selection to avoid biased feature extraction.

### 2. Accuracy Improvement via Threshold Tuning
Continuously refine pose, quality, and feature thresholds using real-world capture data.

### 3. Structured Database (PostgreSQL)
Migrate from file-based storage to PostgreSQL for scalable querying, analytics, and long-term audit trails.

### 4. Moles & Freckles Detection
Extend skin analysis to include moles, freckles, and pigmentation patterns for richer Prakriti indicators.

### 5. Advanced Human-in-the-Loop
Convert expert feedback into structured learning signals and visual PPT-style explainability reports.

---
## Disclaimer

This system is assistive and not a medical diagnostic tool.
Final Prakriti assessment must always be validated by a qualified Ayurvedic practitioner.

---
## Contributors
- Sania Verma 
- Manishika Gupta
- Maanvi
- Mehak

---
## Acknowledgement
- Central Council for Ayurvedic Sciences, Ministry Of AYUSH, Govt. of India
---
_Building trustworthy AI for traditional medicine, one frame at a time_
