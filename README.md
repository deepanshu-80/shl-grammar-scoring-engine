# SHL Grammar Scoring Engine 🎧🗣️

This project automatically scores spoken English grammar from an audio recording.  
It converts **speech → text → analyzes grammar → predicts a fluency score (1–5)** similar to SHL spoken assessment scoring.

## ✅ Features

| Feature | Description |
|--------|-------------|
| 🎙 Speech to Text | Converts audio into transcript |
| 📝 Grammar & Fluency Analysis | Detects mistakes, fillers, fluency patterns |
| 🤖 ML Model Scoring | RandomForest-based score prediction (1–5) |
| 🌐 Web Demo (Gradio) | Test using **microphone** or **file upload** |
| 🚩 Flag System | Save wrong predictions for future retraining |

## 🛠 Requirements

- Python **3.8 – 3.11**
- Works on **Windows / Mac / Linux**
## 📦 Installation

### 1) Clone the repository
git clone https://github.com/deepanshu-80/shl-grammar-scoring-engine.git
cd shl-grammar-scoring-engine

### 2) Create virtual environment
python -m venv venv

### 3) Activate environment

**PowerShell (Windows):**
venv\Scripts\Activate

**Command Prompt:**
venv\Scripts\activate.bat

**Mac/Linux:**
source venv/bin/activate

### 4) Install dependencies
pip install -r requirements.txt


## ▶️ Run Web App (Gradio UI)

python demo/app.py

This will open a browser UI.

You can:
- 🎙 **Record live audio**
- 📁 **Upload audio file (.wav recommended)**

You will receive:
- **Grammar Score (1–5)**
- **Transcript (model interpretation)**

## 👤 Author
**Deepanshu Ruhela**  
