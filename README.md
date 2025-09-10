# 🎤 Risonanza — Voice Emotion & Stress Detection

Risonanza is an open-source system for **automated, reliable stress and emotion detection from voice audio**, enabling real-time analysis with privacy, efficiency, and flexibility. Developed for the Bit-n-Build Hackathon, it demonstrates powerful ML on edge/consumer devices, without cloud dependencies.

## 📌 Problem Statement

People frequently reveal their stress and emotions through their voice, but **traditional detection is subjective and unreliable**. Misinterpretation can lead to problems in critical areas like security, healthcare, and customer service.

**Risonanza's mission:**  
To deliver an accurate, automated system that analyzes voice patterns in real time, reliably detecting stress and emotions across diverse speakers and environments.


## 💡 Solution & Workflow

Risonanza combines feature-rich audio analysis with classical ML models in a streamlined pipeline:

**Workflow:**
1. **Input Audio:** User uploads or records voice data
2. **Feature Extraction:** 
   - MFCCs
   - Spectral features (centroid, rolloff, bandwidth)
   - Pitch, rhythm, energy
3. **Emotion Classification:**  
   - Predicts one of 8 emotions:
     - Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprise
4. **Stress Calculation:**  
   - Maps emotion probabilities to a stress percent using scientific weights
5. **Display & UX:**
   - Real-time, interactive analysis via Streamlit frontend
   - Visualization, feedback, and actionable insights


## 🔬 Technical Features

- **Lightweight Machine Learning:** Feature extraction with Random Forest (scikit-learn)
- **Efficient Resource Usage:** Runs on standard hardware, edge devices (IoT, robotics)
- **Privacy-Friendly:** No data leaves device — *local only*
- **Open Source (MIT License):** Free for academia, research, and commercial use
- **Scalable & Modular:** Easy integration into broader analytics systems


## 📦 File Structure

```
Risonanza/
│
├── .gitignore
├── app.py                      # Streamlit interface
├── README.md                   # Project documentation (this file)
├── requirements.txt            # Python packages
├── setup_dataset.py            # Utility for downloading RAVDESS
├── stress_detection_model.pkl  # Saved ML model after training
└── risonanza/
    ├── __init__.py
    ├── audio_feature.py        # Core audio processing & augmentation
    ├── helpers.py              # Dataset loading, feature building
    └── stress_detector.py      # ML model, stress probability logic
```


## 🗂️ How To Run / Usage Guide

### 1. Clone & Setup

```
git clone <repository-url>
cd Risonanza
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # Mac/Linux
pip install -r requirements.txt
```

### 2. Download Dataset

```
python setup_dataset.py
```
- Downloads and organizes the RAVDESS emotional speech audio dataset for training.

### 3. Train the Model

Start the app and use sidebar's "Train Model" — or run scripts to build features and train using local audio files.
- Model will auto-save as `stress_detection_model.pkl`.

### 4. Launch Analysis App

```
streamlit run app.py
```

- Analyze uploaded or live-recorded voice
- View emotion prediction, stress score, and audio visualizations
- Check usage tips and real-world scenarios in the About tab


## ⚙️ Stress Mapping Logic

```
# Order: Neutral, Calm, Happy, Sad, Angry, Fear, Disgust, Surprise
weights = np.array([0.0, -1.0, -1.0, 0.8, 1.0, 1.0, 0.7, 0.3])
stress_score = np.dot(pred_probs, weights)
stress_percent = ((stress_score + 1) / 2) * 100
```
- **Interpretation:**  
  - 0% → calmest  
  - 100% → highest stress

## 🎯 Real World Applications

- Security: Suspicious behavior/anomaly voice detection
- Customer Experience: Real-time user feedback & emotion analysis
- Healthcare: Early-stage mental health stress tracking
- Robotics: Emotion-aware robot interaction
- Workplace Wellness: Passive emotion/stress monitoring in high-pressure fields

## 👩‍💻 Contributors

- Sai Srikar B
- Shwetha Ram R
- M.V. Harish Kumar

## 📜 License

Risonanza is MIT Licensed. See [LICENSE](https://github.com/harishtpj/Risonanza/blob/ec20819dd5c49f3d9d1df118f545c6cbc85d7b5f/LICENSE) for details.

*Inspired by a mission for reliable, privacy-respecting emotional analytics. Made for Bit-n-Build Hackathon, 2025 — The Valour Team.*
