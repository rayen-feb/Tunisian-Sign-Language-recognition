# Tunisian Sign Language Recognition

Deep learning model (MobileNetV2) for recognizing Tunisian Sign Language gestures from images.

## 📁 Project Structure
```
.
├── Data/raw/          # Training images by class (3aslema, 3ayla, etc.)
├── models/            # tsl_model.h5 (trained)
├── plots/             # Training curves
├── main.py            # Train model
├── predict.py         # Test on single image (terminal)
├── app_gradio.py      # Web app — upload/photo → sign as text + voice
├── hand_crop.py       # MediaPipe hand cropping (focus model on the hand)
├── word_map.py        # Maps labels → Arabic phrases for voice (gTTS)
├── requirements.txt   # Dependencies
└── README.md          # This file
```

## 🚀 Quick Start
1. Install: `pip install -r requirements.txt`
2. Train: `python main.py`
3. Test image: `python predict.py` (enter path)
4. **Web app:** `python app_gradio.py`

## 🌐 Web App (Gradio)
Upload, drag & drop, or take a photo of a sign and get the predicted sign as **text** + confidence.

```bash
python app_gradio.py
```
Then open the local URL shown in the terminal (default: http://127.0.0.1:7860).

### Features
- 🖐️ **Two modes:**
  - **Single Image** — upload or webcam a sign → predicted sign as text
  - **Sentence Prediction** — upload multiple signs → combined sentence
- 📷 **Webcam capture** with a 3-second on-screen countdown
- ✋ **Hand cropping** (MediaPipe) — the model focuses only on the hand for better accuracy
- 🔊 **Voice spelling** of the predicted sign in Arabic (gTTS)
- 📊 **Top 3 predictions** with confidence bars
- 🖼️ Example images from `Data/test/`

### Hand model
`hand_crop.py` uses MediaPipe's `HandLandmarker` which needs a `hand_landmarker.task` file in the project root. If it's missing, the app automatically falls back to a centered crop (still works, just less precise).

## Classes
Auto-detected from `Data/raw/` folders (currently ~57 signs).

## 📊 Status
See [TODO.md](TODO.md)
