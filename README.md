# 🤟 Tunisian Sign Language Recognition

A deep learning project using **MobileNetV2** to recognize Tunisian Sign Language gestures from images.  
The system integrates **MediaPipe** for hand detection, supports ~57 classes, and provides predictions as **text, confidence scores, and Arabic voice spelling** via gTTS.

---

## 📸 Preview
![Sign Language Recognition Demo](https://images.unsplash.com/photo-1551240903-154be3f2e18b?q=80&w=1169&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D?q=80&w=1170&auto=format&fit=crop)


---

## 🚀 Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
Train the model:

bash
python main.py
Test on a single image:

bash
python predict.py path/to/image.jpg
Launch the Gradio web app:

bash
python app_gradio.py
Then open the local URL (default: http://127.0.0.1:7860).

🌐 Web App (Gradio)
Upload, drag & drop, or capture a photo of a sign to get predictions with text + confidence scores.

Features
🖐️ Two modes:

Single Image → predict one sign

Sentence Prediction → combine multiple signs

📷 Webcam capture with a 3‑second countdown

✋ Hand cropping via MediaPipe

🔊 Voice spelling in Arabic (gTTS)

📊 Top‑3 predictions with confidence bars

🖼️ Example images from Data/test/

🧠 Hand Model
hand_crop.py uses MediaPipe HandLandmarker.
If hand_landmarker.task is missing, the app falls back to a centered crop (less precise but functional).

📚 Classes
Auto‑detected from Data/raw/ folders — currently ~57 signs.

📊 Status
See [Il semble que le résultat n’était pas sûr à afficher. Changeons un peu et essayons autre chose !] for pending tasks and improvements.

🔗 Links
📂 GitHub Repository

🎥 Live Demo (local): http://127.0.0.1:7860  
(Deploy on Hugging Face Spaces or Streamlit Cloud for public access)

🔮 Future Work
Expand dataset with more Tunisian sign classes

Deploy on Hugging Face Spaces for public demo

Add real‑time video recognition with continuous prediction

Improve accuracy with transfer learning and fine‑tuning


---

✨ This is the **full code for your README** — clean, professional, and ready to showcase.  
Would you like me to also create a **matching portfolio HTML section** (like the Soil Fertilization one) that pulls directly from this README so your site stays consistent?

---

✨ This is the **full code for your README** — clean, professional, and ready to showcase.  
Would you like me to also create a **matching portfolio HTML section** (like the So
