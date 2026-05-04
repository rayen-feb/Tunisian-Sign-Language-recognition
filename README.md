# Tunisian Sign Language Recognition

Deep learning model (MobileNetV2) for recognizing Tunisian Sign Language gestures from images.

## 📁 Project Structure
```
.
├── Data/raw/          # Training images by class (3aslema, 3ayla, etc.)
├── models/            # tsl_model.h5 (trained)
├── plots/             # Training curves
├── main.py            # Train model
├── predict.py         # Test on single image
├── requirements.txt   # Dependencies
└── TODO.md            # Progress
```

## 🚀 Quick Start
1. Install: `pip install -r requirements.txt`
2. Train: `python main.py`
3. Test image: `python predict.py` (enter path)

## 📊 Status
See [TODO.md](TODO.md)

## Classes
Auto-detected from `Data/raw/` folders (~40 signs).
