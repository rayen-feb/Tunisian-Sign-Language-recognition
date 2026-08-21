# Improve Generalization for New Images

✅ Plan approved (stronger aug/TTA/dropout)

## Completed
- [x] 1. Edit main.py:
  * ✅ Stronger aug (shear/shift/channel)
  * ✅ Dropout 0.6
  * ✅ Cosine LR schedule (added `cosine_lr_schedule` + `LearningRateScheduler`)
  * ✅ Fixed preprocessing: use `preprocess_input` (ImageNet norm) instead of `rescale=1./255` for consistency with predict.py
- [x] 2. predict.py: TTA (16 aug predictions average) — already implemented
- [x] 3. metrics_complete.py: fixed preprocessing to use `preprocess_input` (match training pipeline)

## Next Steps
- [ ] 4. Retrain: `python main.py`
- [ ] 5. Test new images: `python predict.py`
- [ ] 6. Metrics: `python metrics_complete.py`

## Progress
- [x] Plan approved
- [x] Cosine LR schedule added to main.py
- [x] metrics_complete.py preprocessing fix
- [x] main.py preprocessing consistency fix
- [x] Web app (app_gradio.py) created — upload/photo → sign as text + voice
- [x] Webcam capture added with 3-second countdown timer
- [x] Sentence prediction mode (multiple images → combined sentence)
- [x] Voice spelling of predicted sign in Arabic (word_map.py + gTTS)
- [x] Hand cropping with MediaPipe (hand_crop.py) — fixed for new MediaPipe Tasks API
- [x] requirements.txt updated (gradio, mediapipe, gTTS, pygame)
- [x] README.md updated with full feature docs
- [ ] Retrained model
- [ ] Tested new images
- [ ] Metrics verified
