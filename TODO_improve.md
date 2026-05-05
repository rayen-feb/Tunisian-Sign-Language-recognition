# Improve Generalization for New Images

✅ Plan approved (stronger aug/TTA/dropout)

- [x] 1. Edit main.py:
  * ✅ Stronger aug (shear/shift/channel)
  * ✅ Dropout 0.6
  * [ ] Cosine LR schedule
- [ ] 2. Edit predict.py: TTA (16 aug predictions average)
- [ ] 3. Retrain: `python main.py`
- [ ] 4. Test new images: `python predict.py`
- [ ] 5. Metrics: `python metrics_complete.py`

Starting edits...

