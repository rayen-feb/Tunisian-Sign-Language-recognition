# Metrics Upgrade Plan (Approved)

**Goal:** Add Confusion Matrix, Precision/Recall, F1-score to main.py

**File:** main.py

**Changes:**
1. Fix DATA_PATH = "Data/raw"
2. Define PLOTS_DIR = "plots"
3. Add test_datagen (no aug, validation_split=0.1, subset='training' for test)
4. After training:
   - test_data = test_datagen.flow_from_directory(...)
   - y_test = test_data.classes
   - y_pred = model.predict(test_data)
   - y_pred_classes = argmax(y_pred)
5. ConfusionMatrixDisplay (sklearn), plot heatmap, save plots/confusion_matrix.png
6. print(classification_report(y_test, y_pred_classes))
   - Save to plots/metrics_report.txt

**Imports add:** from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import seaborn as sns

**Followup:** 
- pip install seaborn
- python main.py (train + new metrics)

**Status:** Plan ready → Edit main.py

