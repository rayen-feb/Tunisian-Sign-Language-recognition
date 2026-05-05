import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "models/tsl_model.h5"
DATA_PATH = "Data/raw"
IMG_SIZE = 224
BATCH_SIZE = 32
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

model = load_model(MODEL_PATH)
test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
test_data = test_datagen.flow_from_directory(DATA_PATH, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode='sparse', subset='training', shuffle=False)

y_test = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = np.argmax(y_pred_prob, axis=1)

class_names = list(test_data.class_indices.keys())
report = classification_report(y_test, y_pred, target_names=class_names)
print(report)

with open(os.path.join(PLOTS_DIR, "metrics_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.show()

print("Done!")
