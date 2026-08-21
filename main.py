import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
HEAD_EPOCHS = 12
FT_EPOCHS = 25
DATA_PATH = "Data/raw"
PLOTS_DIR = "plots"
MODEL_DIR = "models"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "tsl_model.h5")

USE_HORIZONTAL_FLIP = True   # allow flips for stronger augmentation
TEST_IMAGE_PATH = "Data/test/test_car.jpg"

# =========================
# PREPROCESSING
# =========================
def _preprocess(x):
    return preprocess_input(x.astype("float32"))

train_datagen = ImageDataGenerator(
    preprocessing_function=_preprocess,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=USE_HORIZONTAL_FLIP,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    channel_shift_range=30.0,   # color jitter
)

val_datagen = ImageDataGenerator(
    preprocessing_function=_preprocess,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42,
)

val_data = val_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42,
)

print("\n✅ Classes:", train_data.class_indices)
num_classes = train_data.num_classes

# =========================
# CLASS WEIGHTS (for imbalance)
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.6)(x)   # stronger dropout
x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = tf.keras.layers.Dropout(0.4)(x)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=base_model.input, outputs=output)

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# =========================
# PHASE 1
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
)

phase1_callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
]

print("\n🚀 Phase 1: training classification head...\n")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=HEAD_EPOCHS,
    callbacks=phase1_callbacks,
    class_weight=class_weights
)

# =========================
# PHASE 2 (full fine-tuning)
# =========================
for layer in base_model.layers:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6),
    loss=loss_fn,
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
)

phase2_callbacks = [
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-7),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
]

print("\n🚀 Phase 2: fine-tuning full base model...\n")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FT_EPOCHS,
    callbacks=phase2_callbacks,
    class_weight=class_weights
)

# =========================
# EVALUATION
# =========================
val_loss, val_acc, val_top3 = model.evaluate(val_data, verbose=0)
print(f"\n🎯 Final Validation Accuracy: {val_acc:.2%}")
print(f"📉 Final Validation Loss: {val_loss:.4f}")
print(f"🔝 Top-3 Accuracy: {val_top3:.2%}")

val_data.reset()
y_true = val_data.classes
y_pred_probs = model.predict(val_data, steps=math.ceil(val_data.samples / BATCH_SIZE))
y_pred = np.argmax(y_pred_probs, axis=1)
labels = list(train_data.class_indices.keys())

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"📂 Confusion matrix saved to: {cm_path}")

# =========================
# SINGLE IMAGE TEST
# =========================
if os.path.exists(TEST_IMAGE_PATH):
    img = image.load_img(TEST_IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = _preprocess(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    top_indices = prediction[0].argsort()[-3:][::-1]
    print("\n🖼 Test Image Top-3 Predictions:")
    for idx in top_indices:
        print(f"- {labels[idx]}: {prediction[0][idx]:.2%}")
else:
    print("\nℹ No test image provided.")
