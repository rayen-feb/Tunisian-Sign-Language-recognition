import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = 224
MODEL_PATH = "models/tsl_model.h5"
DATA_PATH = "Data/raw"
TEST_IMAGE_PATH = "Data/test/test_car.jpg"

def get_class_indices(data_path):
    """Discover classes from training data folders (sorted for consistency)."""
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    return {name: idx for idx, name in enumerate(class_names)}

def predict_image(model, class_indices, img_path, tta=False, tta_steps=32, top_k=3):
    """Predict on single image with optional TTA (Test-Time Augmentation) and Top-K results."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    if not tta:
        predictions = model.predict(img_array, verbose=0)
    else:
        print(f"🔄 TTA: averaging {tta_steps} augmented predictions...")
        predictions = np.zeros((1, len(class_indices)))
        tta_datagen = ImageDataGenerator(
            rotation_range=20, zoom_range=0.2, width_shift_range=0.1,
            height_shift_range=0.1, horizontal_flip=True,
            brightness_range=[0.8, 1.2], shear_range=0.15
        )
        for _ in range(tta_steps):
            aug_img = next(tta_datagen.flow(img_array, batch_size=1))
            predictions += model.predict(aug_img, verbose=0)
        predictions /= tta_steps

    # Top-K predictions
    top_indices = predictions[0].argsort()[-top_k:][::-1]
    results = [(list(class_indices.keys())[i], predictions[0][i]) for i in top_indices]
    return results, img

def main():
    print("🔮 Tunisian Sign Language Model Tester")
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Train first using: python main.py")
        return
    
    model = load_model(MODEL_PATH)
    print("✅ Model loaded.")
    
    # Get classes
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data path not found: {DATA_PATH}")
        return
    
    class_indices = get_class_indices(DATA_PATH)
    num_classes = len(class_indices)
    print(f"📚 Found {num_classes} classes: {list(class_indices.keys())}")
    
    # Image path (default or user input)
    img_path = input(f"\nEnter image path (default: {TEST_IMAGE_PATH}): ").strip()
    if not img_path:
        img_path = TEST_IMAGE_PATH
    
    # TTA option
    use_tta = input("Use TTA for better new image accuracy? (y/n): ").strip().lower() in ['y', 'yes']
    
    try:
        results, img = predict_image(model, class_indices, img_path, tta=use_tta)
        
        # Visualize
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        title_text = "\n".join([f"{cls}: {conf:.2%}" for cls, conf in results])
        plt.title(f"Predictions:\n{title_text}", fontsize=16, pad=20)
        plt.axis('off')
        plt.show()
        
        print("\n🎯 Top predictions:")
        for cls, conf in results:
            print(f"- {cls}: {conf:.2%}")
        if use_tta:
            print("✅ Predictions averaged with TTA")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
