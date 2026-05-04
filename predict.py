import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Matches training preprocessing

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = 224
MODEL_PATH = "models/tsl_model.h5"
DATA_PATH = "Data/raw"  # Actual casing
TEST_IMAGE_PATH = "Data/test/test_car.jpg"

def get_class_indices(data_path):
    """Discover classes from training data folders (sorted for consistency)."""
    class_names = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    return {name: idx for idx, name in enumerate(class_names)}

def predict_image(model, class_indices, img_path):
    """Predict on single image, return top class + confidence."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    # Load and preprocess (same as training)
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # MobileNetV2 specific normalization
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    
    predicted_class = [k for k, v in class_indices.items() if v == predicted_class_idx][0]
    
    return predicted_class, confidence, img  # Return img for visualization

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
    
    try:
        predicted_class, confidence, img = predict_image(model, class_indices, img_path)
        
        # Visualize
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}\\nConfidence: {confidence:.2%}", fontsize=16, pad=20)
        plt.axis('off')
        plt.show()
        
        print(f"\n🎯 Predicted: {predicted_class} (confidence: {confidence:.2%})")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
