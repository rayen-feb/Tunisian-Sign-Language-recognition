import os # manage system files 
import numpy as np  
import matplotlib.pyplot as plt # graph lib
import tensorflow as tf #deep learning framework
from tensorflow.keras.preprocessing.image import ImageDataGenerator #load images , resize , apply  augentation (rotate , flip , etx ) 
from tensorflow.keras.applications import MobileNetV2 # pretrained cnn model
from tensorflow.keras.preprocessing import image # load sigle test images 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = 224  # all images resizeed to 224 224 
BATCH_SIZE = 32 # model process 32 images at  once 
EPOCHS = 20 #model see dataset 20 times 
DATA_PATH = "data/raw" 
MODEL_PATH = "models/tsl_model.h5"  # where trained model will be saved 


# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    rescale=1./255, # normalise pixels  from [0 255] to [0 1] 
    validation_split=0.2, # 20 % of data for validation
    rotation_range=20, # randomly rotate images by up to 20 degrees
    zoom_range=0.3, # randomly zoom in images by up to 30%
    horizontal_flip=True,  # randomly flip images horizontally
    brightness_range=[0.8, 1.2] # randomly adjust brightness between 80% and 120%
)

train_data = train_datagen.flow_from_directory(    #load images from directory and apply augmentation
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical', #Multi-class classification (one-hot encoding)
    subset='training' #use the training subset defined by validation_split
)

val_data = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("\n✅ Classes:", train_data.class_indices)

# =========================
# MODEL (TRANSFER + CUSTOM HEAD)
# =========================
base_model = MobileNetV2(  #load pretrained model without top layers (classification head)
#loading the MobileNetV2 deep learning model, which is already trained on a huge dataset (ImageNet).
 input_shape=(IMG_SIZE, IMG_SIZE, 3),
 #define input image format (224x224 pixels with 3 color channels)
    include_top=False, #original model was trained to classify 1000 imageNet , we dont need this part   
    #we will add our own classifier
    weights='imagenet'  #load pre trained weights from imagenet databae
    #model already know paths : edges , etxture , shapes ...
)

#-------------------------
# Freeze most layers
#------------------------
#mobilenetv2 has many layers ( like 100) 
#each layer learns different features 
#first layers : edge , line 
#middle layers : shape 
#last layers : high level features ( like face , hand , etx )

for layer in base_model.layers[:-30]:
    layer.trainable = False
#/freeze all layers except the last 30 
# because these layers know general images features 
for layer in base_model.layers[-30:]:
    layer.trainable = True
#only last 30 layers are trained 

# Custom classification head
x = base_model.output
# x take mobile net outputs 
x = tf.keras.layers.GlobalAveragePooling2D()(x)
#convert thee the ouput into 1d vector 
#cnn output is like 3D block (height *width*channles)
#this layer compress it into a single vector
#benefits : reduce complexity , prevent overfitting , faster training 
x = tf.keras.layers.BatchNormalization()(x)
#normalizing values inside the network 

#first dense layer 
x = tf.keras.layers.Dense(256, activation='relu')(x)
#fully connected layer with 256 neurons and ReLU activation function
# relu keep positive signals , remove negative noise 
x = tf.keras.layers.Dropout(0.5)(x)
# randomly disable  50% OF NEURONS DURING TRAINING
# prevent overfitting by forcing the model to learn more robust features

#second dense layer 
x = tf.keras.layers.Dense(128, activation='relu')(x)
#dropout 0.3
x = tf.keras.layers.Dropout(0.3)(x)

#output layer
output = tf.keras.layers.Dense(train_data.num_classes, activation='softmax')(x)
#final layers  with number of neurons equal to number of classes and softmax activation for multi-class classification


#build the final model by connecting the base model with the custom head
model = tf.keras.Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE
# =========================
#this function teels tensor flow 
#how to optimize learnng 
#how to measure mistakes 
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    #adam is smart optimizer 
    #it decides how to adjust weights during training
    #uring training the model try to reduce loss by adjust weight
    #but we face 3 problem :
    #too slow learning : steps are too small
    #too fast learning : steps are too big and we miss the optimal solution
    #adam adjust learning rate during training to find the sweet spot


    loss='categorical_crossentropy',
    #loss function for multi-class classification
    #we have multiple classes ( monday , tuesday .... 
    #label are one hot encoded 
    #it compare what the model predicted vs what correct answer is 

    metrics=['accuracy']
    #show the occurancy
)

model.summary()

# =========================
# CALLBACKS (VERY IMPORTANT)
# =========================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    #stop training if the model stop improving on validation loss for 5 consecutive epochs
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6),
    #reduce learning rate by 30% if validation loss stop improving for 3 consecutive epochs
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
    #save the model weights to MODEL_PATH only when validation accuracy improves    
]
#

# =========================
# TRAINING
# =========================
print("\n🚀 Training started...\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# PLOT RESULTS
# =========================
plt.figure(figsize=(10, 4))

#accurancy plots 
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title("Accuracy")
plt.legend()

#loss plot 
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title("Loss")
plt.legend()


#save the plot
plot_path=os.path.join(PLOTS_DIR, "training_curves.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n Training curves saved to: {plot_path}")
plt.show()

# =========================
# TEST ON IMAGE
# =========================
TEST_IMAGE_PATH = "Data/test/test_car.jpg"

if os.path.exists(TEST_IMAGE_PATH):
    img = image.load_img(TEST_IMAGE_PATH, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    labels = list(train_data.class_indices.keys())

    print("\n🔍 Prediction:", labels[class_index])
else:
    print("\nℹ️ No test image provided.")