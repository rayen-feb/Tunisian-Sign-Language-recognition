import tensorflow as tf  #deep learning framework for build neural netword , train  and run models 
from tensorflow.keras.preprocessing.image import ImageDataGenerator #handles images datasets 
#load image from folders  / preprocess them(resize , normalize ... =
#applie data augmentation : rotation , zoom , flip 
from tensorflow.keras.applications import MobileNetV2
#cnn pre trained cnn ; already trained on millions of images 
#you donr start from zero  : faster + better accurancy 
import os 

#----config------

