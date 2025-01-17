import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras import utils
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
import numpy as np
import json

fine_tuned_model = load_model('models/ResNet-T4.h5')
train_dir = 'data/train/'

dataset = image_dataset_from_directory(
    train_dir,
    image_size=(287, 348),
    batch_size=32, 
    label_mode='categorical', 
    shuffle=True 
)

train_dataset_size = int(0.8 * len(dataset)) 
val_dataset_size = int(0.1 * len(dataset))  
test_dataset_size = len(dataset) - train_dataset_size - val_dataset_size

train_dataset = dataset.take(train_dataset_size)
val_dataset = dataset.skip(train_dataset_size).take(val_dataset_size)
test_dataset = dataset.skip(train_dataset_size + val_dataset_size)



loss, accuracy = fine_tuned_model.evaluate(test_dataset)

print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
