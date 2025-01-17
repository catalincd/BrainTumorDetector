import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing import image, image_dataset_from_directory
from tensorflow.keras import utils
from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
import numpy as np
import json

model_name = 'ResNet-T4'

model = ResNet152(weights='imagenet')

base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(287, 348, 3))

base_model.trainable = False

flatten = tf.keras.layers.Flatten()(base_model.output)
dense = tf.keras.layers.Dense(1024, activation='relu')(flatten)
output = tf.keras.layers.Dense(2, activation='softmax')(dense)

fine_tuned_model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

fine_tuned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dir = 'data/train/'

dataset = image_dataset_from_directory(
    train_dir,
    image_size=(287, 348)
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

train_history = fine_tuned_model.fit(train_dataset, validation_data=val_dataset, epochs=5)
fine_tuned_model.save(f'models/{model_name}.h5')

history_dict = train_history.history 
history_file_path = f'models/{model_name}-history.json'

with open(history_file_path, 'w') as f:
    json.dump(history_dict, f)
