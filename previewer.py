import sys
import os
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QStackedWidget, QHBoxLayout, QPushButton
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
import tensorflow as tf
from tensorflow.keras.preprocessing import image

class ImageCarousel(QWidget):
    def __init__(self, model, image_folder):
        super().__init__()

        self.model = model
        self.image_folder = image_folder
        self.images = self.load_images(image_folder)
        self.current_image_index = 0
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Image Carousel with Predictions')
        
        self.layout = QVBoxLayout()

        self.carousel = QStackedWidget(self)
        self.layout.addWidget(self.carousel)

        self.prediction_label = QLabel('Prediction: ', self)
        self.layout.addWidget(self.prediction_label)

        self.nav_layout = QHBoxLayout()
        self.prev_button = QPushButton('Previous', self)
        self.next_button = QPushButton('Next', self)
        self.nav_layout.addWidget(self.prev_button)
        self.nav_layout.addWidget(self.next_button)
        self.layout.addLayout(self.nav_layout)

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        self.setLayout(self.layout)

        self.show_image(self.current_image_index)

    def load_images(self, folder_path):
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        print(image_files)
        return image_files

    def load_image(self, image_path):
        img = image.load_img(image_path, target_size=(287, 348))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0 
        return img_array

    def predict_image(self, img_array):
        predictions = self.model.predict(img_array)
        return predictions

    def show_image(self, index):
        image_path = self.images[index]
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(574, 768, Qt.AspectRatioMode.KeepAspectRatio)

        image_label = QLabel(self)
        image_label.setPixmap(pixmap)
        self.carousel.addWidget(image_label)

        img_array = self.load_image(image_path)
        predictions = self.predict_image(img_array)

        predicted_class = np.argmax(predictions, axis=1) 
        self.prediction_label.setText(f'Prediction: {predicted_class[0]}')

    def show_previous_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        self.carousel.setCurrentIndex(self.current_image_index)
        self.show_image(self.current_image_index)

    def show_next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.carousel.setCurrentIndex(self.current_image_index)
        self.show_image(self.current_image_index)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def main():
    app = QApplication(sys.argv)
    
    model_path = 'models/ResNet-T4.h5'  
    image_folder = 'data/train/no'  

    model = load_model(model_path)

    window = ImageCarousel(model, image_folder)
    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()
