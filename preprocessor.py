import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
import kagglehub

def download_dataset():
    path = kagglehub.dataset_download("jakupymeraj/cats-and-dogs-image-dataset")
    train_dir = os.path.join(path, "dataset", "training_set")
    return train_dir

def load_data(image_size=(150,150)):
    train_dir = download_dataset()
    cats_dir = os.path.join(train_dir, "cats")
    dogs_dir = os.path.join(train_dir, "dogs")
    data, labels = [], []
    
    for img_file in os.listdir(cats_dir):
        img_path = os.path.join(cats_dir, img_file)
        img = load_img(img_path, target_size=image_size)
        data.append(img_to_array(img))
        labels.append(0)
    
    for img_file in os.listdir(dogs_dir):
        img_path = os.path.join(dogs_dir, img_file)
        img = load_img(img_path, target_size=image_size)
        data.append(img_to_array(img))
        labels.append(1)
    
    data = np.array(data, dtype="float32") / 255.0
    labels = np.array(labels)
    return data, labels

def get_generators(batch_size=32):
    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
    
    return train_generator, val_generator
