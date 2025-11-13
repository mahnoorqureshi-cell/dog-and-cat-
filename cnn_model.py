from tensorflow.keras import models, layers, optimizers
from preprocessor import get_generators

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(1e-4), metrics=['accuracy'])
    return model

def train_and_save_model(epochs=15, batch_size=32):
    train_gen, val_gen = get_generators(batch_size)
    model = build_model()
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save("cats_dogs_cnn_kagglehub.h5")
    return model

if __name__ == "__main__":
    train_and_save_model()
