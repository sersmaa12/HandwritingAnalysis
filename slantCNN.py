import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Гар бичвэрийн зургаа оруулах (замаа өөрөө тохируулна)
image_path = "D:\\Data\\Auth_002\\2.png"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Зургаа зөв зааж өгнө үү.")
 
# Хэмжээг тохируулах (CNN-д тохируулж resize, 128x128 recommended)
img_resized = cv2.resize(img, (128, 128))
img_input = img_resized.astype(np.float32) / 255.0  # normalize 0-1
img_input = img_input.reshape(1, 128, 128, 1)       # batch, height, width, channel

# CNN model: handwriting налуу prediction зориулсан
inputs = keras.Input(shape=(128, 128, 1))
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
slant_output = layers.Dense(1, activation='linear', name='slant_angle')(x)

model = keras.Model(inputs=inputs, outputs=slant_output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Модель сургаж байгаа тохиолдолд:
# model.fit(X_train, y_train, epochs=20)            # handwriting зураг, slant angle буюу градусын ground truth

# Зурган налуугийн хэмийг prediction хийх
pred_slant = model.predict(img_input)
print("Бичгийн налуугийн хэм (градус):", pred_slant[0][0])
