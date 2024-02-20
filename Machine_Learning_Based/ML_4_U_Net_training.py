import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras import layers, models

# Initial crop values (will be updated in the loop) (need to be 64)
crop_starting_row = 135
crop_ending_row = 199

crop_starting_column = 256
crop_ending_column = 320



# Data Generator for Loading Images and Masks
class ImageMaskGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=32, target_size=(64, 64)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.target_size = target_size
        self.image_names = [name for name in os.listdir(image_dir) if name.endswith('.npy')]

    def __len__(self):
        return len(self.image_names) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_names[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs = []
        masks = []
        for name in batch_images:
            img_path = os.path.join(self.image_dir, name)
            turb_mask_path = os.path.join(self.mask_dir, name.replace('.npy', '_turbul.npy'))
            lamina_mask_path = os.path.join(self.mask_dir, name.replace('.npy', '_lami.npy'))

            img = np.load(img_path)
            img = img[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
            turb_mask = self.load_and_binarize_mask(turb_mask_path)
            lamina_mask = self.load_and_binarize_mask(lamina_mask_path)

            # img = cv2.resize(img, self.target_size)
            # turb_mask = cv2.resize(turb_mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            # lamina_mask = cv2.resize(lamina_mask, self.target_size, interpolation=cv2.INTER_NEAREST)

            combined_mask = np.stack([turb_mask, lamina_mask], axis=-1)

            imgs.append(img[..., np.newaxis])
            masks.append(combined_mask)
        return np.array(imgs), np.array(masks)

    def load_and_binarize_mask(self, path):
        mask = np.load(path)
        mask[mask == 1] = 0
        mask[mask == 10] = 1
        mask = mask[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
        return mask


# U-Net Model Definition
def build_unet(input_shape=(64, 64, 1)):
    inputs = tf.keras.layers.Input(input_shape)

    # Encoder
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(2, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# Main Script
if __name__ == "__main__":
    # Adjust these paths and parameters as needed
    image_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/230920_164712/normalized"
    mask_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/230920_164712/normalized/masks"
    batch_size = 8
    target_size = (64, 64)

    # Initialize data generator
    train_gen = ImageMaskGenerator(image_dir, mask_dir, batch_size, target_size)

    # Build and compile the U-Net model
    model = build_unet((64, 64, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_gen, epochs=10, verbose=1)  # Adjust epochs as needed

    # Save the model for future use
    model.save('my_unet_model.h5')

    # Print the model summary to check if everything is alright
    model.summary()


