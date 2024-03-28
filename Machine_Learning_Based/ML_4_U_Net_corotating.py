import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean


# Initial crop values (will be updated in the loop) (need to be 64)
crop_starting_row = 0
crop_ending_row = 64

crop_starting_column = 260
crop_ending_column = 324


def load_dataset(image_dir, mask_dir, target_size=(64, 64)):
    image_names = [name for name in os.listdir(image_dir) if name.endswith('.npy')]

    image_dir_valida = image_dir + '/corotating_231207/glcm/validation/Validation old/statistics'
    image_names_valida = [name for name in os.listdir(image_dir_valida) if name.endswith('.npy')]

    X_train = []
    Y_train = []

    X_vali = []

# making train dataset
    for name in image_names:
        img_path = os.path.join(image_dir, name)
        turb_mask_path = os.path.join(mask_dir, name.replace('.npy', '_turbul.npy'))
        lamina_mask_path = os.path.join(mask_dir, name.replace('.npy', '_lami.npy'))

        # Load and preprocess the image and masks
        img = np.load(img_path)
        img = img[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]

        # print("shape", np.shape(img))

        turb_mask = np.load(turb_mask_path)
        turb_mask = turb_mask[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
        # print("shape turb_mask", np.shape(turb_mask))

        lamina_mask = np.load(lamina_mask_path)
        lamina_mask = lamina_mask[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]
        # print("shape lamina_mask", np.shape(lamina_mask))

        turb_mask[turb_mask == 1] = 0
        turb_mask[turb_mask == 10] = 1

        lamina_mask[lamina_mask == 1] = 0
        lamina_mask[lamina_mask == 10] = 1

        combined_mask = np.stack([turb_mask, lamina_mask], axis=-1)

        X_train.append(img[..., np.newaxis])  # Ensure img is 2D before adding new axis
        Y_train.append(combined_mask)

# making validation dataset
    for vali_name in image_names_valida:

        img_path_valida = os.path.join(image_dir_valida, vali_name)

        img_valida = np.load(img_path_valida)
        img_valida = img_valida[crop_starting_row:crop_ending_row, crop_starting_column:crop_ending_column]

        X_vali.append(img_valida[..., np.newaxis])


    return np.array(X_train), np.array(Y_train), np.array(X_vali), image_names_valida

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

def calculate_trans_position(predicted_mask):

    predicted_mask_single_channel = predicted_mask[..., 1]

    array_trans_posi = []
    for column in range(0, predicted_mask_single_channel.shape[1]):
        for row in range(1, predicted_mask_single_channel.shape[0]):
            if not predicted_mask_single_channel[row, column] == predicted_mask_single_channel[row - 1, column]:
                array_trans_posi.append(row)

    mean_trans_posi = mean(array_trans_posi) + crop_starting_row

    return mean_trans_posi


def display_cal_trans_predicted_mask(image, predicted_mask, index=0):
    # Display the original test image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Test Image at Index: {index}")
    plt.axis('on')

    # Display the predicted mask for a specific channel
    plt.subplot(1, 3, 2)
    predicted_mask_single_channel = predicted_mask[..., 1]  # Adjust the channel index if needed
    plt.imshow(predicted_mask_single_channel, cmap='gray')
    plt.title(f"Predicted Mask (Channel {index})")
    plt.axis('on')

    # Define the kernel size
    kernel_size = 2
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    plt.subplot(1, 3, 3)
    predicted_mask_single_channel_closed = predicted_mask[..., 1]
    # predicted_mask_single_channel_closed = cv2.morphologyEx(predicted_mask_single_channel_closed, cv2.MORPH_CLOSE, kernel)
    predicted_mask_single_channel_closed = cv2.dilate(predicted_mask_single_channel_closed, kernel, iterations=1)
    plt.imshow(predicted_mask_single_channel_closed, cmap='gray')
    plt.title(f"Predicted Mask closed (Channel {index})")
    plt.axis('on')


    plt.show()


# Main Script
if __name__ == "__main__":
    # Adjust these paths and parameters as needed
    image_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set"
    mask_dir = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/save_images/image_with_trans_line/new_data_set/230920_164712/Unet/normalized/masks"
    # Specify the path to the model file
    model_path = "D:/Academic/MSc/Thesis/Project files/Project Complete/data/new data/annotated two regions/dataset 4/statistics/Unet/R_54_118_C_260_324.h5"

    image_dir_valida = image_dir + '/corotating_231207/glcm/validation/Validation old/statistics'
    excel_file = "/trans_details - Unet-day4.xlsx"
    df_trans_details = pd.read_excel(image_dir_valida + excel_file)


    batch_size = 16
    target_size = (64, 64)

    X_train, Y_train, X_vali, image_names_valida = load_dataset(image_dir, mask_dir)


    # Build and compile the U-Net model
    model = build_unet((64, 64, 1))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    if os.path.exists(model_path):
        print("Model file found. Proceeding to load the model.")
        model = load_model(model_path)

    else:
        print("Model file not found. Preparing the model.")
        # Modelcheckpoint
        checkpointer = tf.keras.callbacks.ModelCheckpoint(model_path, verbose=1, save_best_only=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')]

        # train the model
        results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=callbacks)

        # # Save the model for future use
        model.save(model_path)

    # Print the model summary to check if everything is alright
    model.summary()

    # Assuming `model` is your trained U-Net model and `X_test` is loaded
    predicted_masks = model.predict(X_vali)

    # make a binary mask out of resulted predictions of pixel values from Unet model
    predicted_masks = (predicted_masks > 0.5).astype(np.uint8)

    # Display the test image and its predicted mask for the first image in the test set
    # display_cal_trans_predicted_mask(X_vali[2], predicted_masks[2])

    for i, image_name_valida in enumerate(image_names_valida):

        transitional_line_detected = calculate_trans_position(predicted_masks[i])

        # Find the index of the row with the specified image name to update the excel file with newly deteced transitional line
        row_index = df_trans_details[df_trans_details['image name'] == image_name_valida].index

        column_name_excel = f'{crop_starting_column}_{crop_ending_column}'

        if column_name_excel in df_trans_details.columns:
            # Find the index of the row with the specified image name
            row_index = df_trans_details[df_trans_details['image name'] == image_name_valida].index
            # Update the value in the specified column if the row exists
            if not row_index.empty:
                df_trans_details.at[row_index[0], column_name_excel] = transitional_line_detected
        else:
            # Update the value in the 'detected transitional line' column
            df_trans_details.at[row_index[0], column_name_excel] = transitional_line_detected

    # Save the updated DataFrame back to the Excel file
    df_trans_details.to_excel(image_dir_valida + excel_file, index=False)