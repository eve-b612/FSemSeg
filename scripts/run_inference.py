import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import csv

from utils import (
    preprocess_input_image,
    compute_wwr,
    custom_unet_objects,  # or custom_fcn_objects if using FCN
    custom_fcn_objects
)
from download_model import download_model
# ---- Configuration ----
MODEL_PATH = download_model() 
#MODEL_PATH = "models/061224_unet.keras"
DATA_DIR = "test_data"
OUTPUT_DIR = "predictions"

USE_UNET = False  # Set to False to use FCN
MODEL_TYPE = "unet" if USE_UNET else "fcn"
MODEL_PATH = download_model(MODEL_TYPE)

NUM_IMAGES_TO_TEST = 5  # Set to an integer if you want to limit the number

def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    custom_objects = custom_unet_objects if USE_UNET else custom_fcn_objects
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
    print(f"Model {MODEL_PATH} loaded")

    image_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg"))]
    image_files = sorted(image_files)
    if NUM_IMAGES_TO_TEST:
        image_files = image_files[:NUM_IMAGES_TO_TEST]

    for filename in image_files:
        img_path = os.path.join(DATA_DIR, filename)
        base_name = os.path.splitext(filename)[0]

        # Preprocess
        input_img = preprocess_input_image(img_path)

        # Predict
        prediction = model.predict(input_img)
        predicted_mask = np.argmax(prediction, axis=-1)[0]

        # Compute WWR
        wwr = compute_wwr(predicted_mask)

        # Save predicted mask
        mask_path = os.path.join(OUTPUT_DIR, f"{base_name}_{MODEL_TYPE}_pred.png")
        normalized_mask = (predicted_mask.astype(np.uint8) * (255 // 5))
        cv2.imwrite(mask_path, normalized_mask)

        print(f"{filename} → WWR: {round(wwr, 3)} | Saved mask to {mask_path}")


    print(f"\nInference completed")

if __name__ == "__main__":
    run_inference()
