import tensorflow as tf
import numpy as np
from tensorflow import keras
import cv2


#Histogram equalisation for google street view images
def histogram_equalisation(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    equalised_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    resized_image = cv2.resize(equalised_image, (512, 512))
    return resized_image

# Image preprocessing for inference
def preprocess_input_image(image_path):
    """
    Histogram equalisation + normalisation + batching
    """
    image = histogram_equalisation(image_path)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

# Compute WWR
def compute_wwr(label_mask):
    vals, counts = np.unique(label_mask, return_counts=True)
    pixel_count = dict(zip(vals, counts))
    window = pixel_count.get(1, 0)
    balcony = pixel_count.get(4, 0)
    facade = pixel_count.get(2, 0)
    door = pixel_count.get(3, 0)
    full_facade = window + balcony + facade + door
    if full_facade == 0:
        return 0.0
    return (window + balcony) / full_facade

# MeanIoU custom object for model
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_cm = self.add_weight(
            name="total_confusion_matrix",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
            dtype=tf.float32
        )

    def reset_states(self):
        # Clears the confusion matrix state between epochs
        self.total_cm.assign(tf.zeros(shape=(self.num_classes, self.num_classes)))

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=-1)
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, weights=sample_weight)
        cm = tf.cast(cm, tf.float32)
        self.total_cm.assign_add(cm)

    def result(self):
        sum_over_row = tf.reduce_sum(self.total_cm, axis=0)
        sum_over_col = tf.reduce_sum(self.total_cm, axis=1)
        true_positives = tf.linalg.diag_part(self.total_cm)
        denominator = sum_over_row + sum_over_col - true_positives

        iou = tf.math.divide_no_nan(true_positives, denominator)
        return tf.reduce_mean(iou)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
        })
        return config

# Combined dice loss and categorical cross entropy loss function
def combined_dice_crossentropy(y_true, y_pred, alpha=0.4, beta=0.6):
    # Categorical cross entropy
    cce = categorical_crossentropy(y_true, y_pred)

    # Dice loss
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    dice_loss = 1 - (numerator + 1) / (denominator + 1)

    # Combined loss
    combined_loss = alpha * cce + beta * dice_loss

    return combined_loss

from tensorflow.keras.layers import Layer

# BilinearUpsampling layer for U-Net
class BilinearUpsampling(Layer):
    def __init__(self, target_size, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size, method='bilinear')

    def get_config(self):
        config = super(BilinearUpsampling, self).get_config()
        config.update({"target_size": self.target_size})
        return config

# Custom objects for U-Net loading
custom_unet_objects = {"MeanIoU":MeanIoU(num_classes=5), 
                    "combined_dice_crossentropy":combined_dice_crossentropy,
                    "BilinearUpsampling":BilinearUpsampling}

# Custom objects for FCN loading
custom_fcn_objects = {"MeanIoU":MeanIoU(num_classes=5), 
                    "combined_dice_crossentropy":combined_dice_crossentropy}