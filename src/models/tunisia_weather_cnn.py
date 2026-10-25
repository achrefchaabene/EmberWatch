"""Small convolutional model for the first Tunisia ERA5-grid experiment."""

from __future__ import annotations

import tensorflow as tf


def build_tunisia_weather_cnn(input_shape: tuple[int, int, int]) -> tf.keras.Model:
    """Return a same-resolution probability model for a 32x18 ERA5 grid."""
    inputs = tf.keras.Input(shape=input_shape, name="tunisia_fire_weather_inputs")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal", name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", kernel_initializer="he_normal", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)
    x = tf.keras.layers.SpatialDropout2D(0.10, name="spatial_dropout")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", kernel_initializer="he_normal", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)
    outputs = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="fire_probability")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="tunisia_era5_weather_cnn")
