"""A compact U-Net baseline for next-day wildfire spread segmentation."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    """Binary cross-entropy that gives fire pixels more influence on training."""

    def __init__(
        self,
        positive_weight: float = 70.0,
        reduction: str = "none",
        name: str = "weighted_bce",
    ):
        super().__init__(reduction=reduction, name=name)
        self.positive_weight = positive_weight

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        positive_loss = -self.positive_weight * y_true * tf.math.log(y_pred)
        negative_loss = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return positive_loss + negative_loss

    def get_config(self) -> dict:
        return {**super().get_config(), "positive_weight": self.positive_weight}


def _conv_block(inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Two convolution, normalization, and ReLU layers."""
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv1"
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(
        filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn2")(x)
    return tf.keras.layers.ReLU(name=f"{name}_relu2")(x)


def _up_block(inputs: tf.Tensor, skip: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Upsample, combine the matching encoder layer, then refine features."""
    x = tf.keras.layers.Conv2DTranspose(
        filters, 2, strides=2, padding="same", name=f"{name}_up"
    )(inputs)
    x = tf.keras.layers.Concatenate(name=f"{name}_concat")([x, skip])
    return _conv_block(x, filters, name)


def build_unet(
    input_shape: tuple[int, int, int] = (64, 64, 13),
    base_filters: int = 32,
) -> tf.keras.Model:
    """Build a four-level U-Net returning per-pixel fire probabilities."""
    inputs = tf.keras.Input(shape=input_shape, name="wildfire_inputs")

    encoder_1 = _conv_block(inputs, base_filters, "encoder1")
    pool_1 = tf.keras.layers.MaxPool2D(name="pool1")(encoder_1)

    encoder_2 = _conv_block(pool_1, base_filters * 2, "encoder2")
    pool_2 = tf.keras.layers.MaxPool2D(name="pool2")(encoder_2)

    encoder_3 = _conv_block(pool_2, base_filters * 4, "encoder3")
    pool_3 = tf.keras.layers.MaxPool2D(name="pool3")(encoder_3)

    bottleneck = _conv_block(pool_3, base_filters * 8, "bottleneck")

    decoder_3 = _up_block(bottleneck, encoder_3, base_filters * 4, "decoder3")
    decoder_2 = _up_block(decoder_3, encoder_2, base_filters * 2, "decoder2")
    decoder_1 = _up_block(decoder_2, encoder_1, base_filters, "decoder1")

    outputs = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid", name="fire_probability"
    )(decoder_1)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="wildfire_unet")


def compile_unet(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    positive_weight: float = 70.0,
) -> tf.keras.Model:
    """Configure the baseline with weighted binary cross-entropy support."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=WeightedBinaryCrossentropy(positive_weight=positive_weight),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="pixel_accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model
