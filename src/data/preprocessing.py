"""Cleaning and normalization for Next-Day Wildfire Spread inputs.

The constants were estimated from the first 200 train records. Recompute them
from every training record before the final experiment.
"""

import tensorflow as tf

from src.data.tfrecord_loader import INPUT_FEATURES


CHANNELS = (
    "PrevFireMask",
    "elevation",
    "th_sin",
    "th_cos",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "erc",
    "NDVI",
    "population",
)

STATS = {
    "elevation": {"median": 590.0, "mean": 862.7342, "std": 819.9611},
    "vs": {"median": 3.2487, "mean": 3.4181, "std": 1.1903},
    "tmmn": {"median": 283.5952, "mean": 282.4662, "std": 7.8350},
    "tmmx": {"median": 299.8682, "mean": 298.1991, "std": 8.0682},
    "sph": {"median": 0.0054, "mean": 0.0063, "std": 0.0036},
    "pr": {"median": 0.0, "mean": 0.1257, "std": 0.4189},
    "pdsi": {"median": -1.5647, "mean": -0.8873, "std": 2.3871},
    "erc": {"median": 50.6618, "mean": 53.4764, "std": 25.9141},
    "NDVI": {"median": 5384.0, "mean": 5236.1074, "std": 2217.2207},
    "population": {"median": 0.1646, "mean": 0.9293, "std": 1.4090},
    "th_sin": {"mean": -0.1063, "std": 0.7464},
    "th_cos": {"mean": -0.4118, "std": 0.5119},
}


def _channel(inputs: tf.Tensor, name: str) -> tf.Tensor:
    return inputs[..., INPUT_FEATURES.index(name)]


def _replace_and_standardize(
    values: tf.Tensor,
    valid: tf.Tensor,
    name: str,
    log_transform: bool = False,
) -> tf.Tensor:
    """Impute invalid values, optionally apply log1p, then z-normalize."""
    stats = STATS[name]
    clean = tf.where(valid, values, tf.cast(stats["median"], values.dtype))
    if log_transform:
        clean = tf.math.log1p(clean)
    return (clean - stats["mean"]) / stats["std"]


def preprocess_inputs(inputs: tf.Tensor) -> tf.Tensor:
    """Convert raw ``(..., 12)`` inputs to cleaned ``(..., 13)`` inputs."""
    previous_fire = tf.cast(tf.equal(_channel(inputs, "PrevFireMask"), 1), tf.float32)

    elevation = _channel(inputs, "elevation")
    elevation = _replace_and_standardize(
        elevation, tf.math.is_finite(elevation), "elevation"
    )

    direction = _channel(inputs, "th")
    valid_direction = (direction >= 0) & (direction < 360)
    # Circular mean of valid directions in the sampled train records.
    direction = tf.where(valid_direction, direction, tf.cast(194.5, direction.dtype))
    direction = direction * (tf.constant(3.141592653589793, direction.dtype) / 180.0)
    direction_sin = (tf.sin(direction) - STATS["th_sin"]["mean"]) / STATS["th_sin"]["std"]
    direction_cos = (tf.cos(direction) - STATS["th_cos"]["mean"]) / STATS["th_cos"]["std"]

    vs = _channel(inputs, "vs")
    vs = _replace_and_standardize(vs, vs >= 0, "vs")

    tmmn = _channel(inputs, "tmmn")
    tmmn = _replace_and_standardize(tmmn, tmmn > 200, "tmmn")

    tmmx = _channel(inputs, "tmmx")
    tmmx = _replace_and_standardize(tmmx, tmmx > 200, "tmmx")

    sph = _channel(inputs, "sph")
    sph = _replace_and_standardize(sph, sph >= 0, "sph")

    precipitation = _channel(inputs, "pr")
    precipitation = _replace_and_standardize(
        precipitation, precipitation >= 0, "pr", log_transform=True
    )

    pdsi = _channel(inputs, "pdsi")
    pdsi = _replace_and_standardize(pdsi, tf.math.is_finite(pdsi), "pdsi")

    erc = _channel(inputs, "erc")
    erc = _replace_and_standardize(erc, erc >= 0, "erc")

    ndvi = _channel(inputs, "NDVI")
    ndvi = _replace_and_standardize(ndvi, tf.math.is_finite(ndvi), "NDVI")

    population = _channel(inputs, "population")
    population = _replace_and_standardize(
        population, population >= 0, "population", log_transform=True
    )

    return tf.stack(
        (
            previous_fire,
            elevation,
            direction_sin,
            direction_cos,
            vs,
            tmmn,
            tmmx,
            sph,
            precipitation,
            pdsi,
            erc,
            ndvi,
            population,
        ),
        axis=-1,
    )


def preprocess_target(target: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Return binary target and a mask that excludes the raw -1 unknown pixels."""
    valid_mask = tf.cast(tf.not_equal(target, -1), tf.float32)
    binary_target = tf.cast(tf.equal(target, 1), tf.float32)
    return binary_target, valid_mask
