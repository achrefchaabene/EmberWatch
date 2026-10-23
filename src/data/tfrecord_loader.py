"""Load the Next-Day Wildfire Spread TFRecord dataset.

Run from the project root:
    .venv\\Scripts\\python.exe -m src.data.tfrecord_loader
"""

from pathlib import Path
import sys

import tensorflow as tf

# Allow both supported launch styles:
#   python -m src.data.tfrecord_loader
#   python src/data/tfrecord_loader.py  (VS Code's Run Python File button)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import DATA_DIR


HEIGHT = 64
WIDTH = 64
INPUT_FEATURES = (
    "PrevFireMask",
    "elevation",
    "th",
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
TARGET_FEATURE = "FireMask"
ALL_FEATURES = INPUT_FEATURES + (TARGET_FEATURE,)

FEATURE_SPEC = {
    name: tf.io.FixedLenFeature([HEIGHT * WIDTH], tf.float32)
    for name in ALL_FEATURES
}


def record_files(split: str, data_dir: Path = DATA_DIR) -> list[Path]:
    """Return ordered TFRecord shards for ``train``, ``eval``, or ``test``."""
    files = sorted(data_dir.glob(f"next_day_wildfire_spread_{split}_*.tfrecord"))
    if not files:
        raise FileNotFoundError(
            f"No {split!r} TFRecord files found in {data_dir}. "
            "Check DATA_DIR in configs/paths.py."
        )
    return files


def parse_example(serialized_example: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Parse one record into ``X: (64, 64, 12)`` and ``y: (64, 64, 1)``."""
    values = tf.io.parse_single_example(serialized_example, FEATURE_SPEC)
    inputs = tf.stack([values[name] for name in INPUT_FEATURES], axis=-1)
    inputs = tf.reshape(inputs, (HEIGHT, WIDTH, len(INPUT_FEATURES)))
    target = tf.reshape(values[TARGET_FEATURE], (HEIGHT, WIDTH, 1))
    return inputs, target


def load_dataset(
    split: str,
    batch_size: int | None = None,
    shuffle: bool = False,
    data_dir: Path = DATA_DIR,
) -> tf.data.Dataset:
    """Build a performant TensorFlow dataset for a project split."""
    files = [str(path) for path in record_files(split, data_dir)]
    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1_024, reshuffle_each_iteration=True)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset.prefetch(tf.data.AUTOTUNE)


def load_preprocessed_dataset(
    split: str,
    batch_size: int = 8,
    shuffle: bool = False,
    data_dir: Path = DATA_DIR,
) -> tf.data.Dataset:
    """Return cleaned batches as ``(inputs, binary_target, valid_mask)``.

    The mask is supplied as Keras sample weights so raw target pixels marked
    ``-1`` do not contribute to loss or metrics.
    """
    from src.data.preprocessing import preprocess_inputs, preprocess_target

    dataset = load_dataset(split, batch_size=None, shuffle=False, data_dir=data_dir)

    def preprocess_example(inputs: tf.Tensor, target: tf.Tensor):
        binary_target, valid_mask = preprocess_target(target)
        return preprocess_inputs(inputs), binary_target, valid_mask

    dataset = dataset.map(preprocess_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1_024, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def describe_first_sample(split: str = "train") -> None:
    """Print paths, shapes, and fire-pixel counts for one example."""
    files = record_files(split)
    inputs, target = next(iter(load_dataset(split).take(1)))
    print(f"Split: {split}")
    print(f"Shards: {len(files)}")
    print(f"X shape: {inputs.shape}")
    print(f"y shape: {target.shape}")
    print(f"Previous-fire pixels: {int(tf.reduce_sum(inputs[..., 0]).numpy())}")
    print(f"Next-day-fire pixels: {int(tf.reduce_sum(target).numpy())}")


if __name__ == "__main__":
    describe_first_sample()
