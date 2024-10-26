import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory

# V2? augment by shifting around?


# is 128x64 enough resolution? sneezes are distinct enough, but maybe if against hi-hat?
# reduce batch_size if increasing resolution
def create_generators(train_dir="./spectrograms", img_size=(128, 64), batch_size=32):
    # Create training and validation datasets
    # this method auto-labels according to directory
    train_dataset = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_dataset = image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    # Prefetching for performance optimization... or, so I'm told
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset
