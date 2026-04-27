import os

import tensorflow as tf


def init_data(img_size=(224, 224), batch_size=32):
    print("Loading dataset...")

    dataset_base = "Banana Ripeness Classification Dataset"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_base, "train"), image_size=img_size, batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_base, "valid"),
        image_size=img_size,
        batch_size=batch_size,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_base, "test"),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
    )

    print(train_ds.class_names)
    return train_ds, val_ds, test_ds
