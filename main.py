import os

def setup():
    print("Setting up banana...")
    import tensorflow as tf
    from tensorflow.keras import layers, models
    img_size = (224, 224)
    batch_size = 32
    dataset_base = "Banana Ripeness Classification Dataset"
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(dataset_base, "train"),
        image_size=img_size,
        batch_size=batch_size,
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

    #Simple CNN model 
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),

        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(train_ds.class_names), activation='softmax')
    ])

    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train (just a few epochs for testing)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )



        

def main():
    print("Hello from banana!")
    setup()
    print("Banana setup complete!")


if __name__ == "__main__":
    main()


