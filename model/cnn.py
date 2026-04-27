from tensorflow.keras import layers, models


def apply_cnn(train_ds):
    model = models.Sequential(
        [
            layers.Input(shape=(224, 224, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(len(train_ds.class_names), activation="softmax"),
        ]
    )

    # Compile
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    return model


def fit_cnn(model, train_ds, val_ds):
    model.fit(train_ds, validation_data=val_ds, epochs=3)
