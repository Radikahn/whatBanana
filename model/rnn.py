from tensorflow.keras import layers, models


def apply_rnn(train_ds):
    model = models.Sequential(
        [
            layers.Input(shape=(224, 224, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Reshape((224, 224 * 3)),
            layers.LSTM(128),
            layers.Dense(len(train_ds.class_names), activation="softmax"),
        ]
    )

    # Compile
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.summary()

    return model


def fit_rnn(model, train_ds, val_ds):
    model.fit(train_ds, validation_data=val_ds, epochs=3)
