import tensorflow as tf
from keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model.evaluation.metrics import SparsePrecision, SparseRecall, SparseF1Score


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
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            SparseRecall(name="recall"),
            SparsePrecision(name="precision"),
            SparseF1Score(name="F1-score"),
        ],
    )

    model.summary()

    return model


def fit_cnn(model, train_ds, val_ds):

    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=1, 
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'best_cnn_model.keras',
        monitor='val_loss',
        save_best_only=True,
    )

    model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=4,
        callbacks=[early_stop, checkpoint],
    )
