import tensorflow as tf
from tensorflow.keras import models, layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model.evaluation.metrics import SparsePrecision, SparseRecall, SparseF1Score


def apply_resnet(train_ds):
    model = tf.keras.applications.ResNet50(
        include_top = False,
        weights = "imagenet",
        input_shape = (224, 224, 3),
        name = "resnet50",
    )

    model.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = tf.keras.applications.resnet50.preprocess_input(inputs)

    x = model(x, training = False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation = "relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(len(train_ds.class_names), activation="softmax")(x)

    model = models.Model(inputs, outputs)

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

def fit_resnet(model, train_ds, val_ds):

    early_stop = EarlyStopping(
        monitor="val_loss", 
        patience=5, 
        restore_best_weights=True
    )
    
    checkpoint = ModelCheckpoint(
        'best_resnet_model.keras',
        monitor='val_loss',
        save_best_only=True,
    )

    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=30,
        callbacks=[early_stop, checkpoint],
    )
    
    return history
