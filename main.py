import os
import matplotlib.pyplot as plt

from loader.banana import init_data
from model.cnn import apply_cnn, fit_cnn
from model.rnn import apply_rnn, fit_rnn


def plot_history(history):

    # Plot training & Validation Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot training & Validation Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def setup():
    train_ds, val_ds, test_ds = init_data(img_size=(224, 224), batch_size=32)

    cnn_model = apply_cnn(train_ds)
    cnn_history = fit_cnn(cnn_model, train_ds, val_ds)
    plot_history(cnn_history)

    # rnn_model = apply_rnn(train_ds)
    # fit_rnn(rnn_model, train_ds, val_ds)

    return cnn_model, test_ds


def main():
    print("Hello from banana!")
    cnn_model, test_ds = setup()
    print("Banana setup complete!")


if __name__ == "__main__":
    main()
