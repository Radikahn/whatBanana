import os
import matplotlib.pyplot as plt

from loader.banana import init_data
from model.cnn import apply_cnn, fit_cnn
from model.rnn import apply_rnn, fit_rnn
from model.resnet50 import apply_resnet, fit_resnet
from test_model import test_model


def plot_history(history, model_name):

    # Plot training & Validation Loss
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_loss_history.png")
    plt.show()

    # Plot training & Validation Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name.lower()}_accuracy_history.png")
    plt.show()

def setup():
    train_ds, val_ds, test_ds = init_data(img_size=(224, 224), batch_size=32)

    #CNN
    print("Custom CNN Testing:")
    cnn_model = apply_cnn(train_ds)
    cnn_history = fit_cnn(cnn_model, train_ds, val_ds)
    plot_history(cnn_history, "CNN")

    #ResNet50
    print("ResNet50 Testing")
    resnet_model = apply_resnet(train_ds)
    resnet_history = fit_resnet(resnet_model, train_ds, val_ds)
    plot_history(resnet_history, "ResNet50")

    return cnn_model, resnet_model, test_ds


def main():
    print("Hello from banana!")
    cnn_model, resnet_model, test_ds = setup()
    print("Banana setup complete!")

    # Test CNN
    test_model("CNN", "best_cnn_model.keras")
    # Test ResNet50
    test_model("ResNet50", "best_resnet_model.keras")


if __name__ == "__main__":
    main()
