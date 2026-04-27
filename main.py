import os

from loader.banana import init_data
from model.cnn import apply_cnn, fit_cnn
from model.rnn import apply_rnn, fit_rnn


def setup():
    train_ds, val_ds, test_ds = init_data(img_size=(224, 224), batch_size=32)

    cnn_model = apply_cnn(train_ds)
    fit_cnn(cnn_model, train_ds, val_ds)

    # rnn_model = apply_rnn(train_ds)
    # fit_rnn(rnn_model, train_ds, val_ds)


def main():
    print("Hello from banana!")
    setup()
    print("Banana setup complete!")


if __name__ == "__main__":
    main()
