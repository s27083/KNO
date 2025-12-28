import tensorflow as tf
import numpy as np
import sys
import os
import argparse
from PIL import Image
import matplotlib.pyplot as plt

MODEL_PATH = "./model.keras"


def load_and_prepare_image(image_path):
    img = Image.open(image_path).convert("L").resize((28, 28))
    img_array = np.array(img)
    img_array = 255 - img_array  # czarno biala
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28)


def train_and_save_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=1)
    plot_learning_curve(history)
    model.evaluate(x_test, y_test)
    model.save(MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    return model


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("Model exists")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Model not found")
        return train_and_save_model()


def recognize_digit(model, image_path):
    img = load_and_prepare_image(image_path)
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions)
    print(f"Digit: {predicted_digit}")


def plot_learning_curve(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, "bo-", label="Dokładność treningowa")
    plt.plot(epochs, val_acc, "ro-", label="Dokładność walidacyjna")
    plt.title("Dokładność treningu i walidacji")
    plt.xlabel("Epoka")
    plt.ylabel("Dokładność")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, "bo-", label="Strata treningowa")
    plt.plot(epochs, val_loss, "ro-", label="Strata walidacyjna")
    plt.title("Strata treningu i walidacji")
    plt.xlabel("Epoka")
    plt.ylabel("Strata")
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="digit recognistion MNIST.")
    parser.add_argument("--image", type=str)
    args = parser.parse_args()

    model = load_or_train_model()

    if args.image:
        if not os.path.isfile(args.image):
            print(f"File not found {args.image}")
            sys.exit(1)
        recognize_digit(model, args.image)


if __name__ == "__main__":
    main()
