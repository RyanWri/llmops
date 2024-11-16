import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import TensorBoard
import datetime


def load_data():
    # Load MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = (
        datasets.mnist.load_data()
    )
    train_images, test_images = train_images / 255.0, test_images / 255.0
    # Expand dimensions for CNN
    train_images = train_images[..., tf.newaxis]
    test_images = test_images[..., tf.newaxis]
    return (train_images, train_labels), (test_images, test_labels)


def create_model():
    # Define a simple CNN model
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def train_model(model, train_data, test_data):
    # Set up TensorBoard for profiling
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(
        log_dir=log_dir, histogram_freq=1, profile_batch="10,15"
    )

    # Train the model
    model.fit(
        train_data[0],
        train_data[1],
        epochs=5,
        validation_data=test_data,
        callbacks=[tensorboard_callback],
    )

    return log_dir


if __name__ == "__main__":
    # Load data
    train_data, test_data = load_data()

    # Create model
    model = create_model()

    # Train model with profiling
    log_dir = train_model(model, train_data, test_data)

    print(f"Training complete. TensorBoard logs saved to: {log_dir}")
