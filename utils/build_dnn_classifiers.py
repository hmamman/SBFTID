import argparse
import sys
import os

from tensorflow import keras

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utils.ml_classifiers import dnn_classifier

# Get the absolute path to the directory where main.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
# Two levels up from expga.py
sys.path.append(os.path.join(base_path, "../"))

from utils import helpers


def train_and_evaluate(data, dataset_name):
    X, y, input_shape, nb_classes = data()

    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

    model = dnn_classifier(input_shape=input_shape)

    es = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
        restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=1000, validation_data=(X_val, y_val), callbacks=[es],verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

    print(f"{model.__class__.__name__} Model's accuracy on {dataset_name} dataset is: {accuracy:.4f}")

    # Save the model to disk
    dir_path = f"models/{dataset_name}"
    os.makedirs(dir_path, exist_ok=True)
    model_path = f'{dir_path}/dnn_standard_unfair.keras'
    model.save(model_path)
    print(f"DNN Model saved to {model_path}")
    #           Bank    Credit  Census  Compas  Meps
    #   DNN:    0.8994  0.7500  0.8428  0.8299  0.8759


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on a given dataset.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to use.")

    args = parser.parse_args()

    configs = helpers.get_config_dict()

    dataset_name = args.dataset_name.lower()

    data_path = f"data/{dataset_name}.py"
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    config = configs[dataset_name]

    data = helpers.get_data(config.dataset_name)

    train_and_evaluate(data, dataset_name)


if __name__ == "__main__":
    main()
