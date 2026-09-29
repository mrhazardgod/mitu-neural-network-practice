"""Interactive console for all required MLP operations."""

import argparse
import os
from pathlib import Path

import numpy as np

from .data import DatasetManager
from .errors import MismatchedDataError, PracticeError
from .model import NeuralNetwork


def demo(path, output):
    """Run deterministic training and check the saved model round trip."""
    data = DatasetManager()
    data.load_csv(path)
    train, test, target, expected = data.split_normalize()
    model = NeuralNetwork(train.shape[1], 8, activation="sigmoid")
    model.scaler = data.scaler
    print(f"Network created: {model.sizes}, activation={model.activation}")
    print(f"CSV rows={len(data.features)} train={len(train)} test={len(test)}")
    model.train(
        train, target, epochs=500, learning_rate=0.15, batch_size=16, verbose=True
    )
    predicted = model.predict(test)
    print("Test accuracy:", float(np.mean(predicted == expected)))
    print("First predictions:", predicted[:8].ravel().tolist())
    model.save(output)
    restored = NeuralNetwork.load(output)
    print(
        "Saved/reloaded predictions identical:",
        bool(np.array_equal(model.predict(test), restored.predict(test))),
    )
    print("Weights:", output)


def menu():
    """Read commands until exit; recover from invalid user data and files."""
    model, data, split = None, DatasetManager(), None
    while True:
        print(
            "\n1 Create  2 Load CSV  3 Train  4 Predict  5 Save  6 Load weights  7 Plot  0 Exit"
        )
        try:
            choice = input("> ").strip()
            if choice == "0":
                return
            if choice == "1":
                sizes = [int(v) for v in input("input hidden output: ").split()]
                if len(sizes) != 3:
                    raise MismatchedDataError("Укажите три размера.")
                model = NeuralNetwork(
                    *sizes, activation=input("sigmoid/relu: ").strip()
                )
                print("Created:", model.sizes)
            elif choice == "2":
                data.load_csv(input("CSV path: ").strip())
                split = data.split_normalize(
                    float(input("test fraction: ")), input("standard/minmax: ").strip()
                )
                print("Loaded train/test:", len(split[0]), len(split[1]))
            elif choice == "3":
                if model is None or split is None:
                    raise MismatchedDataError("Сначала создайте сеть и загрузите CSV.")
                rate, epochs, batch = input("learning_rate epochs batch: ").split()
                model.scaler = data.scaler
                model.train(
                    split[0],
                    split[2],
                    int(epochs),
                    float(rate),
                    int(batch),
                    verbose=True,
                )
                print("Test accuracy:", np.mean(model.predict(split[1]) == split[3]))
            elif choice == "4":
                if model is None or model.scaler is None:
                    raise MismatchedDataError("Нужна обученная сеть со scaler.")
                text = input(
                    "raw rows separated by ; or @path.csv (header, features only): "
                )
                values = (
                    np.loadtxt(text[1:], delimiter=",", skiprows=1, ndmin=2)
                    if text.startswith("@")
                    else np.asarray(
                        [[float(v) for v in row.split(",")] for row in text.split(";")]
                    )
                )
                transformed = data.transform(values, model.scaler)
                print("Probabilities:", model.forward(transformed)[0].tolist())
                print("Predictions:", model.predict(transformed).tolist())
            elif choice == "5":
                if model is None:
                    raise MismatchedDataError("Сначала создайте сеть.")
                model.save(input("output JSON: "))
                print("Saved")
            elif choice == "6":
                model = NeuralNetwork.load(input("input JSON: "))
                print("Loaded:", model.sizes)
            elif choice == "7":
                if model is None or not model.history:
                    raise MismatchedDataError("История обучения пуста.")
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                path = Path(input("output PNG: "))
                path.parent.mkdir(parents=True, exist_ok=True)
                plt.plot(model.history)
                plt.xlabel("Epoch")
                plt.ylabel("BCE")
                plt.tight_layout()
                plt.savefig(path)
                plt.close()
                print("Saved:", path)
            else:
                print("Неизвестная команда.")
        except (PracticeError, ValueError, OSError, ImportError) as exc:
            print("Ошибка:", exc)
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--path", default=os.getenv("DATA_PATH", "data/input/students.csv")
    )
    parser.add_argument(
        "--output", default=os.getenv("MODEL_PATH", "data/output/model.json")
    )
    args = parser.parse_args()
    if args.demo:
        try:
            demo(args.path, args.output)
        except PracticeError as exc:
            parser.exit(1, str(exc) + "\n")
    else:
        menu()
