"""Numerical correctness and malformed-input regression checks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from nnlab.data import DatasetManager
from nnlab.errors import PracticeError
from nnlab.model import NeuralNetwork
from nnlab.parallel import parallel


class CoreTests(unittest.TestCase):
    def test_gradients(self):
        """Analytic backprop must match central finite differences."""
        features = np.array([[0.2, -0.3], [0.7, 0.4]])
        targets = np.array([[1.0, 0.0], [0.0, 1.0]])
        for activation in ("sigmoid", "relu"):
            model = NeuralNetwork(2, 3, 2, activation)
            _, gradients = model.loss_gradients(features, targets)
            for parameter, expected in zip(
                (model.w1, model.b1, model.w2, model.b2), gradients
            ):
                for index in np.ndindex(parameter.shape):
                    value = parameter[index]
                    epsilon = 1e-6
                    parameter[index] = value + epsilon
                    plus = model.loss_gradients(features, targets)[0]
                    parameter[index] = value - epsilon
                    minus = model.loss_gradients(features, targets)[0]
                    parameter[index] = value
                    self.assertAlmostEqual(
                        (plus - minus) / (2 * epsilon), expected[index], places=6
                    )

    def test_train_only_scaling_and_roundtrip(self):
        """Test rows do not affect scaling, and serialization preserves output."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            path.write_text("x,y\n0,0\n1,0\n2,1\n100,1\n")
            manager = DatasetManager()
            manager.load_csv(path)
            train, test, targets, _ = manager.split_normalize(test_size=0.25)
            self.assertTrue(np.allclose(train.mean(axis=0), 0))
            model = NeuralNetwork(1, 3)
            model.scaler = manager.scaler
            model.train(train, targets, epochs=5)
            file = Path(directory) / "weights.json"
            model.save(file)
            restored = NeuralNetwork.load(file)
            self.assertTrue(
                np.array_equal(model.forward(test)[0], restored.forward(test)[0])
            )

    def test_reject_invalid(self):
        """Reject empty layers, mismatched targets and broken weight files."""
        with self.assertRaises(PracticeError):
            NeuralNetwork(0, 3)
        model = NeuralNetwork(2, 3)
        with self.assertRaises(PracticeError):
            model.train(np.ones((2, 2)), np.ones((2, 2)))
        with self.assertRaises(PracticeError):
            model.load("/no/such/model.json")

    def test_reader_failure_does_not_deadlock(self):
        """Reader failure must terminate threads and propagate a domain error."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.png"
            path.write_text("invalid image")
            with self.assertRaises(PracticeError):
                parallel([path], workers=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
