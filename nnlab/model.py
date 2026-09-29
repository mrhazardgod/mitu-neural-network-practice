"""One-hidden-layer MLP with binary cross entropy and SGD."""

import json
from pathlib import Path

import numpy as np

from .errors import DataFileError, InvalidLayerSizeError, MismatchedDataError


class NeuralNetwork:
    """Create an MLP with sigmoid outputs and sigmoid or ReLU hidden units."""

    def __init__(
        self, input_size, hidden_size, output_size=1, activation="sigmoid", seed=42
    ):
        sizes = (input_size, hidden_size, output_size)
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in sizes):
            raise InvalidLayerSizeError(
                "Размеры слоёв должны быть целыми положительными."
            )
        if activation not in ("sigmoid", "relu"):
            raise InvalidLayerSizeError("Активация: sigmoid или relu.")
        self.sizes, self.activation = sizes, activation
        self.rng = np.random.default_rng(seed)
        self.w1 = self.rng.normal(0, np.sqrt(2 / input_size), (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = self.rng.normal(
            0, np.sqrt(1 / hidden_size), (hidden_size, output_size)
        )
        self.b2 = np.zeros((1, output_size))
        self.history = []
        self.scaler = None

    @staticmethod
    def sigmoid(values):
        """Return stable sigmoid values for a numeric array."""
        return np.exp(-np.logaddexp(0, -values))

    def forward(self, features):
        """Return probabilities and the cache required by backpropagation."""
        features = np.asarray(features, dtype=float)
        if (
            features.ndim != 2
            or features.shape[1] != self.sizes[0]
            or not np.isfinite(features).all()
        ):
            raise MismatchedDataError("Неверная размерность или нечисловые признаки.")
        hidden_logits = features @ self.w1 + self.b1
        hidden = (
            self.sigmoid(hidden_logits)
            if self.activation == "sigmoid"
            else np.maximum(0, hidden_logits)
        )
        logits = hidden @ self.w2 + self.b2
        return self.sigmoid(logits), (features, hidden_logits, hidden, logits)

    def loss_gradients(self, features, targets):
        """Return mean binary cross entropy and gradients for all parameters."""
        probs, cache = self.forward(features)
        targets = np.asarray(targets, dtype=float)
        if (
            targets.shape != probs.shape
            or not len(targets)
            or not np.isfinite(targets).all()
            or np.any((targets < 0) | (targets > 1))
        ):
            raise MismatchedDataError(
                "Цель должна иметь размер (N, outputs) и значения от 0 до 1."
            )
        features, hidden_logits, hidden, logits = cache
        loss = float(np.mean(np.logaddexp(0, logits) - targets * logits))
        delta = (probs - targets) / targets.size
        grad_w2 = hidden.T @ delta
        grad_b2 = delta.sum(axis=0, keepdims=True)
        derivative = (
            hidden * (1 - hidden) if self.activation == "sigmoid" else hidden_logits > 0
        )
        delta_hidden = (delta @ self.w2.T) * derivative
        return loss, (
            features.T @ delta_hidden,
            delta_hidden.sum(axis=0, keepdims=True),
            grad_w2,
            grad_b2,
        )

    def train(
        self,
        features,
        targets,
        epochs=500,
        learning_rate=0.1,
        batch_size=16,
        verbose=False,
    ):
        """Train on minibatches; return accumulated BCE history."""
        if (
            not isinstance(epochs, int)
            or epochs < 1
            or not isinstance(batch_size, int)
            or batch_size < 1
            or not np.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise MismatchedDataError(
                "Нужны положительные epochs, batch_size и learning_rate."
            )
        self.loss_gradients(features, targets)
        for epoch in range(epochs):
            order = self.rng.permutation(len(features))
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                _, gradients = self.loss_gradients(features[indices], targets[indices])
                for parameter, gradient in zip(
                    (self.w1, self.b1, self.w2, self.b2), gradients
                ):
                    parameter -= learning_rate * gradient
            loss, _ = self.loss_gradients(features, targets)
            if not np.isfinite(loss):
                raise MismatchedDataError(
                    "Обучение расходится; уменьшите learning_rate."
                )
            self.history.append(loss)
            if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0):
                print(f"epoch={epoch + 1} BCE={loss:.6f}")
        return self.history

    def predict(self, features):
        """Return binary predictions for already transformed features."""
        return (self.forward(features)[0] >= 0.5).astype(int)

    def save(self, path):
        """Save weights, architecture, scaling parameters and history as JSON."""
        data = dict(
            sizes=self.sizes,
            activation=self.activation,
            history=self.history,
            scaler=self.scaler,
        )
        data.update(
            {key: getattr(self, key).tolist() for key in ("w1", "b1", "w2", "b2")}
        )
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        except (OSError, ValueError) as exc:
            raise DataFileError(str(exc)) from exc

    @classmethod
    def load(cls, path):
        """Validate a JSON file and restore a prediction-ready model."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            model = cls(*data["sizes"], activation=data["activation"])
            for key in ("w1", "b1", "w2", "b2"):
                array = np.asarray(data[key], dtype=float)
                if (
                    array.shape != getattr(model, key).shape
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("Повреждённые веса: " + key)
                setattr(model, key, array)
            model.history = data.get("history", [])
            model.scaler = data.get("scaler")
            if model.scaler:
                for key in ("offset", "scale"):
                    arr = np.asarray(model.scaler[key], dtype=float)
                    if (
                        arr.shape != (model.sizes[0],)
                        or not np.isfinite(arr).all()
                        or (key == "scale" and np.any(arr <= 0))
                    ):
                        raise ValueError("Повреждённый scaler")
            return model
        except (OSError, ValueError, KeyError, TypeError, InvalidLayerSizeError) as exc:
            raise DataFileError(str(exc)) from exc
