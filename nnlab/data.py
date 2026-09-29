"""CSV loading and training-only scaling."""

import csv
from pathlib import Path

import numpy as np

from .errors import DataFileError, MismatchedDataError


class DatasetManager:
    """Load a numeric CSV with headers and a target in its last column."""

    def __init__(self):
        self.features = self.targets = None
        self.scaler = None

    def load_csv(self, path):
        """Return raw feature and target arrays after validating the CSV."""
        try:
            with Path(path).open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            if (
                len(rows) < 3
                or len(rows[0]) < 2
                or any(len(row) != len(rows[0]) for row in rows)
            ):
                raise ValueError(
                    "Нужны заголовок и минимум две строки одинаковой ширины."
                )
            values = np.asarray(rows[1:], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("NaN и бесконечность не допускаются.")
            self.features, self.targets = values[:, :-1], values[:, -1:]
            return self.features, self.targets
        except (OSError, ValueError) as exc:
            raise DataFileError(str(exc)) from exc

    def split_normalize(self, test_size=0.25, method="standard", seed=42):
        """Split first, fit scaling on train only, and return train/test arrays."""
        if (
            self.features is None
            or not 0 < test_size < 1
            or method not in ("standard", "minmax")
        ):
            raise MismatchedDataError(
                "Загрузите данные; доля test должна быть между 0 и 1."
            )
        order = np.random.default_rng(seed).permutation(len(self.features))
        count = max(1, min(len(order) - 1, int(len(order) * test_size)))
        test_idx, train_idx = order[:count], order[count:]
        train = self.features[train_idx]
        offset = train.mean(axis=0) if method == "standard" else train.min(axis=0)
        scale = train.std(axis=0) if method == "standard" else np.ptp(train, axis=0)
        scale[scale == 0] = 1
        self.scaler = {
            "method": method,
            "offset": offset.tolist(),
            "scale": scale.tolist(),
        }
        return (
            self.transform(train, self.scaler),
            self.transform(self.features[test_idx], self.scaler),
            self.targets[train_idx],
            self.targets[test_idx],
        )

    @staticmethod
    def transform(features, scaler):
        """Transform new features using the saved training statistics."""
        values = np.asarray(features, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != len(scaler["offset"])
            or not np.isfinite(values).all()
        ):
            raise MismatchedDataError("Неверные признаки для сохранённого scaler.")
        return (values - np.asarray(scaler["offset"])) / np.asarray(scaler["scale"])
