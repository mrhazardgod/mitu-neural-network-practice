"""Recursive data preparation with CLI/ENV configuration and diagnostics."""

import argparse
import csv
import importlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import DataFileError, MismatchedDataError, PracticeError


def scan_files(directory, extensions):
    """Return deterministically sorted matching paths, including subdirectories."""
    root = Path(directory)
    if not root.is_dir():
        raise DataFileError(f"Каталог не найден: {root}")
    allowed = {"." + ext.lower().lstrip(".") for ext in extensions}
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in allowed
    )


def read_array(path, image_size=64):
    """Load a numeric CSV or normalized grayscale image, never pickle data."""
    try:
        if path.suffix.lower() == ".csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
                raise ValueError("Неверная прямоугольная CSV-таблица")
            result = np.asarray(rows[1:], dtype=np.float32)
        else:
            with Image.open(path) as image:
                result = (
                    np.asarray(
                        image.convert("L").resize((image_size, image_size)),
                        dtype=np.float32,
                    )
                    / 255
                )
        if not np.isfinite(result).all():
            raise ValueError("NaN или бесконечность")
        return result
    except (OSError, ValueError) as exc:
        raise DataFileError(f"{path}: {exc}") from exc


def prepare(path, extensions, output, image_size=64):
    """Combine compatible files into one numeric .npy and return statistics."""
    start = time.perf_counter()
    files = scan_files(path, extensions)
    if not files:
        raise DataFileError("Подходящие файлы не найдены.")
    kinds = {"csv" if f.suffix.lower() == ".csv" else "image" for f in files}
    if len(kinds) > 1:
        raise MismatchedDataError("CSV и изображения готовятся раздельно.")
    arrays = [read_array(f, image_size) for f in files]
    try:
        if kinds == {"csv"}:
            result = np.concatenate(arrays, axis=0)
            offset = result.min(axis=0)
            scale = np.ptp(result, axis=0)
            scale[scale == 0] = 1
            result = (result - offset) / scale
        else:
            result = np.stack(arrays)
    except ValueError as exc:
        raise MismatchedDataError("Несовместимая ширина CSV.") from exc
    output = Path(output)
    if output.suffix != ".npy":
        raise DataFileError("Выходной файл должен иметь расширение .npy")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, result, allow_pickle=False)
    except OSError as exc:
        raise DataFileError(str(exc)) from exc
    stats = dict(
        files=len(files),
        shape=list(result.shape),
        dtype=str(result.dtype),
        minimum=float(result.min()),
        maximum=float(result.max()),
        seconds=round(time.perf_counter() - start, 6),
        output=str(output),
    )
    print(stats)
    return stats


def doctor():
    """Report library versions and check NVIDIA availability with subprocess."""
    print("Python:", sys.version.split()[0], "Platform:", platform.platform())
    for name in ("numpy", "PIL", "matplotlib"):
        try:
            module = importlib.import_module(name)
            print(name, getattr(module, "__version__", "available"))
        except ImportError:
            print(name, "missing")
    print("TensorFlow/PyTorch: not required; MLP uses NumPy")
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            print(
                "NVIDIA:",
                result.stdout.strip()
                if result.returncode == 0
                else result.stderr.strip(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print("NVIDIA diagnostic failed:", exc)
    else:
        # An external command is still exercised on CPU-only systems.
        result = subprocess.run(
            [sys.executable, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        print("subprocess:", result.stdout.strip())
        print("CUDA: nvidia-smi not found; using CPU")
    print("Working directory:", Path.cwd())


def main():
    """Dispatch prepare and doctor subcommands, exiting cleanly on errors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--path", default=os.getenv("DATA_PATH", "data/input/images"))
    command.add_argument("--ext", nargs="+", default=["png", "jpg"])
    command.add_argument(
        "--output", default=os.getenv("OUTPUT_PATH", "data/output/prepared.npy")
    )
    command.add_argument("--size", type=int, default=64)
    sub.add_parser("doctor")
    args = parser.parse_args()
    try:
        logging.basicConfig(level=args.log_level.upper())
        if args.command == "doctor":
            doctor()
        elif args.size < 1:
            raise MismatchedDataError("Размер изображения должен быть положительным.")
        else:
            prepare(args.path, args.ext, args.output, args.size)
    except (PracticeError, ValueError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
