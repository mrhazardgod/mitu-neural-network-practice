"""Bounded producer/consumer pipeline and spawned-process augmentation."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .errors import DataFileError, PracticeError
from .prepare import read_array, scan_files


def augment(task):
    """Return original, rotation, reflection and deterministic noise variants."""
    index, array = task
    noise = np.random.default_rng(42 + index).normal(0, 0.02, array.shape)
    return index, np.stack(
        [array, np.rot90(array), np.fliplr(array), np.clip(array + noise, 0, 1)]
    ).astype(np.float32)


def sequential(files):
    """Read and augment the same sorted images without concurrency."""
    return np.concatenate([augment((i, read_array(p)))[1] for i, p in enumerate(files)])


def parallel(files, workers=2):
    """Use one producer, reader threads and a process pool; propagate failures."""
    if workers < 1:
        raise DataFileError("workers must be positive")
    paths, arrays = queue.Queue(maxsize=workers * 2), queue.Queue(maxsize=workers * 2)
    lock = threading.Lock()
    state = {"read": 0, "callbacks": 0}
    errors = []
    results = []

    def producer():
        for item in enumerate(files):
            paths.put(item)
        for _ in range(workers):
            paths.put(None)

    def consumer():
        while True:
            item = paths.get()
            if item is None:
                arrays.put(None)
                return
            index, path = item
            try:
                array = read_array(path)
                with lock:
                    state["read"] += 1
                logging.info("read %s", path.name)
                arrays.put((index, array, None))
            except (PracticeError, OSError, ValueError) as exc:
                arrays.put((index, None, str(exc)))

    def finished(_):
        with lock:
            state["callbacks"] += 1

    readers = [
        threading.Thread(target=consumer, name=f"reader-{i}") for i in range(workers)
    ]
    production = threading.Thread(target=producer, name="producer")
    with mp.get_context("spawn").Pool(workers) as pool:
        production.start()
        for thread in readers:
            thread.start()
        pending = []
        closed = 0
        while closed < workers:
            item = arrays.get()
            if item is None:
                closed += 1
                continue
            index, array, error = item
            if error:
                errors.append(error)
                continue
            pending.append(
                pool.apply_async(augment, ((index, array),), callback=finished)
            )
            if len(pending) >= workers * 2:
                results.append(pending.pop(0).get())
        production.join()
        for thread in readers:
            thread.join()
        for job in pending:
            results.append(job.get())
        pool.close()
        pool.join()
    if errors:
        raise DataFileError("; ".join(errors))
    results.sort(key=lambda item: item[0])
    return np.concatenate([values for _, values in results]), state


def main():
    """Measure equal workloads and save results plus augmented arrays."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=os.getenv("DATA_PATH", "data/input/images"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="data/output/performance_report.json")
    args = parser.parse_args()
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"), format="%(threadName)s %(message)s"
    )
    try:
        files = scan_files(args.path, ["png", "jpg", "jpeg"])
        if not files:
            raise DataFileError("Нет изображений.")
        start = time.perf_counter()
        seq = sequential(files)
        seq_time = time.perf_counter() - start
        start = time.perf_counter()
        par, state = parallel(files, args.workers)
        par_time = time.perf_counter() - start
        report = {
            "files": len(files),
            "augmented_items": len(par),
            "sequential_seconds": seq_time,
            "parallel_seconds": par_time,
            "speedup": seq_time / par_time,
            "equal_results": bool(np.array_equal(seq, par)),
            "reader_count": state["read"],
            "callback_count": state["callbacks"],
            "workers": args.workers,
            "note": "Small workloads may be slower because spawning and IPC dominate.",
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        np.save(output.with_name("augmented.npy"), par, allow_pickle=False)
        print(json.dumps(report, indent=2))
    except (PracticeError, OSError, ValueError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
