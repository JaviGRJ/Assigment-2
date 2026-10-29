"""Reproducible benchmark for Assignment 2."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import platform
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

from methods import (
    Matrix,
    assert_close,
    count_nonzero,
    dense_blocked,
    dense_ijk,
    dense_ikj,
    sparse_csr,
    to_csr,
)


ROOT = Path(__file__).resolve().parent

DEFAULT_SIZES = [32, 64, 96, 128]
DEFAULT_DENSITIES = [0.01, 0.05, 0.10]

SEED = 20261028
REPETITIONS = 5
WARMUPS = 1

TIME_LIMIT_SECONDS = 60.0
MEMORY_LIMIT_MIB = 2048.0

NON_ZERO_VALUES = [
    float(value)
    for value in range(-8, 9)
    if value != 0
]


Method = Callable[[object, object], Matrix]


def make_matrix(
    rng: random.Random,
    n: int,
    density: float,
) -> Matrix:
    """Generate an n by n matrix with the requested non-zero probability."""
    matrix: Matrix = []

    for _ in range(n):
        row: list[float] = []

        for _ in range(n):
            if rng.random() < density:
                row.append(rng.choice(NON_ZERO_VALUES))
            else:
                row.append(0.0)

        matrix.append(row)

    return matrix


def calculate_iqr(values: list[float]) -> float:
    """Calculate the inclusive interquartile range."""
    if len(values) < 2:
        return 0.0

    quartiles = statistics.quantiles(
        values,
        n=4,
        method="inclusive",
    )

    return quartiles[2] - quartiles[0]


def run_method(
    method: Method,
    left: object,
    right: object,
    expected: Matrix,
    size: int,
    repetitions: int,
    warmups: int,
) -> tuple[list[float], float, str]:
    """Run, validate and measure one method configuration."""
    for _ in range(warmups):
        warmup_result = method(left, right)
        assert_close(warmup_result, expected)

    tracemalloc.reset_peak()

    samples: list[float] = []
    status = "ok"
    started = time.perf_counter()

    for _ in range(repetitions):
        sample_started = time.perf_counter()

        result = method(left, right)

        elapsed_seconds = time.perf_counter() - sample_started
        elapsed_milliseconds = elapsed_seconds * 1000.0

        assert_close(result, expected)
        samples.append(elapsed_milliseconds)

        total_seconds = time.perf_counter() - started

        if total_seconds > TIME_LIMIT_SECONDS:
            status = "timeout"
            break

    peak_bytes = tracemalloc.get_traced_memory()[1]
    peak_memory_mib = peak_bytes / (1024 * 1024)

    if peak_memory_mib > MEMORY_LIMIT_MIB:
        status = "memory_limit"

    return samples, peak_memory_mib, status


def append_measurements(
    rows: list[dict[str, object]],
    *,
    part: str,
    method_name: str,
    size: int,
    density: float,
    nnz_a: int,
    nnz_b: int,
    nnz_c: int,
    samples: list[float],
    peak_memory_mib: float,
    status: str,
) -> None:
    """Append raw samples to the result table."""
    sparsity = 1.0 - density

    for repetition, time_ms in enumerate(samples, start=1):
        rows.append(
            {
                "part": part,
                "method": method_name,
                "size": size,
                "density": f"{density:.2f}",
                "sparsity": f"{sparsity:.2f}",
                "nnz_a": nnz_a,
                "nnz_b": nnz_b,
                "nnz_c": nnz_c,
                "repetition": repetition,
                "time_ms": f"{time_ms:.6f}",
                "peak_memory_mib": f"{peak_memory_mib:.6f}",
                "status": status,
            }
        )


def write_results(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write all raw measurements to CSV."""
    fields = [
        "part",
        "method",
        "size",
        "density",
        "sparsity",
        "nnz_a",
        "nnz_b",
        "nnz_c",
        "repetition",
        "time_ms",
        "peak_memory_mib",
        "status",
    ]

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    """Write median, IQR and memory summary for each configuration."""
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}

    for row in rows:
        key = (
            str(row["part"]),
            str(row["method"]),
            str(row["size"]),
            str(row["density"]),
            str(row["sparsity"]),
            str(row["nnz_a"]),
            str(row["nnz_b"]),
            str(row["nnz_c"]),
        )

        grouped.setdefault(key, []).append(row)

    fields = [
        "part",
        "method",
        "size",
        "density",
        "sparsity",
        "nnz_a",
        "nnz_b",
        "nnz_c",
        "median_ms",
        "iqr_ms",
        "peak_memory_mib",
        "samples",
        "status",
    ]

    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()

        for key, group in grouped.items():
            successful_times = [
                float(row["time_ms"])
                for row in group
                if row["status"] == "ok"
            ]

            statuses = {str(row["status"]) for row in group}
            status = "ok" if statuses == {"ok"} else str(group[-1]["status"])

            writer.writerow(
                {
                    "part": key[0],
                    "method": key[1],
                    "size": key[2],
                    "density": key[3],
                    "sparsity": key[4],
                    "nnz_a": key[5],
                    "nnz_b": key[6],
                    "nnz_c": key[7],
                    "median_ms": (
                        f"{statistics.median(successful_times):.6f}"
                        if successful_times
                        else ""
                    ),
                    "iqr_ms": (
                        f"{calculate_iqr(successful_times):.6f}"
                        if len(successful_times) >= 2
                        else ""
                    ),
                    "peak_memory_mib": group[-1]["peak_memory_mib"],
                    "samples": f"{len(successful_times)}/{len(group)}",
                    "status": status,
                }
            )


def run_benchmark(
    sizes: list[int],
    densities: list[float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Run dense and sparse Assignment 2 experiments."""
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []

    dense_methods: list[tuple[str, Method]] = [
        ("baseline_ijk", dense_ijk),
        ("reordered_ikj", dense_ikj),
        ("blocked", dense_blocked),
    ]

    for size in sizes:
        matrix_a = make_matrix(rng, size, 1.0)
        matrix_b = make_matrix(rng, size, 1.0)

        expected = dense_ijk(matrix_a, matrix_b)

        nnz_a = count_nonzero(matrix_a)
        nnz_b = count_nonzero(matrix_b)
        nnz_c = count_nonzero(expected)

        for method_name, method in dense_methods:
            samples, peak_memory_mib, status = run_method(
                method,
                matrix_a,
                matrix_b,
                expected,
                size,
                REPETITIONS,
                WARMUPS,
            )

            append_measurements(
                rows,
                part="dense",
                method_name=method_name,
                size=size,
                density=1.0,
                nnz_a=nnz_a,
                nnz_b=nnz_b,
                nnz_c=nnz_c,
                samples=samples,
                peak_memory_mib=peak_memory_mib,
                status=status,
            )

    for density in densities:
        for size in sizes:
            matrix_a = make_matrix(rng, size, density)
            matrix_b = make_matrix(rng, size, density)

            expected = dense_ijk(matrix_a, matrix_b)

            csr_a = to_csr(matrix_a)
            csr_b = to_csr(matrix_b)

            nnz_a = count_nonzero(matrix_a)
            nnz_b = count_nonzero(matrix_b)
            nnz_c = count_nonzero(expected)

            dense_samples, dense_memory, dense_status = run_method(
                dense_ijk,
                matrix_a,
                matrix_b,
                expected,
                size,
                REPETITIONS,
                WARMUPS,
            )

            append_measurements(
                rows,
                part="sparse",
                method_name="dense_sparse_input",
                size=size,
                density=density,
                nnz_a=nnz_a,
                nnz_b=nnz_b,
                nnz_c=nnz_c,
                samples=dense_samples,
                peak_memory_mib=dense_memory,
                status=dense_status,
            )

            def run_sparse(
                left: object,
                right: object,
                matrix_size: int = size,
            ) -> Matrix:
                return sparse_csr(left, right, matrix_size)

            sparse_samples, sparse_memory, sparse_status = run_method(
                run_sparse,
                csr_a,
                csr_b,
                expected,
                size,
                REPETITIONS,
                WARMUPS,
            )

            append_measurements(
                rows,
                part="sparse",
                method_name="sparse_csr",
                size=size,
                density=density,
                nnz_a=nnz_a,
                nnz_b=nnz_b,
                nnz_c=nnz_c,
                samples=sparse_samples,
                peak_memory_mib=sparse_memory,
                status=sparse_status,
            )

    environment = {
        "assignment": 2,
        "run_timestamp": dt.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "python_version": sys.version,
        "operating_system": platform.platform(),
        "seed": SEED,
        "sizes": sizes,
        "densities": densities,
        "repetitions": REPETITIONS,
        "warmups": WARMUPS,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "memory_limit_mib": MEMORY_LIMIT_MIB,
        "memory_metric": "Python traced allocation peak via tracemalloc",
        "timing_boundary": (
            "multiplication call only; input generation, CSR conversion "
            "and validation excluded"
        ),
        "numeric_type": "Python float",
        "baseline_reused_from": "Assignment 1 dense i-j-k implementation",
    }

    return rows, environment


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
    )

    parser.add_argument(
        "--densities",
        nargs="+",
        type=float,
        default=DEFAULT_DENSITIES,
    )

    args = parser.parse_args()

    if any(size <= 0 for size in args.sizes):
        parser.error("all sizes must be positive")

    if any(not 0.0 < density <= 1.0 for density in args.densities):
        parser.error("densities must be greater than 0 and at most 1")

    tracemalloc.start()

    rows, environment = run_benchmark(
        args.sizes,
        args.densities,
    )

    write_results(ROOT / "results.csv", rows)
    write_summary(ROOT / "summary.csv", rows)

    (ROOT / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    summary_configurations = {
        (
            row["part"],
            row["method"],
            row["size"],
            row["density"],
        )
        for row in rows
    }

    print(
        json.dumps(
            {
                "raw_measurements": len(rows),
                "configurations": len(summary_configurations),
            }
        )
    )


if __name__ == "__main__":
    main()