"""Basic dense baseline for Assignment 2."""

from __future__ import annotations

from typing import TypeAlias


Matrix: TypeAlias = list[list[float]]


def zeros(n: int) -> Matrix:
    return [[0.0] * n for _ in range(n)]


def dense_ijk(a: Matrix, b: Matrix) -> Matrix:
    """Basic dense matrix multiplication using the i-j-k loop order."""
    n = len(a)
    c = zeros(n)

    for i in range(n):
        for j in range(n):
            total = 0.0
            for k in range(n):
                total += a[i][k] * b[k][j]
            c[i][j] = total

    return c

def dense_ikj(a: Matrix, b: Matrix) -> Matrix:
    """Dense multiplication using the reordered i-k-j loop order."""
    n = len(a)
    c = zeros(n)

    for i in range(n):
        for k in range(n):
            aik = a[i][k]

            if aik == 0.0:
                continue

            for j in range(n):
                c[i][j] += aik * b[k][j]

    return c

def dense_blocked(
    a: Matrix,
    b: Matrix,
    block_size: int = 16,
) -> Matrix:
    """Dense multiplication using 16 by 16 cache-style blocks."""
    n = len(a)
    c = zeros(n)

    for ii in range(0, n, block_size):
        for kk in range(0, n, block_size):
            for jj in range(0, n, block_size):
                for i in range(ii, min(ii + block_size, n)):
                    for k in range(kk, min(kk + block_size, n)):
                        aik = a[i][k]

                        if aik == 0.0:
                            continue

                        for j in range(jj, min(jj + block_size, n)):
                            c[i][j] += aik * b[k][j]

    return c

def assert_close(
    actual: Matrix,
    expected: Matrix,
    tolerance: float = 1e-12,
) -> None:
    """Check two matrices using absolute and relative tolerance."""
    if len(actual) != len(expected):
        raise AssertionError("matrix dimensions differ")

    for i, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        if len(actual_row) != len(expected_row):
            raise AssertionError(f"row {i} dimensions differ")

        for j, (actual_value, expected_value) in enumerate(
            zip(actual_row, expected_row)
        ):
            allowed = tolerance + tolerance * abs(expected_value)

            if abs(actual_value - expected_value) > allowed:
                raise AssertionError(
                    f"mismatch at ({i}, {j}): "
                    f"{actual_value} != {expected_value}"
                )


def self_test() -> None:
    """Run small correctness tests."""
    a = [
        [1.0, 2.0],
        [3.0, 4.0],
    ]

    b = [
        [5.0, 6.0],
        [7.0, 8.0],
    ]

    expected = [
        [19.0, 22.0],
        [43.0, 50.0],
    ]

    result = dense_ijk(a, b)
    assert_close(result, expected)
    assert_close(dense_ikj(a, b), expected)
    assert_close(dense_blocked(a, b, block_size=1), expected)


if __name__ == "__main__":
    self_test()
    print("Assignment 2 baseline self-tests passed.")