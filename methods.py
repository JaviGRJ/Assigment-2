from __future__ import annotations

from typing import TypeAlias

Matrix: TypeAlias = list[list[float]]
CSR: TypeAlias = tuple[list[float], list[int], list[int]]


def zeros(n: int) -> Matrix:
    return [[0.0] * n for _ in range(n)]


def dense_ijk(a: Matrix, b: Matrix) -> Matrix:
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


def dense_blocked(a: Matrix, b: Matrix, block_size: int = 16) -> Matrix:
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


def to_csr(matrix: Matrix) -> CSR:
    values: list[float] = []
    columns: list[int] = []
    row_ptr = [0]
    for row in matrix:
        for column, value in enumerate(row):
            if value != 0.0:
                values.append(value)
                columns.append(column)
        row_ptr.append(len(values))
    return values, columns, row_ptr


def csr_nnz(matrix: CSR) -> int:
    return len(matrix[0])


def sparse_csr(a: CSR, b: CSR, n: int) -> Matrix:
    a_values, a_columns, a_rows = a
    b_values, b_columns, b_rows = b
    b_by_row = [
        {column: b_values[index] for index, column in enumerate(b_columns[b_rows[k] : b_rows[k + 1]], b_rows[k])}
        for k in range(n)
    ]
    c = zeros(n)
    for i in range(n):
        for a_index in range(a_rows[i], a_rows[i + 1]):
            k = a_columns[a_index]
            aik = a_values[a_index]
            for j, bkj in b_by_row[k].items():
                c[i][j] += aik * bkj
    return c


def count_nonzero(matrix: Matrix) -> int:
    return sum(value != 0.0 for row in matrix for value in row)


def assert_close(actual: Matrix, expected: Matrix, tolerance: float = 1e-12) -> None:
    if len(actual) != len(expected):
        raise AssertionError("matrix dimensions differ")
    for i, (actual_row, expected_row) in enumerate(zip(actual, expected)):
        for j, (actual_value, expected_value) in enumerate(zip(actual_row, expected_row)):
            allowed = tolerance + tolerance * abs(expected_value)
            if abs(actual_value - expected_value) > allowed:
                raise AssertionError(
                    f"mismatch at ({i}, {j}): {actual_value} != {expected_value}"
                )


def self_test() -> None:
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert_close(dense_ijk(a, b), expected)
    assert_close(dense_ikj(a, b), expected)
    assert_close(dense_blocked(a, b, 1), expected)
    assert_close(sparse_csr(to_csr(a), to_csr(b), 2), expected)


if __name__ == "__main__":
    self_test()
    print("Assignment 2 method self-tests passed.")