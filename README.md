# Assignment 2 — Optimised and Sparse Matrix Multiplication

## Description

This project extends the validated dense multiplication baseline from
Assignment 1. It compares:

- `dense_ijk`: basic dense multiplication;
- `dense_ikj`: reordered loop version;
- `dense_blocked`: blocked multiplication with blocks of `16 × 16`;
- `sparse_csr`: sparse multiplication using CSR.

The project is implemented in Python and uses no third-party packages.

## Requirements

- Python 3.10 or later.
- Windows or another system compatible with Python.
- No external dependencies.

## Execution

Run these commands from the `assignment-2` directory:

```powershell
python methods.py
python benchmark_assignment2.py
python generate_graphics.py
```

The first command runs the correctness tests.

The second command generates the inputs, validates the methods, performs five
measurements per configuration, measures time and memory, and creates the
result files.

The third command generates the figures.

## Experiment

Dense matrix sizes:

```text
32, 64, 96, 128
```

Sparse densities:

```text
1%, 5%, 10%
```

The corresponding sparsities are 99%, 95% and 90%.

The experiment uses:

- random seed `20261028`;
- Python `float` values;
- one warm-up;
- five repetitions;
- 60-second time limit;
- 2048 MiB memory limit.

The same logical sparse inputs are used by the dense and CSR methods.
Conversion to CSR is performed before timing and is excluded from the measured
multiplication time.

## Outputs

The benchmark creates:


results.csv
summary.csv
environment.json


`results.csv` contains the raw measurements. `summary.csv` contains the median,
IQR, memory and status for each configuration. `environment.json` records the
experiment settings.

The figure script creates:

runtime.svg
memory.svg


The figures show runtime and peak traced Python allocation.

## Reproducibility

The benchmark measures only the multiplication call. Input generation,
validation and CSR conversion are outside the timed region.

Every result is checked against the basic `dense_ijk` implementation using a
tolerance of `1e-12`. Sparse results also record `density`, `sparsity`,
`nnz_a`, `nnz_b` and `nnz_c`.

The `dense_ijk` baseline is based on the validated implementation from
Assignment 1. The Assignment 2 directory is self-contained and does not need
files from `assignment-1`.

## Materials

- `methods.py`: multiplication methods and correctness tests.
- `benchmark_assignment2.py`: benchmark and result generation.
- `generate_graphics.py`: figure generation.
- `REPORT.md`: report source.
- `Assignment02_Javier_Guerra_Report.pdf`: final report.

The course specification is:

Universidad de Las Palmas de Gran Canaria, *BIG DATA | Individual
Assignments: Matrix Multiplication*, Academic Year 2026–2027.

GitHub Copilot assisted with planning, implementation, debugging and report
drafting. The student remains responsible for reviewing and understanding the
work.
