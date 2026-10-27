"""Shared full-precision, upper-triangular QUBO text I/O."""

from pathlib import Path

import numpy as np


def read_xqx_instances(input_txt: str | Path) -> list[np.ndarray]:
    """Read 1-based sparse records, rejecting malformed or ambiguous instances."""
    tokens = iter(Path(input_txt).read_text(encoding="ascii").split())

    def take() -> str:
        try:
            return next(tokens)
        except StopIteration as error:
            raise ValueError("Unexpected end of QUBO input") from error

    count = int(take())
    if count < 1:
        raise ValueError("At least one QUBO instance is required")
    instances = []
    for instance_index in range(1, count + 1):
        size, records = int(take()), int(take())
        if not 1 <= size <= 30000 or not 0 <= records <= size * (size + 1) // 2:
            raise ValueError(f"Invalid header for instance {instance_index}")
        Q = np.zeros((size, size), dtype=np.float64)
        seen = set()
        for _ in range(records):
            i, j, value = int(take()), int(take()), float(take())
            if not 1 <= i <= j <= size or not np.isfinite(value):
                raise ValueError(f"Invalid coefficient in instance {instance_index}")
            if (i, j) in seen:
                raise ValueError(f"Duplicate coefficient in instance {instance_index}")
            seen.add((i, j))
            Q[i - 1, j - 1] = value
        instances.append(Q)
    if next(tokens, None) is not None:
        raise ValueError("Unexpected trailing QUBO records")
    return instances


def write_xqx_instances(instances: list[np.ndarray], output_txt: str | Path) -> None:
    """Save float64 QUBOs with no rescaling and 17 significant decimal digits.

    The first token is the instance count. Each instance starts with
    ``n_variables n_records``, followed by 1-based ``i j coefficient`` records.
    Off-diagonal entries contain complete pair contributions, counted once.
    Use xqx's --qubo flag to minimize this upper-triangular objective.
    """
    if not instances:
        raise ValueError("At least one QUBO instance is required")
    for Q in instances:
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1] or not 1 <= Q.shape[0] <= 30000:
            raise ValueError("Each QUBO must be square with 1 to 30000 variables")
        if not np.isfinite(Q).all():
            raise ValueError("QUBO coefficients must be finite")
        if np.any(np.tril(Q, k=-1)):
            raise ValueError("QUBO matrices must be upper triangular")

    with open(output_txt, "w", encoding="ascii", newline="\n") as output:
        output.write(f"{len(instances)}\n")
        for Q in instances:
            rows, cols = np.nonzero(Q)
            output.write(f"{Q.shape[0]} {len(rows)}\n")
            for i, j in zip(rows, cols):
                output.write(f"{i + 1} {j + 1} {Q[i, j]:.17g}\n")
