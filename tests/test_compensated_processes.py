"""Fresh interpreters retain expansion arithmetic across cache generations."""

import os
from pathlib import Path
import subprocess
import sys


WORKER = r"""
from decimal import Decimal, localcontext
import sys
import numpy as np
from numba import njit

# Vary the compilation order before importing the eagerly compiled gufuncs.
for index in range(int(sys.argv[1])):
    def shift(value):
        return value + 1.0
    assert njit(shift)(1.0) == 2.0

from blademaster_warp._compensated_kernels import array_sum, array_product, array_divide

with localcontext() as context:
    context.prec = 100
    a = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
    b = np.array([[3.0], [7.0]])
    al = np.full_like(a, 2.0**-57)
    bl = np.full_like(b, -2.0**-55)
    for kind, operation in [('sum', array_sum), ('product', array_product), ('divide', array_divide)]:
        high, low = operation.ufunc(a, al, b, bl) if kind == 'divide' else operation.ufunc(a, b)
        for i, j in np.ndindex(a.shape):
            x, y = Decimal.from_float(a[i, j]), Decimal.from_float(b[i, 0])
            if kind == 'sum':
                expected = x + y
            elif kind == 'product':
                expected = x * y
            else:
                expected = (x + Decimal.from_float(al[i, j])) / (y + Decimal.from_float(bl[i, 0]))
            actual = Decimal.from_float(high[i, j]) + Decimal.from_float(low[i, j])
            assert abs(actual - expected) < Decimal('1e-31'), (kind, actual, expected)
"""


def test_fresh_processes_after_partial_cache_eviction(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "cache"), PYTHONPATH=str(root))
    for shifts in [(0,), (3, 5, 7), (1, 4, 6)]:
        # Emulate an interrupted cache generation without modifying a live cache.
        for path in (tmp_path / "cache").rglob("_compensated_kernels.array_*"):
            path.unlink()
        workers = [
            subprocess.Popen(
                [sys.executable, "-X", "faulthandler", "-c", WORKER, str(count)],
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for count in shifts
        ]
        results = []
        for worker in workers:
            try:
                output, _ = worker.communicate(timeout=120)
            except subprocess.TimeoutExpired:
                worker.kill()
                output, _ = worker.communicate()
            results.append((worker.returncode, output))
        assert all(code == 0 for code, _ in results), results
