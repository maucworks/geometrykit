# This file was generated with the assistance of an AI coding tool.
"""M1 acceptance tests: the banded solver and the grid operators.

The load-bearing checks are analytic. A second difference is exact for
quadratics on any spacing and for cubics on a uniform one, and the 13-point
biharmonic inherits that (it is a product of second differences). So:

- impose a *cubic* on the boundary of a uniform grid and the interior must come
  back exactly for the biharmonic, and likewise the Laplacian;
- impose a *quadratic* on a non-uniform grid — the test that the stencils
  respect the spacing rather than assuming uniform;
- and the solver itself against a dense reference, because a factorisation that
  is merely close is not a factorisation.
"""

from __future__ import annotations

import random

import pytest
from geometrykit.runtimes._npu import linalg_np


def _uniform(nx: int, ny: int, step: float = 1.0):
    return linalg_np.Grid2D(
        [i * step for i in range(nx + 1)], [j * step for j in range(ny + 1)]
    )


def _non_uniform(nx: int, ny: int):
    xs = [0.0]
    for i in range(nx):
        xs.append(xs[-1] + 0.05 + 0.15 * ((i * 7) % 5) / 5.0)
    ys = [0.0]
    for j in range(ny):
        ys.append(ys[-1] + 0.04 + 0.2 * ((j * 3) % 4) / 4.0)
    return linalg_np.Grid2D(xs, ys)


class TestBandedSolver:
    """The factorisation, against a dense solve."""

    @staticmethod
    def _dense_solve(matrix, rhs):
        n = len(matrix)
        aug = [list(matrix[i]) + [rhs[i]] for i in range(n)]
        for col in range(n):
            piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
            aug[col], aug[piv] = aug[piv], aug[col]
            for r in range(n):
                if r == col or aug[r][col] == 0.0:
                    continue
                factor = aug[r][col] / aug[col][col]
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
        return [aug[i][n] / aug[i][i] for i in range(n)]

    def test_matches_a_dense_solve(self):
        random.seed(11)
        n, bandwidth = 24, 4
        dense = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dense[i][i] = 6.0 + i * 0.1
            for k in range(1, bandwidth + 1):
                if i + k < n:
                    value = random.uniform(-0.4, 0.4)
                    dense[i][i + k] = value
                    dense[i + k][i] = value
        rhs = [random.uniform(-1, 1) for _ in range(n)]
        rows = [
            {j: dense[i][j] for j in range(n) if dense[i][j] != 0.0} for i in range(n)
        ]
        factor = linalg_np.banded_cholesky(rows, n, bandwidth)
        got = linalg_np.banded_solve(factor, rhs, n, bandwidth)
        want = self._dense_solve(dense, rhs)
        assert max(abs(a - b) for a, b in zip(got, want)) < 1e-10

    def test_rejects_a_non_positive_definite_matrix(self):
        rows = [{0: 1.0, 1: 3.0}, {0: 3.0, 1: 1.0}]
        with pytest.raises(ValueError, match="not positive definite"):
            linalg_np.banded_cholesky(rows, 2, 1)


class TestGridOperators:
    """The stencils, on functions whose derivatives they must reproduce."""

    def test_second_difference_is_exact_for_cubics_on_a_uniform_grid(self):
        grid = _uniform(8, 8, 0.25)

        def u(x, y):
            return x**3 + 2.0 * y**3

        for i in (2, 4, 6):
            for j in (2, 4, 6):
                value = sum(
                    coefficient * u(*grid.coordinates(ii, jj))
                    for (ii, jj), coefficient in grid.second_difference_x(i, j).items()
                )
                assert value == pytest.approx(6.0 * grid.coordinates(i, j)[0], abs=1e-10)

    def test_second_difference_is_exact_for_quadratics_on_a_non_uniform_grid(self):
        grid = _non_uniform(8, 8)

        def u(x, y):
            return 3.0 * x**2 - 5.0 * y**2

        for i in (2, 4, 6):
            for j in (2, 4, 6):
                value = sum(
                    coefficient * u(*grid.coordinates(ii, jj))
                    for (ii, jj), coefficient in grid.second_difference_x(i, j).items()
                )
                assert value == pytest.approx(6.0, abs=1e-10)
                value = sum(
                    coefficient * u(*grid.coordinates(ii, jj))
                    for (ii, jj), coefficient in grid.second_difference_y(i, j).items()
                )
                assert value == pytest.approx(-10.0, abs=1e-10)

    def test_biharmonic_annihilates_a_cubic(self):
        """x^3*y is biharmonic; the 13-point stencil must see zero."""
        grid = _uniform(8, 8, 0.2)

        def u(x, y):
            return x**3 * y

        for i in (2, 4, 6):
            for j in (2, 4, 6):
                value = sum(
                    coefficient * u(*grid.coordinates(ii, jj))
                    for (ii, jj), coefficient in grid.biharmonic(i, j).items()
                )
                assert abs(value) < 1e-9, (i, j, value)


class TestDirichletSolve:
    """Whole systems, against functions the operators annihilate."""

    def test_laplacian_reproduces_a_harmonic_cubic(self):
        grid = _uniform(12, 12, 1.0 / 12.0)

        def u(x, y):
            return x**3 - 3.0 * x * y**2  # harmonic

        solution = linalg_np.dirichlet_solve(
            grid, grid.positive_laplacian, lambda i, j: (u(*grid.coordinates(i, j)), 0.0, 0.0)
        )
        worst = 0.0
        for i in range(grid.nx + 1):
            for j in range(grid.ny + 1):
                got = solution[grid.index(i, j)][0]
                worst = max(worst, abs(got - u(*grid.coordinates(i, j))))
        assert worst < 1e-10, worst

    def test_biharmonic_reproduces_a_cubic(self):
        """The plate case: the fill's operator, on data it should reproduce exactly."""
        grid = _uniform(12, 12, 1.0 / 12.0)

        def u(x, y):
            return x**3 * y + 0.5 * y**3

        solution = linalg_np.dirichlet_solve(
            grid,
            grid.biharmonic,
            lambda i, j: (u(*grid.coordinates(i, j)), 0.0, 0.0),
            rings=2,
        )
        worst = 0.0
        for i in range(grid.nx + 1):
            for j in range(grid.ny + 1):
                got = solution[grid.index(i, j)][0]
                worst = max(worst, abs(got - u(*grid.coordinates(i, j))))
        assert worst < 1e-10, worst

    def test_solve_is_fast_enough(self):
        import time

        grid = _uniform(16, 16, 1.0 / 16.0)
        start = time.perf_counter()
        linalg_np.dirichlet_solve(
            grid,
            grid.biharmonic,
            lambda i, j: (grid.coordinates(i, j)[0], 0.0, 0.0),
            rings=2,
        )
        assert time.perf_counter() - start < 2.0