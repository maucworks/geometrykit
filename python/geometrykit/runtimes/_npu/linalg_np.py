# This file was generated with the assistance of an AI coding tool.
"""Banded linear algebra for grid problems (runtime `_npu`).

The plate/biharmonic operators that a minimising fill needs are *local*: on a
row-major grid the matrix is banded with bandwidth = the number of columns.
That makes a direct factorisation cheap and, more importantly, honest — the
biharmonic operator's condition number already grows like n^4, and forming
normal equations on top of it squares that. Nothing here squares anything: the
operator is assembled once and factorised once.

Contents:

- ``banded_cholesky`` / ``banded_solve`` — symmetric positive definite banded
  factorisation and solve (Cholesky, upper, bandwidth ``b``).
- ``Grid2D`` — a row-major index map with the spacing-aware difference stencils
  for a (possibly non-uniform) grid.
- ``dirichlet_solve`` — assemble "operator = rhs inside, value = datum on the
  boundary ring" as one banded SPD system and solve it.

The stencils are exact for the polynomials their order admits on the grid they
are built for: the second difference is exact for quadratics on any spacing and
for cubics on a uniform one, and the 13-point biharmonic inherits that. That is
what the verification leans on.
"""

from __future__ import annotations

from typing import Callable, Sequence

_EPS = 1e-14


class Grid2D:
    """Row-major index map and difference stencils for an (nx+1) x (ny+1) grid.

    ``nx``/``ny`` are the numbers of *intervals*, so there are ``(nx+1)*(ny+1)``
    points and the interior (unknown) points are those with 0 < i < nx and
    0 < j < ny. Spacings may be non-uniform (``xs``/``ys`` give the station
    coordinates).
    """

    def __init__(self, xs: Sequence[float], ys: Sequence[float]) -> None:
        self.xs = list(xs)
        self.ys = list(ys)
        self.nx = len(self.xs) - 1
        self.ny = len(self.ys) - 1
        self.n = (self.nx + 1) * (self.ny + 1)
        self.bandwidth = self.nx + 1  # a neighbour two columns away is (ny+1) apart

    def index(self, i: int, j: int) -> int:
        return i * (self.ny + 1) + j

    def coordinates(self, i: int, j: int) -> tuple:
        return self.xs[i], self.ys[j]

    def is_inside(self, i: int, j: int, margin: int = 1) -> bool:
        """Whether ``(i, j)`` is an unknown at ``margin`` rings in from the edge."""
        return margin <= i <= self.nx - margin and margin <= j <= self.ny - margin

    def second_difference_x(self, i: int, j: int) -> dict:
        """Pointwise three-point second difference along x (non-uniform).

        Exact for quadratics on any spacing. Note this operator is *not*
        symmetric on a non-uniform grid -- the symmetric (variational) form is
        what an energy minimisation needs, and on a uniform grid the two
        coincide. System solves here therefore use uniform grids.
        """
        left = self.xs[i] - self.xs[i - 1]
        right = self.xs[i + 1] - self.xs[i]
        total = left + right
        return {
            (i - 1, j): 2.0 / (total * left),
            (i, j): -2.0 / (total * right) - 2.0 / (total * left),
            (i + 1, j): 2.0 / (total * right),
        }

    def second_difference_y(self, i: int, j: int) -> dict:
        down = self.ys[j] - self.ys[j - 1]
        up = self.ys[j + 1] - self.ys[j]
        total = down + up
        return {
            (i, j - 1): 2.0 / (total * down),
            (i, j): -2.0 / (total * up) - 2.0 / (total * down),
            (i, j + 1): 2.0 / (total * up),
        }

    def mixed_difference(self, i: int, j: int) -> dict:
        """The four-point mixed second difference (equal to D_x D_y on a grid)."""
        left = self.xs[i] - self.xs[i - 1]
        right = self.xs[i + 1] - self.xs[i]
        down = self.ys[j] - self.ys[j - 1]
        up = self.ys[j + 1] - self.ys[j]
        scale = 1.0 / ((left + right) * (down + up))
        return {
            (i - 1, j - 1): scale,
            (i - 1, j + 1): -scale,
            (i + 1, j - 1): -scale,
            (i + 1, j + 1): scale,
        }

    def laplacian(self, i: int, j: int) -> dict:
        """The usual ``D_xx + D_yy`` — negative definite, so *not* usable as-is
        with `banded_cholesky`. Use `positive_laplacian` for a system solve."""
        stencil = self.second_difference_x(i, j)
        for key, value in self.second_difference_y(i, j).items():
            stencil[key] = stencil.get(key, 0.0) + value
        return stencil

    def positive_laplacian(self, i: int, j: int) -> dict:
        """``-(D_xx + D_yy)`` — the sign that makes the system positive definite."""
        return {key: -value for key, value in self.laplacian(i, j).items()}

    def biharmonic(self, i: int, j: int) -> dict:
        """Thin-plate operator ``(D_xx + D_yy)^2`` — the 13-point stencil.

        This is the Euler-Lagrange operator of the thin-plate energy
        ``|D_xx|^2 + 2|D_xy|^2 + |D_yy|^2``, assembled as one operator rather
        than as the normal equations of the individual difference rows.
        """
        parts = (self.second_difference_x, self.second_difference_y)
        stencil = {}
        for part_a in parts:
            for point_a, value_a in part_a(i, j).items():
                if not (0 <= point_a[0] <= self.nx and 0 <= point_a[1] <= self.ny):
                    continue
                if not (1 <= point_a[0] <= self.nx - 1 and 1 <= point_a[1] <= self.ny - 1):
                    # a second difference at a ring point would reach outside the
                    # grid; that contribution is dropped, which is why the
                    # unknowns must sit at least two rings in (see dirichlet_solve)
                    continue
                for part_b in parts:
                    for point_b, value_b in part_b(*point_a).items():
                        if not (0 <= point_b[0] <= self.nx and 0 <= point_b[1] <= self.ny):
                            continue
                        stencil[point_b] = stencil.get(point_b, 0.0) + value_a * value_b
        return stencil


def _as_band(rows: list, n: int, bandwidth: int) -> list:
    """Upper-band storage: ``ab[i][k]`` is ``A[i][i+k]`` for ``k = 0..bandwidth``."""
    band = [[0.0] * (bandwidth + 1) for _ in range(n)]
    for i, row in enumerate(rows):
        for j, value in row.items():
            if j < i:
                # the upper triangle is what the factorisation reads; taking
                # both would double every off-diagonal entry, since a stencil
                # assembly naturally produces both directions
                continue
            offset = j - i
            if offset > bandwidth:
                # dropping this silently would corrupt the matrix (it happened:
                # a two-step stencil on a reduced grid needs 2*stride+2, not
                # stride), so it is an error
                raise ValueError(
                    f"entry ({i},{j}) is {offset} off the diagonal, beyond the "
                    f"bandwidth {bandwidth} — compute it from the rows"
                )
            band[i][offset] += value
    return band


def banded_cholesky(rows: list, n: int, bandwidth: int) -> list:
    """Cholesky factor ``R`` (upper, bandwidth ``b``) of a symmetric banded SPD matrix.

    ``rows[i]`` is ``{j: value}`` for the non-zeros of row ``i``. Only the upper
    triangle is read, so the caller may pass full rows (a stencil assembly
    naturally yields both directions) **provided the operator is symmetric** —
    which the variational operators are and the pointwise non-uniform ones are
    not. Returns the band storage of ``R``: ``r[i][k] = R[i][i+k]``.
    """
    a = _as_band(rows, n, bandwidth)
    r = [[0.0] * (bandwidth + 1) for _ in range(n)]
    for j in range(n):
        lo = max(0, j - bandwidth)
        diagonal = a[j][0]
        for k in range(lo, j):
            diagonal -= r[k][j - k] * r[k][j - k]
        if diagonal <= _EPS:
            raise ValueError(
                f"matrix is not positive definite at row {j} (pivot {diagonal:.3e})"
            )
        r[j][0] = diagonal**0.5
        top = min(n - 1, j + bandwidth)
        for i in range(j + 1, top + 1):
            total = a[j][i - j]
            for k in range(lo, j):
                if i - k <= bandwidth:
                    total -= r[k][j - k] * r[k][i - k]
            r[j][i - j] = total / r[j][0]
    return r


def banded_solve(factor: list, rhs: Sequence[float], n: int, bandwidth: int) -> list:
    """Solve ``A x = rhs`` given the Cholesky factor from `banded_cholesky`."""
    y = [0.0] * n
    for i in range(n):  # forward: R^T y = rhs
        total = rhs[i]
        lo = max(0, i - bandwidth)
        for k in range(lo, i):
            total -= factor[k][i - k] * y[k]
        y[i] = total / factor[i][0]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):  # back: R x = y
        total = y[i]
        top = min(n - 1, i + bandwidth)
        for k in range(i + 1, top + 1):
            total -= factor[i][k - i] * x[k]
        x[i] = total / factor[i][0]
    return x


def dirichlet_solve(
    grid: Grid2D,
    operator: Callable[[int, int], dict],
    datum: Callable[[int, int], tuple],
    interior_rhs: Callable[[int, int], tuple] | None = None,
    rings: int = 1,
) -> list:
    """Solve an operator equation inside the grid, values fixed on ``rings`` rings.

    ``operator(i, j)`` gives the stencil of the differential operator at an
    unknown point; ``datum(i, j)`` gives the fixed value on the fixed rings;
    ``interior_rhs(i, j)`` (optional) is the right-hand side.

    The fixed points are *eliminated*, not turned into rows: their
    contribution moves to the right-hand side, which keeps the assembled
    matrix symmetric (and therefore usable by `banded_cholesky`). Adding
    "x = value" rows instead makes it asymmetric, because the operator rows
    still reference those columns.

    ``rings`` matters for the biharmonic: its stencil is a *composition* of
    second differences, so an unknown must sit at least two rings in for the
    composition to stay inside the grid. One ring of Dirichlet values plus a
    second ring of first cross-derivatives -- what the fill's G1 condition
    provides -- is therefore not just convenient, it is what makes the operator
    well posed without ghost points.

    Returns the solution as a list of ``(x, y, z)`` triples indexed by
    ``grid.index(i, j)`` — one system per component, each banded and SPD.
    """
    unknowns = [
        (i, j)
        for i in range(grid.nx + 1)
        for j in range(grid.ny + 1)
        if grid.is_inside(i, j, rings)
    ]
    index = {point: k for k, point in enumerate(unknowns)}
    n = len(unknowns)

    rows: list = [dict() for _ in range(n)]
    rhs: list = [[0.0, 0.0, 0.0] for _ in range(n)]
    for (i, j) in unknowns:
        k = index[(i, j)]
        for (ii, jj), coefficient in operator(i, j).items():
            if (ii, jj) in index:
                rows[k][index[(ii, jj)]] = rows[k].get(index[(ii, jj)], 0.0) + coefficient
            else:
                fixed = datum(ii, jj)
                for comp in range(3):
                    rhs[k][comp] -= coefficient * fixed[comp]
        if interior_rhs is not None:
            contribution = interior_rhs(i, j)
            for comp in range(3):
                rhs[k][comp] += contribution[comp]

    # the bandwidth follows from the assembled rows: a two-step stencil on a
    # reduced grid reaches 2*stride+2, not stride
    bandwidth = max(
        (j - i for i, row in enumerate(rows) for j in row if j > i), default=0
    )
    factor = banded_cholesky(rows, n, bandwidth)
    solution = [[0.0] * n for _ in range(3)]
    for comp in range(3):
        solution[comp] = banded_solve(
            factor, [rhs[k][comp] for k in range(n)], n, bandwidth
        )

    out = [None] * grid.n
    for k, (i, j) in enumerate(unknowns):
        out[grid.index(i, j)] = (
            solution[0][k],
            solution[1][k],
            solution[2][k],
        )
    for i in range(grid.nx + 1):
        for j in range(grid.ny + 1):
            if out[grid.index(i, j)] is None:
                out[grid.index(i, j)] = datum(i, j)
    return out
