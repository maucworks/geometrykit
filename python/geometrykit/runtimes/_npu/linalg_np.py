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


# ---------------------------------------------------------------------------
# The fill: a simply-supported plate on a uniform grid
# ---------------------------------------------------------------------------

_PLATE_STENCIL = (
    # (di, dj): coefficient of the 13-point thin-plate operator, h = 1
    (0, 0, 20.0),
    (1, 0, -8.0), (-1, 0, -8.0), (0, 1, -8.0), (0, -1, -8.0),
    (2, 0, 1.0), (-2, 0, 1.0), (0, 2, 1.0), (0, -2, 1.0),
    (1, 1, 2.0), (1, -1, 2.0), (-1, 1, 2.0), (-1, -1, 2.0),
)


def plate_fill(
    values: list,
    nx: int,
    ny: int,
    slopes: dict | None = None,
    curvatures: dict | None = None,
    claims: dict | None = None,
    fixed: set | None = None,
    steps: tuple = (1.0, 1.0),
) -> list:
    """Fill a grid interior with a minimising plate through its boundary data.

    ``values`` is a flat row-major list of ``(x, y, z)`` triples, ``(nx+1)``
    by ``(ny+1)``, with the boundary ring already set to the boundary data;
    the interior is replaced by the solution of the thin-plate equation.

    The biharmonic is fourth order, so boundary data alone leaves it
    under-determined -- one more condition per boundary station is needed.
    Closures, per side:

    - **simply supported** (the default): zero cross-curvature on the
      boundary, imposed by the reflection ``u_ghost = 2 u_edge - u_inner``.
      This is what makes the fill well posed with no ghost unknowns, and it
      is why a bilinear-plus-linear datum (``x*y``, which satisfies both
      conditions) is reproduced exactly;
    - **clamped** (a side with slope data in ``slopes``): the ghost is
      eliminated by the cross-derivative condition instead,
      ``u_ghost = u_inner -+ 2*step*slope`` -- the G1 condition as a boundary
      condition of the PDE rather than a correction on a finished surface.
      The relation is exact for affine slope data, so ``x*y`` and ``x**2``
      are reproduced exactly under it too.

    ``slopes`` (optional) maps a side name -- ``"west"``, ``"east"``,
    ``"south"``, ``"north"`` -- to a list of 3-vectors, one per station
    *along* that side (west/east: ``ny+1`` entries indexed by the second
    grid index; south/north: ``nx+1`` entries by the first). Each vector is
    the demanded cross-derivative *per unit grid parameter in the
    increasing-index direction* at that station; the ``-+`` sign above folds
    the side's orientation, so callers on the west/south sides pass the
    slope as-wanted, east/north sides pass it negated. Sides absent from the
    dict fall back to the simply-supported closure. ``steps`` is the uniform
    grid spacing ``(du, dv)``.

    With ``curvatures`` a side carries its *cross-curvature* too -- G2,
    still as a condition of the solve. The second derivative is
    direction-even, so no per-side orientation is folded. The 13-point
    stencil has one ghost per side, which the slope relation consumes -- a
    curvature condition cannot ride along there: value, slope and curvature
    together cost exactly one degree of freedom per boundary station, and
    pinning the first ring to the Hermite extrapolation is the *wrong* way
    to pay it (the plate loses its natural first ring, and the fill
    roughens by ~3x on flat ribbons, measured). The consistent scheme
    instead **eliminates the second ring inward**: the cross second
    difference across the first two spans
    ``(u_edge - 2 u_ring1 + u_ring2) / dv^2`` is set to the curvature data,
    so ``u_ring2 = 2 u_ring1 - u_edge + dv^2 * kappa``. The first inner
    ring stays a free unknown carrying the plate's own answer; ring 2
    becomes a linear function of it wherever referenced, and ring 2's own
    PDE rows are absorbed into their inner ring's rows by the reduction.
    The relation is exact for quadratics, so slope and curvature data taken
    from ``x**2`` reproduce it exactly. A side with curvatures needs slopes
    on that side. When two sides' elimination belts overlap (only on tiny
    grids), the west/east claims win.

    The claim mechanism is also available to callers directly: the
    ``slopes``/``curvatures`` data compile into *claims* -- substitution
    rules stating that one grid point is a linear function of others plus a
    constant. The assembly reduces the system by those claims (`P^T A P`
    over the substitution map). Claims may chain (a substituted point that
    references a substituted point) and may take the general form
    ``(weights, additive)`` -- that is how an interior derivative line (a
    G1 condition on a station line inside the domain, the loose fill's
    through-the-edge solve) enters the same precise reduction.

    Returns the filled grid with the boundary ring untouched.
    """
    stride = ny + 1
    du, dv = steps
    west = None if slopes is None else slopes.get("west")
    east = None if slopes is None else slopes.get("east")
    south = None if slopes is None else slopes.get("south")
    north = None if slopes is None else slopes.get("north")
    cw = None if curvatures is None else curvatures.get("west")
    ce = None if curvatures is None else curvatures.get("east")
    cs = None if curvatures is None else curvatures.get("south")
    cn = None if curvatures is None else curvatures.get("north")

    values = list(values)
    claimed: dict = {}

    if cw is not None and nx > 4:
        if west is None:
            raise ValueError("a curvature condition needs slope data on the side")
        for j in range(1, ny):
            claimed[(2, j)] = ((1, j), (0, j), cw[j], du * du)
    if ce is not None and nx > 4:
        if east is None:
            raise ValueError("a curvature condition needs slope data on the side")
        for j in range(1, ny):
            claimed[(nx - 2, j)] = ((nx - 1, j), (nx, j), ce[j], du * du)
    if cs is not None and ny > 4:
        if south is None:
            raise ValueError("a curvature condition needs slope data on the side")
        for i in range(1, nx):
            claimed.setdefault((i, 2), ((i, 1), (i, 0), cs[i], dv * dv))
    if cn is not None and ny > 4:
        if north is None:
            raise ValueError("a curvature condition needs slope data on the side")
        for i in range(1, nx):
            claimed.setdefault((i, ny - 2), ((i, ny - 1), (i, ny), cn[i], dv * dv))

    # -- the reduction --------------------------------------------------------
    # A point claimed by a curvature condition is a *linear function of its
    # first ring* (that is the curvature condition: the cross second
    # difference across the first two spans equals the data). The reduced
    # system is therefore P^T A P over the substitution operator P: in full
    # coordinates every point maps to itself, except a claimed point, which
    # maps to 2*inner - edge + step^2*kappa. Merely dropping the claimed
    # columns is not that reduction -- it loses the claimed rows' own
    # contributions and renders the reduced matrix indefinite (exactly what
    # the first elimination attempt hit, with the Hermite pin on ring 1).
    interior = [(i, j) for i in range(1, nx) for j in range(1, ny)]
    extra_fixed = set(fixed) if fixed is not None else set()
    reduced = [
        point for point in interior
        if point not in claimed and point not in extra_fixed
    ]
    index = {point: k for k, point in enumerate(reduced)}

    def _value(point):
        return values[point[0] * stride + point[1]]

    def _sum_terms(a: dict, scale_a: float, b: dict, scale_b: float) -> dict:
        out: dict = {}
        for point, value in a.items():
            out[point] = out.get(point, 0.0) + scale_a * value
        for point, value in b.items():
            out[point] = out.get(point, 0.0) + scale_b * value
        return out

    def reference(i: int, j: int):
        """In-grid terms (a dict) and a fixed constant of one grid reference."""
        if 0 <= i <= nx and 0 <= j <= ny:
            return {(i, j): 1.0}, ()
        # clamped sides: the ghost is eliminated by the slope relation
        if i < 0 and west is not None and 0 <= j <= ny:
            return {(1, j): 1.0}, tuple(-2.0 * du * c for c in west[j])
        if i > nx and east is not None and 0 <= j <= ny:
            return {(nx - 1, j): 1.0}, tuple(2.0 * du * c for c in east[j])
        if j < 0 and south is not None and 0 <= i <= nx:
            return {(i, 1): 1.0}, tuple(-2.0 * dv * c for c in south[i])
        if j > ny and north is not None and 0 <= i <= nx:
            return {(i, ny - 1): 1.0}, tuple(2.0 * dv * c for c in north[i])
        # no slope data for this side: the simply-supported reflection
        if i < 0:
            return _sum_terms(reference(0, j)[0], 2.0, reference(1, j)[0], -1.0), ()
        if i > nx:
            return (
                _sum_terms(reference(nx, j)[0], 2.0, reference(nx - 1, j)[0], -1.0),
                (),
            )
        if j < 0:
            return _sum_terms(reference(i, 0)[0], 2.0, reference(i, 1)[0], -1.0), ()
        return (
            _sum_terms(reference(i, ny)[0], 2.0, reference(i, ny - 1)[0], -1.0),
            (),
        )

    # full operator: rows for EVERY interior point (ghosts and reflections
    # as in the slope-only closure), fixed references separated out.
    full: dict = {}
    for (i, j) in interior:
        row: dict = {}
        const: list = [0.0, 0.0, 0.0]
        for (di, dj, coefficient) in _PLATE_STENCIL:
            terms, extra = reference(i + di, j + dj)
            for (ii, jj), w in terms.items():
                row[(ii, jj)] = row.get((ii, jj), 0.0) + coefficient * w
            if extra:
                for comp in range(3):
                    const[comp] += coefficient * extra[comp]
        full[(i, j)] = (row, const)

    def _decl(point):
        """(variable weights, additive constant) of one full coordinate.

        Claims compose: at corner quadrants an east/west belt's inner point
        can itself be claimed by the north/south belt, so the substitution
        resolves transitively until it lands on free unknowns only. Each
        level is exact for the datum, so the composition is exact too.
        """
        weights: dict = {}
        additive: list = [0.0, 0.0, 0.0]
        stack = [(point, 1.0)]
        while stack:
            p, w = stack.pop()
            claim = claimed.get(p)
            if claim is None:
                weights[p] = weights.get(p, 0.0) + w
                continue
            if isinstance(claim, tuple) and len(claim) == 4:
                # the curvature-belt form: 2*inner - edge + step^2*kappa
                inner, edge_point, kappa, step_sq = claim
                stack.append((inner, 2.0 * w))
                v = _value(edge_point)
                for comp in range(3):
                    additive[comp] += w * (-v[comp] + step_sq * kappa[comp])
            else:
                # the general form: (weights dict, additive 3-vector); nested
                # claimed points inside the weights resolve transitively
                sub_weights, sub_add = claim
                for q, weight in sub_weights.items():
                    stack.append((q, w * weight))
                for comp in range(3):
                    additive[comp] += w * sub_add[comp]
        return weights, (additive[0], additive[1], additive[2])

    n = len(reduced)
    rows: list = [dict() for _ in range(n)]
    rhs: list = [[0.0, 0.0, 0.0] for _ in range(n)]

    for r in interior:
        row, const = full[r]
        weights_r, _add_r = _decl(r)
        for a, alpha in weights_r.items():
            ta = index.get(a)
            for q, w in row.items():
                if q[0] == 0 or q[0] == nx or q[1] == 0 or q[1] == ny \
                        or q in extra_fixed:
                    # fixed: the boundary ring or an interior Dirichlet line
                    if ta is not None:
                        value = _value(q)
                        for comp in range(3):
                            rhs[ta][comp] -= alpha * w * value[comp]
                    continue
                weights_q, add_q = _decl(q)
                if ta is None:
                    continue
                for b, beta in weights_q.items():
                    tb = index.get(b)
                    if tb is not None:
                        rows[ta][tb] = rows[ta].get(tb, 0.0) + alpha * beta * w
                for comp in range(3):
                    rhs[ta][comp] -= alpha * w * add_q[comp]
            if any(const):
                for comp in range(3):
                    rhs[ta][comp] -= alpha * const[comp]

    bandwidth = max(
        (j - i for i, row in enumerate(rows) for j in row if j > i), default=0
    )
    factor = banded_cholesky(rows, n, bandwidth)
    solution = [banded_solve(factor, [rhs[k][c] for k in range(n)], n, bandwidth)
                for c in range(3)]

    out = list(values)
    for k, (i, j) in enumerate(reduced):
        out[i * stride + j] = (solution[0][k], solution[1][k], solution[2][k])
    for (i, j), claim in claimed.items():
        if isinstance(claim, tuple) and len(claim) == 4:
            inner, edge_point, kappa, step_sq = claim
            base = out[inner[0] * stride + inner[1]]
            edge = out[edge_point[0] * stride + edge_point[1]]
            out[i * stride + j] = (
                2.0 * base[0] - edge[0] + step_sq * kappa[0],
                2.0 * base[1] - edge[1] + step_sq * kappa[1],
                2.0 * base[2] - edge[2] + step_sq * kappa[2],
            )
        else:
            weights, additive = claim
            value = [additive[0], additive[1], additive[2]]
            for q, weight in weights.items():
                base = out[q[0] * stride + q[1]]
                for comp in range(3):
                    value[comp] += weight * base[comp]
            out[i * stride + j] = tuple(value)
    return out
