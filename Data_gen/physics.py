"""Axisymmetric linear-elasticity FEM stress solver and zone-based life model.

Stress is obtained from a real 2D axisymmetric linear-elasticity finite-element
solve (scikit-fem, quadratic ``ElementTriP2`` triangles) of the disc meridional
cross-section under centrifugal body load.  Life is computed from a Palmgren-Miner
accumulation over the flight cycle using piecewise log-log (knee-based) S-N laws,
selectable per zone (``lifing_mode="zonal"``) or uniform across the disc
(``lifing_mode="uniform"``).  No analytical stress multipliers remain.
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP2,
    ElementVector,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import grad

from .config import (
    CYCLE_PHASE_WEIGHTS,
    CYCLE_SPEED_FACTORS,
    UNIFORM_SN_PARAMS,
    ZONAL_SN_PARAMS,
    ZONE_ID_TO_NAME,
)

logger = logging.getLogger(__name__)

# --- Material: Ti-6Al-4V -----------------------------------------------------
E_MODULUS_MPA = 114e3      # Young's modulus [MPa]
POISSON_RATIO = 0.34       # Poisson's ratio [-]
DENSITY_KG_M3 = 4430.0     # density [kg/m^3]

# --- Operating point ---------------------------------------------------------
OMEGA_REF_RAD_S = 3000.0   # takeoff rotational speed [rad/s]

EPS = 1e-6

# Zone names ordered by zone id (0..4) for fast vectorised parameter lookup.
_ZONE_NAMES_BY_ID = [ZONE_ID_TO_NAME[i] for i in range(len(ZONE_ID_TO_NAME))]


def _axisymmetric_C() -> np.ndarray:
    """4x4 isotropic elasticity matrix for axisymmetric stress/strain order
    [xx, rr, tt(hoop), xr].  Units: MPa (since E is in MPa)."""
    E = E_MODULUS_MPA
    nu = POISSON_RATIO
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    C = np.array(
        [
            [lam + 2.0 * mu, lam, lam, 0.0],
            [lam, lam + 2.0 * mu, lam, 0.0],
            [lam, lam, lam + 2.0 * mu, 0.0],
            [0.0, 0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )
    return C


def _strain_components(u, w):
    """Return axisymmetric strain components for a vector field.

    Coordinate 0 = x (axial), coordinate 1 = r (radial).
    Returns (e_xx, e_rr, e_tt, e_xr) where e_xr is the engineering shear γ_xr.
    """
    # u.value[0] = u_x, u.value[1] = u_r ; gradients via skfem helper.
    du = grad(u)  # shape (2, 2, nelems, nqp): du[i, j] = d(u_i)/d(x_j)
    r = w.x[1]
    r_safe = np.where(np.abs(r) < EPS, EPS, r)
    e_xx = du[0, 0]
    e_rr = du[1, 1]
    e_tt = u.value[1] / r_safe
    e_xr = du[0, 1] + du[1, 0]  # engineering shear
    return e_xx, e_rr, e_tt, e_xr


def _assemble_and_solve(mesh: MeshTri, omega: float):
    """Assemble and solve the axisymmetric elasticity system at angular speed omega.

    Coordinates of ``mesh`` are expected in METRES.  Returns nodal stress
    components (sxx, srr, stt, sxr) in Pa together with the global basis used.
    """
    C = _axisymmetric_C() * 1e6  # MPa -> Pa for a fully SI solve

    element = ElementVector(ElementTriP2())
    basis = Basis(mesh, element)

    @BilinearForm
    def stiffness(u, v, w):
        eu = _strain_components(u, w)
        ev = _strain_components(v, w)
        # sigma = C : eu  ; integrand = ev : sigma, integrated with axisymmetric
        # measure 2*pi*r dA (the constant 2*pi cancels and is dropped).
        r = w.x[1]
        sig = [sum(C[i, j] * eu[j] for j in range(4)) for i in range(4)]
        integrand = sum(ev[i] * sig[i] for i in range(4))
        return integrand * r

    @LinearForm
    def body_load(v, w):
        # Centrifugal body force density f_r = rho * omega^2 * r (radial), f_x = 0.
        r = w.x[1]
        f_r = DENSITY_KG_M3 * (omega ** 2) * r
        vr = v.value[1]
        return f_r * vr * r

    K = stiffness.assemble(basis)
    f = body_load.assemble(basis)

    # BC: fix the single node closest to the bore inner-face midpoint in x only.
    # Bore inner face is the minimum-radius surface; midpoint is at x=0.
    coords = mesh.p  # (2, nverts) in metres
    r_min = coords[1].min()
    target = np.array([0.0, r_min])
    dist = np.linalg.norm(coords.T - target[None, :], axis=1)
    fixed_vertex = int(np.argmin(dist))

    # Map that vertex to the x-DOF of the vector P2 basis.
    dofs = basis.nodal_dofs  # shape (2, nverts): row 0 -> x dof, row 1 -> r dof
    fixed_dof = int(dofs[0, fixed_vertex])

    x = solve(*condense(K, f, D=np.array([fixed_dof]), x=np.zeros(basis.N)))

    # Recover nodal stresses by projecting element-wise stress onto P2 nodes.
    sxx, srr, stt, sxr = _nodal_stresses(basis, x, C)
    return basis, sxx, srr, stt, sxr


def _nodal_stresses(basis, x, C):
    """L2-project element stress fields to the P2 nodal points of the mesh.

    Returns four (nverts,) arrays of stress components in the same units as C.
    """
    # Stress as a functional of the solution evaluated at quadrature points, then
    # projected to a scalar P2 basis via mass-matrix solve.
    from skfem import ElementTriP2 as _P2
    from skfem.helpers import grad as _grad

    scalar_basis = basis.with_element(_P2())

    def stress_field(component):
        @LinearForm
        def lf(v, w):
            du = _grad(w["disp"])
            r = w.x[1]
            r_safe = np.where(np.abs(r) < EPS, EPS, r)
            e_xx = du[0, 0]
            e_rr = du[1, 1]
            e_tt = w["disp"].value[1] / r_safe
            e_xr = du[0, 1] + du[1, 0]
            e = [e_xx, e_rr, e_tt, e_xr]
            sig = sum(C[component, j] * e[j] for j in range(4))
            return sig * v * w.x[1]

        return lf

    @BilinearForm
    def mass(u, v, w):
        return u * v * w.x[1]

    disp = basis.interpolate(x)
    M = mass.assemble(scalar_basis)
    out = []
    for comp in range(4):
        b = stress_field(comp).assemble(scalar_basis, disp=disp)
        out.append(solve(M, b))
    # Map scalar-P2 dof ordering to mesh vertex ordering.
    return tuple(_scalar_dofs_to_vertices(scalar_basis, arr) for arr in out)


def _scalar_dofs_to_vertices(scalar_basis, values):
    """Extract per-vertex values from a scalar P2 dof vector.

    For P2 on triangles the first ``nverts`` nodal dofs correspond to mesh
    vertices (edge-midpoint dofs follow); we return the vertex-aligned slice.
    """
    nverts = scalar_basis.mesh.p.shape[1]
    nodal = scalar_basis.nodal_dofs[0]  # (nverts,)
    return values[nodal]


def compute_phase_equivalent_stresses(
    nodes: np.ndarray,
    zone_ids: np.ndarray,
    region_ids: np.ndarray,
    geometry_params: Dict[str, float],
    radial_breaks: np.ndarray,
    mesh_obj,
    triangles: np.ndarray,
) -> np.ndarray:
    """Axisymmetric FEM von Mises stress field scaled across the 7 flight phases.

    Parameters
    ----------
    nodes:
        (N, 2) array [x, r] in MILLIMETRES (the mesh vertex coordinates).
    zone_ids, region_ids:
        (N,) int32 zone/region labels (unused by the FEM solve, kept for the
        stable call signature and possible diagnostics).
    geometry_params, radial_breaks:
        Geometry description, used only for diagnostic logging on failure.
    mesh_obj:
        The ``skfem.MeshTri`` from :class:`mesh_ops.MeshData`, coordinates in mm.
    triangles:
        (T, 3) connectivity (kept for signature stability; mesh_obj carries it).

    Returns
    -------
    (N, 7) float64 array of von Mises stress per flight phase, in MPa.  On FEM
    failure a zero array of this shape is returned so generation can continue.
    """
    n_nodes = nodes.shape[0]
    n_phases = CYCLE_SPEED_FACTORS.shape[0]

    try:
        # Solve in SI: convert mesh coordinates mm -> metres.
        mesh_m = MeshTri(mesh_obj.p * 1e-3, mesh_obj.t)

        basis, sxx, srr, stt, sxr = _assemble_and_solve(mesh_m, OMEGA_REF_RAD_S)

        # Stresses come back in Pa -> convert to MPa.
        sxx = sxx * 1e-6
        srr = srr * 1e-6
        stt = stt * 1e-6
        sxr = sxr * 1e-6

        base_vm = np.sqrt(
            0.5 * ((sxx - srr) ** 2 + (srr - stt) ** 2 + (stt - sxx) ** 2 + 6.0 * sxr ** 2)
        )

        # Vertex ordering of mesh_m matches the input `nodes` ordering, since the
        # mesh was built from those vertices.  Guard against any mismatch.
        if base_vm.shape[0] != n_nodes:
            raise RuntimeError(
                f"FEM nodal stress count {base_vm.shape[0]} != node count {n_nodes}"
            )

        phase_scale = CYCLE_SPEED_FACTORS ** 2
        phase_stress = base_vm[:, None] * phase_scale[None, :]
        return phase_stress.astype(np.float64)

    except Exception as exc:  # noqa: BLE001 — must not crash dataset generation
        logger.warning(
            "Axisymmetric FEM solve failed (%s: %s); returning zero stress. "
            "geometry_params=%s",
            type(exc).__name__,
            exc,
            geometry_params,
        )
        return np.zeros((n_nodes, n_phases), dtype=np.float64)


def compute_stress_max(phase_stress: np.ndarray) -> np.ndarray:
    return np.max(phase_stress, axis=1).astype(np.float64)


def _sn_param_arrays(zone_ids: np.ndarray, lifing_mode: str):
    """Return per-node (knee_stress, knee_life, slope_high, slope_low) arrays.

    ``zonal``: each node uses its zone's S-N set.  ``uniform``: every node uses
    the single :data:`UNIFORM_SN_PARAMS` set (web zone values).
    """
    if lifing_mode not in ("zonal", "uniform"):
        raise ValueError(f"lifing_mode must be 'zonal' or 'uniform', got {lifing_mode!r}")

    n = zone_ids.shape[0]
    knee_s = np.empty(n, dtype=np.float64)
    knee_n = np.empty(n, dtype=np.float64)
    m_high = np.empty(n, dtype=np.float64)
    m_low = np.empty(n, dtype=np.float64)

    if lifing_mode == "uniform":
        p = UNIFORM_SN_PARAMS
        knee_s[:] = p["knee_stress_mpa"]
        knee_n[:] = p["knee_life"]
        m_high[:] = p["slope_high"]
        m_low[:] = p["slope_low"]
        return knee_s, knee_n, m_high, m_low

    for zid, name in enumerate(_ZONE_NAMES_BY_ID):
        p = ZONAL_SN_PARAMS[name]
        mask = zone_ids == zid
        knee_s[mask] = p["knee_stress_mpa"]
        knee_n[mask] = p["knee_life"]
        m_high[mask] = p["slope_high"]
        m_low[mask] = p["slope_low"]
    return knee_s, knee_n, m_high, m_low


def _piecewise_sn_life_per_phase(
    sigma_a: np.ndarray,
    zone_ids: np.ndarray,
    lifing_mode: str,
) -> np.ndarray:
    """Per-phase cycles-to-failure from piecewise log-log (knee-based) S-N laws."""
    knee_s, knee_n, m_high, m_low = _sn_param_arrays(zone_ids, lifing_mode)
    sigma_knee = knee_s[:, None]
    n_knee = knee_n[:, None]
    slope_high = m_high[:, None]
    slope_low = m_low[:, None]

    ratio = np.maximum(sigma_a, EPS) / np.maximum(sigma_knee, EPS)
    # Above the knee (high stress, short life) the curve is steep (slope_high);
    # below the knee (long life) it is shallower (slope_low).
    high_branch = n_knee * np.power(np.maximum(ratio, EPS), -slope_high)
    low_branch = n_knee * np.power(np.maximum(ratio, EPS), -slope_low)
    return np.where(sigma_a >= sigma_knee, high_branch, low_branch)


def compute_life_raw(
    phase_stress: np.ndarray,
    zone_ids: np.ndarray,
    nodes: np.ndarray,
    geometry_params: Dict[str, float],
    radial_breaks: np.ndarray,
    lifing_mode: str = "zonal",
) -> np.ndarray:
    """Palmgren-Miner life from per-phase von Mises stress and zone/uniform S-N laws.

    The per-phase stress amplitude is taken as half the von Mises range relative
    to the engine-off (zero-speed) state, i.e. half the phase von Mises value,
    consistent with an R=-1 / fully-reversed S-N anchoring.
    """
    sigma_eq = np.maximum(phase_stress, 0.0)
    # Each flight phase cycles from rest to its peak: amplitude = 0.5 * range.
    sigma_a = 0.5 * sigma_eq
    n_fail = _piecewise_sn_life_per_phase(sigma_a, zone_ids, lifing_mode)

    damage_per_cycle = np.sum(CYCLE_PHASE_WEIGHTS[None, :] / np.maximum(n_fail, 1e-30), axis=1)
    # Floor only to guard against a genuine zero-damage divide; do NOT clip the
    # real (very large) life values that arise when stresses sit below the
    # endurance limit, otherwise the life field collapses to a constant.
    return (1.0 / np.maximum(damage_per_cycle, 1e-300)).astype(np.float64)
