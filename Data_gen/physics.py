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
from typing import Any, Dict

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import grad

from .config import (
    BLADE_EQUIV_CG_RADIUS_MM,
    BLADE_EQUIV_MASS_KG,
    BLADE_EQUIV_NUM_BLADES,
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
OMEGA_REF_RAD_S = 4000.0   # takeoff rotational speed [rad/s]

EPS = 1e-6
RIM_TOP_X_TOL_M = 5e-4
RIM_TOP_HORIZONTAL_TOL_M = 1e-6
RIM_TOP_R_TOL_M = 2e-3


def blade_equiv_force_n(omega: float) -> float:
    """Total blade-equivalent centrifugal force [N] at angular speed *omega*.

    Annular/axisymmetric surrogate for N discrete blades of mass m_blade each,
    rotating at *omega* with their centre of gravity at radius r_cg:

        F_total = N * m_blade * omega^2 * r_cg

    This load is fixed (same physical basis for every generated sample) and is
    applied as a boundary traction on the rear platform load-transfer face
    inside :func:`_assemble_and_solve`.
    """
    r_cg_m = BLADE_EQUIV_CG_RADIUS_MM * 1e-3
    return float(BLADE_EQUIV_NUM_BLADES) * float(BLADE_EQUIV_MASS_KG) * (float(omega) ** 2) * r_cg_m


def extract_rim_top_metadata(rim_face_metadata: Dict[str, np.ndarray] | None) -> Dict[str, float | None]:
    """Return rim-top face metadata in mm and m from contour metadata."""
    out: Dict[str, float | None] = {
        "rim_top_r_mm": None,
        "rim_top_x_min_mm": None,
        "rim_top_x_max_mm": None,
        "rim_top_r_m": None,
        "rim_top_x_min_m": None,
        "rim_top_x_max_m": None,
    }
    if rim_face_metadata is None:
        return out
    if "blade_rim_top_r_mm" in rim_face_metadata:
        out["rim_top_r_mm"] = float(np.asarray(rim_face_metadata["blade_rim_top_r_mm"]).reshape(-1)[0])
        out["rim_top_r_m"] = float(out["rim_top_r_mm"]) * 1e-3
    if "blade_rim_top_x_min_mm" in rim_face_metadata:
        out["rim_top_x_min_mm"] = float(np.asarray(rim_face_metadata["blade_rim_top_x_min_mm"]).reshape(-1)[0])
        out["rim_top_x_min_m"] = float(out["rim_top_x_min_mm"]) * 1e-3
    if "blade_rim_top_x_max_mm" in rim_face_metadata:
        out["rim_top_x_max_mm"] = float(np.asarray(rim_face_metadata["blade_rim_top_x_max_mm"]).reshape(-1)[0])
        out["rim_top_x_max_m"] = float(out["rim_top_x_max_mm"]) * 1e-3
    return out


def select_blade_rim_top_facets(
    mesh: MeshTri,
    radial_breaks_m: np.ndarray | None = None,
    rim_top_r_m: float | None = None,
    rim_top_x_min_m: float | None = None,
    rim_top_x_max_m: float | None = None,
) -> Dict[str, Any]:
    """Reproduce the production blade-load facet selection and audit metadata."""
    bf_ids = mesh.boundary_facets()
    if bf_ids.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return {
            "boundary_facet_ids": np.empty(0, dtype=int),
            "facet_ids": np.empty(0, dtype=int),
            "facet_mid_x_m": empty,
            "facet_mid_r_m": empty,
            "facet_lengths_m": empty,
            "facet_delta_x_m": empty,
            "facet_delta_r_m": empty,
            "facet_radius_deviation_m": empty,
            "used_fallback": rim_top_r_m is None,
            "rim_top_r_m": rim_top_r_m,
            "rim_top_x_min_m": rim_top_x_min_m,
            "rim_top_x_max_m": rim_top_x_max_m,
        }

    fv_x = mesh.p[0, mesh.facets[:, bf_ids]]
    fv_r = mesh.p[1, mesh.facets[:, bf_ids]]
    mid_x = fv_x.mean(axis=0)
    mid_r = fv_r.mean(axis=0)
    dx = fv_x[1] - fv_x[0]
    dr = fv_r[1] - fv_r[0]
    ds = np.sqrt(dx ** 2 + dr ** 2)

    used_fallback = rim_top_r_m is None
    if rim_top_r_m is not None:
        rim_mask = (
            (np.abs(dr) < RIM_TOP_HORIZONTAL_TOL_M)
            & (mid_r >= rim_top_r_m - RIM_TOP_R_TOL_M)
            & (mid_r <= rim_top_r_m + RIM_TOP_R_TOL_M)
        )
        if rim_top_x_min_m is not None and rim_top_x_max_m is not None:
            rim_mask = (
                rim_mask
                & (mid_x >= rim_top_x_min_m - RIM_TOP_X_TOL_M)
                & (mid_x <= rim_top_x_max_m + RIM_TOP_X_TOL_M)
            )
    else:
        if radial_breaks_m is not None and len(radial_breaks_m) > 4:
            r4_m = float(radial_breaks_m[4])
        else:
            r4_m = float(mesh.p[1].max()) - 0.020
        rim_zone_mask = mesh.p[1] >= r4_m * 0.95
        r_top_m = float(mesh.p[1, rim_zone_mask].max()) if np.any(rim_zone_mask) else float(mesh.p[1].max())
        rim_mask = (
            (np.abs(dr) < RIM_TOP_HORIZONTAL_TOL_M)
            & (mid_r >= r_top_m - RIM_TOP_R_TOL_M)
        )
        rim_top_r_m = r_top_m

    selected = bf_ids[rim_mask]
    radius_deviation = mid_r[rim_mask] - float(rim_top_r_m) if selected.size > 0 and rim_top_r_m is not None else np.empty(0, dtype=np.float64)
    return {
        "boundary_facet_ids": bf_ids.astype(int),
        "facet_ids": selected.astype(int),
        "facet_mid_x_m": mid_x[rim_mask].astype(np.float64),
        "facet_mid_r_m": mid_r[rim_mask].astype(np.float64),
        "facet_lengths_m": ds[rim_mask].astype(np.float64),
        "facet_delta_x_m": dx[rim_mask].astype(np.float64),
        "facet_delta_r_m": dr[rim_mask].astype(np.float64),
        "facet_radius_deviation_m": radius_deviation.astype(np.float64),
        "used_fallback": used_fallback,
        "rim_top_r_m": rim_top_r_m,
        "rim_top_x_min_m": rim_top_x_min_m,
        "rim_top_x_max_m": rim_top_x_max_m,
    }


def compute_blade_rim_face_geometry(selection: Dict[str, Any]) -> Dict[str, float]:
    """Return effective meridional length, mean radius, and axisymmetric area."""
    lengths = np.asarray(selection["facet_lengths_m"], dtype=np.float64)
    radii = np.asarray(selection["facet_mid_r_m"], dtype=np.float64)
    face_length_m = float(lengths.sum()) if lengths.size else 0.0
    mean_radius_m = float(radii.mean()) if radii.size else 0.0
    face_area_m2 = float(np.sum(2.0 * np.pi * radii * lengths)) if lengths.size else 0.0
    return {
        "face_length_m": face_length_m,
        "mean_radius_m": mean_radius_m,
        "face_area_m2": face_area_m2,
    }


def compute_blade_rim_traction_pa(
    target_force_n: float,
    mean_radius_m: float,
    face_length_m: float,
) -> float:
    """Return uniform radial traction that recovers the requested resultant."""
    if target_force_n <= 0.0 or mean_radius_m <= 1e-12 or face_length_m <= 1e-12:
        return 0.0
    return float(target_force_n) / (2.0 * np.pi * float(mean_radius_m) * float(face_length_m))


def recover_blade_rim_resultant_n(
    traction_pa: float,
    facet_mid_r_m: np.ndarray,
    facet_lengths_m: np.ndarray,
) -> float:
    """Integrate the axisymmetric traction over the selected facets."""
    radii = np.asarray(facet_mid_r_m, dtype=np.float64)
    lengths = np.asarray(facet_lengths_m, dtype=np.float64)
    if traction_pa == 0.0 or radii.size == 0 or lengths.size == 0:
        return 0.0
    return float(np.sum(float(traction_pa) * (2.0 * np.pi * radii * lengths)))

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


def _assemble_and_solve(mesh: MeshTri, omega: float, radial_breaks_m: np.ndarray | None = None,
                        rim_top_r_m: float | None = None,
                        rim_top_x_min_m: float | None = None,
                        rim_top_x_max_m: float | None = None,
                        include_body_force: bool = True,
                        include_blade_rim_load: bool = True):
    """Assemble and solve the axisymmetric elasticity system at angular speed omega.

    Coordinates of ``mesh`` are expected in METRES.  Returns nodal stress
    components (sxx, srr, stt, sxr) in Pa together with the global basis used.

    In addition to the centrifugal body load, a blade-equivalent centrifugal
    force (see :func:`blade_equiv_force_n`) is applied as a radial traction on
    the rim-top / blade-attachment surface — the horizontal boundary at
    r = rim_top_r_m (the ligament + arm land face at r5 + h_arm).

    This represents the physical reality: blades attach at the outer rim and
    pull it radially outward.  The rear drive arm receives NO direct blade
    traction by default.

    When rim_top_r_m is provided (from geometry metadata), the rim-top face is
    selected precisely by r proximity.  When omitted, a fallback selects the
    highest nearly-horizontal boundary facets in the rim zone (r >= r4).
    """
    C = _axisymmetric_C() * 1e6  # MPa -> Pa for a fully SI solve

    element = ElementVector(ElementTriP2())
    basis = Basis(mesh, element)

    @BilinearForm
    def stiffness(u, v, w):
        eu = _strain_components(u, w)
        ev = _strain_components(v, w)
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
    f = body_load.assemble(basis) if include_body_force else np.zeros(basis.N, dtype=np.float64)

    # --- Blade-equivalent centrifugal traction on the rim-top blade-attachment face ---
    f_total = blade_equiv_force_n(omega) if include_blade_rim_load else 0.0  # [N]
    selection = select_blade_rim_top_facets(
        mesh=mesh,
        radial_breaks_m=radial_breaks_m,
        rim_top_r_m=rim_top_r_m,
        rim_top_x_min_m=rim_top_x_min_m,
        rim_top_x_max_m=rim_top_x_max_m,
    )
    face_geom = compute_blade_rim_face_geometry(selection)
    t_r = compute_blade_rim_traction_pa(
        target_force_n=f_total,
        mean_radius_m=face_geom["mean_radius_m"],
        face_length_m=face_geom["face_length_m"],
    )
    recovered_force_n = recover_blade_rim_resultant_n(
        traction_pa=t_r,
        facet_mid_r_m=selection["facet_mid_r_m"],
        facet_lengths_m=selection["facet_lengths_m"],
    )
    if selection["facet_ids"].size > 0 and f_total > 0.0:
        fb = FacetBasis(mesh, element, facets=selection["facet_ids"])

        @LinearForm
        def blade_traction(v, w):
            r = w.x[1]
            vr = v.value[1]
            return t_r * vr * r

        f = f + blade_traction.assemble(fb)
    elif f_total > 0.0:
        logger.warning("Blade-equivalent load requested but no rim-top face facets were found; load not applied.")

    # BC: fix the single node closest to the bore inner-face midpoint in x only.
    coords = mesh.p
    r_min = coords[1].min()
    target = np.array([0.0, r_min])
    dist = np.linalg.norm(coords.T - target[None, :], axis=1)
    fixed_vertex = int(np.argmin(dist))

    dofs = basis.nodal_dofs
    fixed_dof = int(dofs[0, fixed_vertex])

    x = solve(*condense(K, f, D=np.array([fixed_dof]), x=np.zeros(basis.N)))

    # Recover nodal stresses by projecting element-wise stress onto P2 nodes.
    sxx, srr, stt, sxr = _nodal_stresses(basis, x, C)
    closure_error = abs(recovered_force_n - f_total) / max(abs(f_total), 1e-12) if f_total > 0.0 else 0.0
    diagnostics = {
        "include_body_force": bool(include_body_force),
        "include_blade_rim_load": bool(include_blade_rim_load),
        "target_force_n": float(f_total),
        "traction_pa": float(t_r),
        "recovered_force_n": float(recovered_force_n),
        "closure_error_rel": float(closure_error),
        "used_fallback": bool(selection["used_fallback"]),
        "selected_facet_ids": np.asarray(selection["facet_ids"], dtype=int),
        "selected_facet_mid_x_m": np.asarray(selection["facet_mid_x_m"], dtype=np.float64),
        "selected_facet_mid_r_m": np.asarray(selection["facet_mid_r_m"], dtype=np.float64),
        "selected_facet_lengths_m": np.asarray(selection["facet_lengths_m"], dtype=np.float64),
        "selected_facet_delta_x_m": np.asarray(selection["facet_delta_x_m"], dtype=np.float64),
        "selected_facet_delta_r_m": np.asarray(selection["facet_delta_r_m"], dtype=np.float64),
        "selected_facet_radius_deviation_m": np.asarray(selection["facet_radius_deviation_m"], dtype=np.float64),
        "face_length_m": float(face_geom["face_length_m"]),
        "mean_radius_m": float(face_geom["mean_radius_m"]),
        "face_area_m2": float(face_geom["face_area_m2"]),
        "rim_top_r_m": None if selection["rim_top_r_m"] is None else float(selection["rim_top_r_m"]),
        "rim_top_x_min_m": None if selection["rim_top_x_min_m"] is None else float(selection["rim_top_x_min_m"]),
        "rim_top_x_max_m": None if selection["rim_top_x_max_m"] is None else float(selection["rim_top_x_max_m"]),
    }
    return basis, sxx, srr, stt, sxr, diagnostics


def solve_axisymmetric_response(
    nodes: np.ndarray,
    zone_ids: np.ndarray,
    region_ids: np.ndarray,
    geometry_params: Dict[str, float],
    radial_breaks: np.ndarray,
    mesh_obj,
    triangles: np.ndarray,
    rim_face_metadata: Dict[str, np.ndarray] | None = None,
    include_body_force: bool = True,
    include_blade_rim_load: bool = True,
) -> Dict[str, Any]:
    """Solve the production axisymmetric FEM state and return fields plus diagnostics."""
    n_nodes = nodes.shape[0]
    n_phases = CYCLE_SPEED_FACTORS.shape[0]
    meta = extract_rim_top_metadata(rim_face_metadata)
    try:
        mesh_m = MeshTri(mesh_obj.p * 1e-3, mesh_obj.t)
        basis, sxx, srr, stt, sxr, diagnostics = _assemble_and_solve(
            mesh_m,
            OMEGA_REF_RAD_S,
            radial_breaks_m=radial_breaks * 1e-3,
            rim_top_r_m=meta["rim_top_r_m"],
            rim_top_x_min_m=meta["rim_top_x_min_m"],
            rim_top_x_max_m=meta["rim_top_x_max_m"],
            include_body_force=include_body_force,
            include_blade_rim_load=include_blade_rim_load,
        )

        sxx = sxx * 1e-6
        srr = srr * 1e-6
        stt = stt * 1e-6
        sxr = sxr * 1e-6
        base_vm = np.sqrt(
            0.5 * ((sxx - srr) ** 2 + (srr - stt) ** 2 + (stt - sxx) ** 2 + 6.0 * sxr ** 2)
        )
        if base_vm.shape[0] != n_nodes:
            raise RuntimeError(f"FEM nodal stress count {base_vm.shape[0]} != node count {n_nodes}")
        phase_scale = CYCLE_SPEED_FACTORS ** 2
        phase_stress = base_vm[:, None] * phase_scale[None, :]
        return {
            "ok": True,
            "basis": basis,
            "stress_components_mpa": {
                "sxx": sxx.astype(np.float64),
                "srr": srr.astype(np.float64),
                "stt": stt.astype(np.float64),
                "sxr": sxr.astype(np.float64),
            },
            "base_vm_mpa": base_vm.astype(np.float64),
            "phase_stress_mpa": phase_stress.astype(np.float64),
            "load_diagnostics": diagnostics,
            "metadata": meta,
        }
    except Exception as exc:  # noqa: BLE001 — must not crash dataset generation
        logger.warning(
            "Axisymmetric FEM solve failed (%s: %s); returning zero stress. "
            "geometry_params=%s",
            type(exc).__name__,
            exc,
            geometry_params,
        )
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "basis": None,
            "stress_components_mpa": {
                "sxx": np.zeros(n_nodes, dtype=np.float64),
                "srr": np.zeros(n_nodes, dtype=np.float64),
                "stt": np.zeros(n_nodes, dtype=np.float64),
                "sxr": np.zeros(n_nodes, dtype=np.float64),
            },
            "base_vm_mpa": np.zeros(n_nodes, dtype=np.float64),
            "phase_stress_mpa": np.zeros((n_nodes, n_phases), dtype=np.float64),
            "load_diagnostics": {
                "include_body_force": bool(include_body_force),
                "include_blade_rim_load": bool(include_blade_rim_load),
                "target_force_n": float(blade_equiv_force_n(OMEGA_REF_RAD_S) if include_blade_rim_load else 0.0),
                "traction_pa": 0.0,
                "recovered_force_n": 0.0,
                "closure_error_rel": 1.0 if include_blade_rim_load else 0.0,
                "used_fallback": bool(meta["rim_top_r_m"] is None),
                "selected_facet_ids": np.empty(0, dtype=int),
                "selected_facet_mid_x_m": np.empty(0, dtype=np.float64),
                "selected_facet_mid_r_m": np.empty(0, dtype=np.float64),
                "selected_facet_lengths_m": np.empty(0, dtype=np.float64),
                "selected_facet_delta_x_m": np.empty(0, dtype=np.float64),
                "selected_facet_delta_r_m": np.empty(0, dtype=np.float64),
                "selected_facet_radius_deviation_m": np.empty(0, dtype=np.float64),
                "face_length_m": 0.0,
                "mean_radius_m": 0.0,
                "face_area_m2": 0.0,
                "rim_top_r_m": meta["rim_top_r_m"],
                "rim_top_x_min_m": meta["rim_top_x_min_m"],
                "rim_top_x_max_m": meta["rim_top_x_max_m"],
            },
            "metadata": meta,
        }


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
    rim_face_metadata: Dict[str, np.ndarray] | None = None,
    include_body_force: bool = True,
    include_blade_rim_load: bool = True,
) -> np.ndarray:
    """Axisymmetric FEM von Mises stress field scaled across the 7 flight phases.

    Parameters
    ----------
    nodes:
        (N, 2) array [x, r] in MILLIMETRES (the mesh vertex coordinates).
    zone_ids, region_ids:
        (N,) int32 zone/region labels.
    geometry_params, radial_breaks:
        Geometry description used for diagnostics.
    mesh_obj:
        The ``skfem.MeshTri`` from :class:`mesh_ops.MeshData`, coordinates in mm.
    triangles:
        (T, 3) connectivity (kept for signature stability).
    rim_face_metadata : dict or None
        Optional geometry-tagged rim-top face coordinates from ``ContourData.metadata``.
        Expected keys: ``blade_rim_top_r_mm``, ``blade_rim_top_x_min_mm``,
        ``blade_rim_top_x_max_mm``.  When present, the blade traction is applied
        precisely to the tagged rim-top (ligament + arm land) face rather than
        via a radial threshold.

    Returns
    -------
    (N, 7) float64 array of von Mises stress per flight phase, in MPa.
    """
    response = solve_axisymmetric_response(
        nodes=nodes,
        zone_ids=zone_ids,
        region_ids=region_ids,
        geometry_params=geometry_params,
        radial_breaks=radial_breaks,
        mesh_obj=mesh_obj,
        triangles=triangles,
        rim_face_metadata=rim_face_metadata,
        include_body_force=include_body_force,
        include_blade_rim_load=include_blade_rim_load,
    )
    return response["phase_stress_mpa"].astype(np.float64)


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
