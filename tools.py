from pathlib import Path

import numpy as np


def load_diagnostic(sim, species: str | None, key: str, n_workers: int) -> np.ndarray:
    diag = sim[species][key] if species is not None else sim[key]
    diag.load_all(n_workers=n_workers)
    return np.asarray(diag.data, dtype=np.float64)


def load_fields(
    sim_label: str, cache_dir: Path, simulations: dict[str, Path], fields: list[str], n_workers: int = 8
) -> dict[str, np.ndarray]:
    """Load (or read from cache) the cell-centered EM fields (n_t, nx)."""
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{sim_label}_fields.npz"
    if cache.exists():
        # print("Loading fields from cache:", cache)
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}

    # If the cache doesn't exist, load the fields from the simulation and save them to cache
    # Start by importing osiris_utils (so that the Simulation class is available) and loading the simulation

    try:
        import osiris_utils as ou
    except ImportError:
        raise ImportError("osiris_utils is required to load the simulation data. Please install it.") from None

    sim = ou.Simulation(str(simulations[sim_label]))
    out = {}
    for key in fields:
        arr = load_diagnostic(sim, None, key, n_workers)

        def _center_field(arr: np.ndarray, staggered: bool) -> np.ndarray:
            """Average a Yee-staggered field (dumped at +dx/2) to cell centers (periodic)."""
            if not staggered:
                return arr
            return 0.5 * (arr + np.roll(arr, 1, axis=1))  # axis 0 is time, axis 1 is x1

        out[key] = _center_field(arr, key in {"e1", "b2", "b3"}).astype(np.float32)
    np.savez(cache, **out)
    return out


def load_species_diagnostics(
    sim_label: str,
    species_up: str,
    species_down: str,
    cache_dir: Path,
    simulations: dict[str, Path],
    n_workers: int = 8,
) -> dict[str, np.ndarray]:
    """Load (or read from cache) the derived fluid moments of one electron species."""
    cache_dir.mkdir(exist_ok=True)
    cache = cache_dir / f"{sim_label}_{species_up}_{species_down}.npz"
    if cache.exists():
        # print("Loading species diagnostics from cache:", cache)
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}
    try:
        import osiris_utils as ou
    except ImportError:
        raise ImportError("osiris_utils is required to load the simulation data. Please install it.") from None

    sim = ou.Simulation(str(simulations[sim_label]))
    raw_up = {
        k: load_diagnostic(sim, species_up, k, n_workers)
        for k in [
            "n",
            "ufl1",
            "ufl2",
            "ufl3",
            "vfl1",
            "vfl2",
            "vfl3",
            "nvfl1",
            "P11",
            "P12",
            "P22",
            "Q111",
            "Q112",
            "Q122",
            "T11",
            "T12",
            "M11",
            "M12",
            "M22",
        ]
    }

    raw_down = {
        k: load_diagnostic(sim, species_down, k, n_workers)
        for k in [
            "n",
            "ufl1",
            "ufl2",
            "ufl3",
            "vfl1",
            "vfl2",
            "vfl3",
            "nvfl1",
            "P11",
            "P12",
            "P22",
            "Q111",
            "Q112",
            "Q122",
            "T11",
            "T12",
            "M11",
            "M12",
            "M22",
        ]
    }

    def _density_weighted_average(key: str) -> np.ndarray:
        """Compute the density-weighted average of a quantity from two species."""
        return (raw_up[key] * raw_up["n"] + raw_down[key] * raw_down["n"]) / (raw_up["n"] + raw_down["n"])

    # n = int(f dv) = int(f_up + f_down) dv = n_up + n_down
    n = raw_up["n"] + raw_down["n"]  # Total number density

    # v = int(v f dv) / n = (int(v f_up dv) + int(v f_down dv)) / n = (n_up v_up + n_down v_down) / n
    vfl1, vfl2, vfl3 = (
        _density_weighted_average("vfl1"),
        _density_weighted_average("vfl2"),
        _density_weighted_average("vfl3"),
    )  # Fluid velocity
    ufl1, ufl2 = _density_weighted_average("ufl1"), _density_weighted_average("ufl2")  # Fluid momentum

    # M = int(u u f dv) = int(u u f_up dv) + int(u u f_down dv) = M_up + M_down
    M11 = raw_up["M11"] + raw_down["M11"]  # Centered Momentum Flux Tensor xx
    M12 = raw_up["M12"] + raw_down["M12"]  # Centered Momentum Flux Tensor xy

    # T = int((v - u) (v - u) f dv) = int((v - u) (v - u) f_up dv) + int((v - u) (v - u) f_down dv)
    T11 = raw_up["T11"] + raw_down["T11"]  # Temperature tensor xx (already centered)
    T12 = raw_up["T12"] + raw_down["T12"]  # Temperature tensor xy (already centered)

    # P = int(u v f dv) = int(u v f_up dv) + int(u v f_down dv) = P_up + P_down
    P11 = raw_up["P11"] + raw_down["P11"]  # Mixed Momentum Flux Tensor xx
    P12 = raw_up["P12"] + raw_down["P12"]  # Mixed Momentum Flux Tensor xy

    # Q = int(u u v f dv) = int(u u v f_up dv) + int(u u v f_down dv) = Q_up + Q_down
    Q111 = raw_up["Q111"] + raw_down["Q111"]  # Mixed Third Moment Tensor xxx
    Q112 = raw_up["Q112"] + raw_down["Q112"]  # Mixed Third Moment Tensor xxy
    Q122 = raw_up["Q122"] + raw_down["Q122"]  # Mixed Third Moment Tensor xyy

    # nvfl = int(v f dv) = int(v f_up dv) + int(v f_down dv) = n_up v_up + n_down v_down
    nvfl1 = raw_up["nvfl1"] + raw_down["nvfl1"]  # Number flux

    out = {
        "n": n,
        "vfl1": vfl1,
        "vfl2": vfl2,
        "vfl3": vfl3,
        "ufl1": ufl1,
        "ufl2": ufl2,
        "P11": P11,
        "P12": P12,
        "T11": T11,
        "T12": T12,
        "M11": M11,
        "M12": M12,
        "Q111": Q111,  # raw third moments kept for the model-discovery flux term
        "Q112": Q112,
        "Q122": Q122,
        "nvfl1": nvfl1,  # number flux, kept for the continuity equation
    }

    # out["Q221"] = opt["Q122"]  # Q221 = Q122 (fully symmetric third moment)
    # out["Q221c"] = opt["Q122"] - 2 * ufl2 * raw["P12"] - vfl1 * opt["M22"] + 2 * n * ufl2**2 * vfl1
    out = {k: v.astype(np.float32) for k, v in out.items()}
    np.savez(cache, **out)
    return out


def load_all(sim_label: str, species: list[str], simulations: dict[str, Path], cache_dir: Path, fields: list[str], n_workers: int = 8) -> dict[str, np.ndarray]:
    """Species moments + EM fields, all shaped (n_t, nx) float32."""
    d = load_species_diagnostics(sim_label, species[0], species[1], n_workers=n_workers, cache_dir=cache_dir, simulations=simulations)
    d.update(load_fields(sim_label, simulations=simulations, cache_dir=cache_dir, fields=fields, n_workers=n_workers))
    return d
