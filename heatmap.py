import os
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.io import loadmat
from scipy.signal import savgol_filter

plt.style.use(["science", "ieee"])

MAT_PATH = "phases_1.mat"
OUTDIR = "results"

USE_CRAMERI_CMAP = False
CRAMERI_CMAP_NAME = "batlow"

FMIN_HZ = 71000.0
FMAX_HZ = 73000.0
ANCHOR_KHZ = None

PHASE_MODE = "12"
PHIBITS = np.array([3, 4, 10, 11, 12, 13, 15], dtype=int)

CASES = (
    {"N": 55, "a": 4, "x0": 1},
    {"N": 135, "a": 4, "x0": 1},
)

WRAPS_MIN = 1.0
WRAPS_MAX = 45.0
WRAPS_NUM = 700

DELTA_MIN = 0.1
DELTA_MAX = 90.0
DELTA_NUM = 700

MAX_STEPS = 256
MIN_DF_KHZ = 1e-3
MIN_PERIOD_STEPS = 1


@dataclass(frozen=True)
class SweepCase:
    N: int
    a: int
    x0: int


@dataclass(frozen=True)
class PhaseData:
    f_sel_khz: np.ndarray
    f_min_khz: float
    f_max_khz: float
    f_anchor_khz: float
    phi_sel_mat: np.ndarray


@dataclass(frozen=True)
class Calibration:
    f_sel_khz: np.ndarray
    f_min_khz: float
    f_max_khz: float
    phi_unwrapped_avg: np.ndarray
    slope: float
    intercept: float


def wrap_360(phi_deg: float | np.ndarray) -> float | np.ndarray:
    return np.mod(phi_deg, 360.0)


def wrap_pm180(phi_deg: float | np.ndarray) -> float | np.ndarray:
    return (phi_deg + 180.0) % 360.0 - 180.0


def phi_target_deg(x: int, N: int) -> float:
    return 360.0 * (x % N) / float(N)


def interp_at(x_grid: np.ndarray, y_grid: np.ndarray, x: float) -> float:
    return float(np.interp(x, x_grid, y_grid, left=y_grid[0], right=y_grid[-1]))


def modular_period(case: SweepCase, max_steps: int) -> int:
    x = int(case.x0)
    for step in range(1, max_steps + 1):
        x = (case.a * x) % case.N
        if x == case.x0:
            return step
    raise RuntimeError(
        f"Could not find period for N={case.N}, a={case.a}, x0={case.x0}."
    )


def smooth_phase(phi: np.ndarray, window: int = 11, polyorder: int = 2) -> np.ndarray:
    n = phi.shape[-1]
    min_window = polyorder + 2 if polyorder % 2 == 0 else polyorder + 1
    window = min(window, n)
    if window % 2 == 0:
        window -= 1
    if window < min_window:
        return np.array(phi, dtype=float)
    return savgol_filter(phi, window_length=window, polyorder=polyorder, axis=-1)


def load_phase_data() -> PhaseData:
    data = loadmat(MAT_PATH)
    f_hz = np.squeeze(data["freqs"])
    f_khz = f_hz / 1000.0
    mask = (f_hz >= FMIN_HZ) & (f_hz <= FMAX_HZ)
    if not np.any(mask):
        raise RuntimeError("Selected frequency band is empty.")

    phibit_idx = PHIBITS - 1
    phi_12 = smooth_phase(np.array(data["phi_12_unwrapped"])[phibit_idx, :])
    phi_13 = smooth_phase(np.array(data["phi_13_unwrapped"])[phibit_idx, :])

    if PHASE_MODE == "12":
        phi = phi_12
    elif PHASE_MODE == "13":
        phi = phi_13
    else:
        raise ValueError("PHASE_MODE must be '12' or '13'.")

    f_min_khz = FMIN_HZ / 1000.0
    f_max_khz = FMAX_HZ / 1000.0
    f_anchor_khz = f_min_khz if ANCHOR_KHZ is None else float(ANCHOR_KHZ)

    return PhaseData(
        f_sel_khz=f_khz[mask],
        f_min_khz=f_min_khz,
        f_max_khz=f_max_khz,
        f_anchor_khz=f_anchor_khz,
        phi_sel_mat=phi[:, mask],
    )


def affine_scale_from_fit(
    f_sel_khz: np.ndarray,
    phi_sel: np.ndarray,
    f_min_khz: float,
    f_max_khz: float,
    wraps: float,
    f_anchor_khz: float,
) -> tuple[np.ndarray, float, float]:
    if wraps <= 0:
        raise ValueError("wraps must be positive.")

    slope_raw, intercept_raw = np.polyfit(f_sel_khz, phi_sel, deg=1)
    y_min = slope_raw * f_min_khz + intercept_raw
    y_max = slope_raw * f_max_khz + intercept_raw
    y_anchor = slope_raw * f_anchor_khz + intercept_raw
    denom = y_max - y_min

    scale = 1.0 if np.isclose(denom, 0.0) else 360.0 * wraps / denom
    shift = -scale * y_anchor

    phi_cal = scale * phi_sel + shift
    slope_cal = float(scale * slope_raw)
    intercept_cal = float(scale * intercept_raw + shift)

    return phi_cal, slope_cal, intercept_cal


def build_calibration(data: PhaseData, wraps: float) -> Calibration:
    phis = []
    slopes = []
    intercepts = []

    for phi_sel in data.phi_sel_mat:
        phi_cal, slope, intercept = affine_scale_from_fit(
            data.f_sel_khz,
            phi_sel,
            data.f_min_khz,
            data.f_max_khz,
            wraps,
            data.f_anchor_khz,
        )
        phis.append(phi_cal)
        slopes.append(slope)
        intercepts.append(intercept)

    return Calibration(
        f_sel_khz=data.f_sel_khz,
        f_min_khz=data.f_min_khz,
        f_max_khz=data.f_max_khz,
        phi_unwrapped_avg=np.mean(np.stack(phis, axis=0), axis=0),
        slope=float(np.mean(slopes)),
        intercept=float(np.mean(intercepts)),
    )


def align_calibration(cal: Calibration, case: SweepCase) -> Calibration:
    target = float(wrap_360(phi_target_deg(case.x0, case.N)))
    measured = float(
        wrap_360(interp_at(cal.f_sel_khz, cal.phi_unwrapped_avg, cal.f_min_khz))
    )
    shift = float(wrap_pm180(target - measured))
    return replace(
        cal,
        phi_unwrapped_avg=cal.phi_unwrapped_avg + shift,
        intercept=cal.intercept + shift,
    )


def frequency_for_state(
    phi_base_deg: float,
    slope: float,
    intercept: float,
    last_f_khz: float | None,
    min_df_khz: float,
    f_min_khz: float,
    f_max_khz: float,
) -> float | None:
    if np.isclose(slope, 0.0):
        return None

    def f_of_n(n: int) -> float:
        return float((phi_base_deg + 360.0 * n - intercept) / slope)

    if slope > 0:
        n_lo = int(
            np.ceil((f_min_khz * slope + intercept - phi_base_deg) / 360.0 - 1e-12)
        )
        n_hi = int(
            np.floor((f_max_khz * slope + intercept - phi_base_deg) / 360.0 + 1e-12)
        )
    else:
        n_lo = int(
            np.ceil((f_max_khz * slope + intercept - phi_base_deg) / 360.0 - 1e-12)
        )
        n_hi = int(
            np.floor((f_min_khz * slope + intercept - phi_base_deg) / 360.0 + 1e-12)
        )

    if n_lo > n_hi:
        return None

    if last_f_khz is None:
        n_values = np.arange(n_lo, n_hi + 1, dtype=int)
        f_values = np.array([f_of_n(int(n)) for n in n_values], dtype=float)
        return float(
            f_values[np.argmin(np.abs(f_values - 0.5 * (f_min_khz + f_max_khz)))]
        )

    f_required = last_f_khz + min_df_khz

    if slope > 0:
        n = int(
            np.ceil((f_required * slope + intercept - phi_base_deg) / 360.0 - 1e-12)
        )
        n = max(n, n_lo)
        return None if n > n_hi else f_of_n(n)

    n = int(np.floor((f_required * slope + intercept - phi_base_deg) / 360.0 + 1e-12))
    n = min(n, n_hi)
    return None if n < n_lo else f_of_n(n)


def return_distances(cal: Calibration, case: SweepCase) -> np.ndarray:
    distances = np.full(MAX_STEPS + 1, np.inf, dtype=float)
    x = int(case.x0)
    last_f = float(cal.f_min_khz)
    phi0 = float(wrap_360(interp_at(cal.f_sel_khz, cal.phi_unwrapped_avg, last_f)))

    for step in range(1, MAX_STEPS + 1):
        x = (case.a * x) % case.N
        phi_base = float(wrap_360(phi_target_deg(x, case.N)))
        f_khz = frequency_for_state(
            phi_base,
            cal.slope,
            cal.intercept,
            last_f,
            MIN_DF_KHZ,
            cal.f_min_khz,
            cal.f_max_khz,
        )
        if f_khz is None or not np.isfinite(f_khz):
            break

        phi_t = float(wrap_360(interp_at(cal.f_sel_khz, cal.phi_unwrapped_avg, f_khz)))
        distances[step] = abs(float(wrap_pm180(phi_t - phi0)))
        last_f = float(f_khz)

    return distances


def detected_periods_for_deltas(
    distances: np.ndarray, delta_values: np.ndarray
) -> np.ndarray:
    steps = np.arange(distances.size)
    valid_steps = steps >= max(int(MIN_PERIOD_STEPS), 1)
    hits = (distances[None, :] <= delta_values[:, None]) & valid_steps[None, :]
    detected = np.full(len(delta_values), -1, dtype=int)
    has_hit = np.any(hits, axis=1)
    detected[has_hit] = np.argmax(hits[has_hit], axis=1)
    return detected


def centered_extent(x_values: np.ndarray, y_values: np.ndarray) -> list[float]:
    dx = float(x_values[1] - x_values[0]) if len(x_values) > 1 else 1.0
    dy = float(y_values[1] - y_values[0]) if len(y_values) > 1 else 1.0
    return [
        float(x_values[0] - 0.5 * dx),
        float(x_values[-1] + 0.5 * dx),
        float(y_values[0] - 0.5 * dy),
        float(y_values[-1] + 0.5 * dy),
    ]


def heatmap_cmap() -> ListedColormap:
    if not USE_CRAMERI_CMAP:
        return ListedColormap(["white", "black"])

    try:
        import cmcrameri.cm as cmc

        base = getattr(cmc, CRAMERI_CMAP_NAME)
    except (ImportError, AttributeError):
        base = plt.get_cmap(f"cmc.{CRAMERI_CMAP_NAME}")

    return ListedColormap([base(0.08), base(0.88)])


def plot_heatmap(
    heat: np.ndarray,
    wraps_values: np.ndarray,
    delta_values: np.ndarray,
    outpdf: str,
    title: str,
) -> None:
    cmap = heatmap_cmap()
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots()
    image = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=centered_extent(wraps_values, delta_values),
    )

    ax.set_title(title)
    ax.set_xlabel("Wraps")
    ax.set_ylabel(r"$\Delta$ (deg)")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(outpdf, bbox_inches="tight")
    plt.close(fig)


def run_case(
    case: SweepCase, data: PhaseData, wraps_values: np.ndarray, delta_values: np.ndarray
) -> None:
    true_period = modular_period(case, MAX_STEPS)
    heat = np.zeros((len(delta_values), len(wraps_values)), dtype=int)
    detected_periods = np.full_like(heat, -1, dtype=int)

    print(f"N={case.N}, a={case.a}, x0={case.x0}: true period r={true_period}")

    for j, wraps in enumerate(wraps_values):
        cal = align_calibration(build_calibration(data, float(wraps)), case)
        distances = return_distances(cal, case)
        detected = detected_periods_for_deltas(distances, delta_values)
        detected_periods[:, j] = detected
        heat[:, j] = detected == true_period

    success_count = int(np.sum(heat))
    print(f"N={case.N}: success count {success_count} / {heat.size}")

    prefix = f"N{case.N}_a{case.a}_x0_{case.x0}"

    outpdf = os.path.join(OUTDIR, f"{prefix}_heatmap.pdf")
    plot_heatmap(
        heat,
        wraps_values,
        delta_values,
        outpdf,
        title=rf"$N={case.N}$, $a={case.a}$, $r={true_period}$",
    )
    print(f"Saved {outpdf}")


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    data = load_phase_data()
    wraps_values = np.linspace(WRAPS_MIN, WRAPS_MAX, WRAPS_NUM)
    delta_values = np.linspace(DELTA_MIN, DELTA_MAX, DELTA_NUM)

    for params in CASES:
        run_case(SweepCase(**params), data, wraps_values, delta_values)


if __name__ == "__main__":
    main()
