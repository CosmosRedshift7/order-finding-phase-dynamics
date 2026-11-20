from fractions import Fraction
from frequency_control import frequency_steps
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

USE_CRAMERI_CMAP = True
# Crameri's categorical palette: the first four entries maximize distinction.
# https://www.fabiocrameri.ch/categorical-colour-maps/
CRAMERI_CMAP_NAME = "vik"

# Category codes label the output; they do not affect the phase iteration.
NO_RETURN = 0
INVALID_PERIOD = 1
VALID_MULTIPLE = 2
EXACT_ORDER = 3
CLASS_LABELS = ("No return", "Invalid period", "Valid multiple", "Exact order")

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
WRAPS_NUM = 1500

DELTA_MIN = 0.1
DELTA_MAX = 90.0
DELTA_NUM = 1500

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


def return_distances(cal: Calibration, case: SweepCase) -> np.ndarray:
    distances = np.full(MAX_STEPS + 1, np.inf, dtype=float)
    initial_phase = Fraction(360 * (case.x0 % case.N), case.N)
    trajectory = frequency_steps(
        cal.slope, cal.intercept, cal.f_min_khz, cal.f_max_khz,
        MIN_DF_KHZ, case.a, initial_phase, MAX_STEPS,
    )
    initial_frequency, _, _ = next(trajectory)
    phi0 = interp_at(cal.f_sel_khz, cal.phi_unwrapped_avg, initial_frequency)
    for step, (frequency, _, _) in enumerate(trajectory, start=1):
        phase = interp_at(cal.f_sel_khz, cal.phi_unwrapped_avg, frequency)
        distances[step] = abs(float(wrap_pm180(phase - phi0)))

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


def classify_detected_periods(detected: np.ndarray, true_period: int) -> np.ndarray:
    """Label first-return candidates using the original reference period.

    For the configured coprime bases and initial states, true_period is the
    multiplicative order. A positive k is valid iff k is a multiple of it.
    This classification does not change how a candidate is generated.
    """
    classes = np.full(detected.shape, NO_RETURN, dtype=int)
    has_return = detected > 0
    classes[has_return] = INVALID_PERIOD
    nonminimal_multiple = (detected > true_period) & (detected % true_period == 0)
    classes[nonminimal_multiple] = VALID_MULTIPLE
    classes[detected == true_period] = EXACT_ORDER
    return classes


def heatmap_cmap() -> ListedColormap:
    if not USE_CRAMERI_CMAP:
        return ListedColormap(["white", "0.75", "0.4", "black"], name="period_classes")

    try:
        import cmcrameri.cm as cmc

        base = getattr(cmc, CRAMERI_CMAP_NAME)
    except (ImportError, AttributeError):
        base = plt.get_cmap(f"cmc.{CRAMERI_CMAP_NAME}")

    if CRAMERI_CMAP_NAME.endswith("S"):
        # Integer indices select the first distinct categorical entries.
        colors = base(np.arange(len(CLASS_LABELS)))
    else:
        # Also support a continuous Crameri map if the setting is changed.
        colors = base(np.linspace(0.08, 0.92, len(CLASS_LABELS)))
    return ListedColormap(colors, name=f"{CRAMERI_CMAP_NAME}_period_classes")


def plot_heatmap(
    heat: np.ndarray,
    wraps_values: np.ndarray,
    delta_values: np.ndarray,
    outpdf: str,
    title: str,
) -> None:
    cmap = heatmap_cmap()
    boundaries = np.arange(len(CLASS_LABELS) + 1) - 0.5
    norm = BoundaryNorm(boundaries, cmap.N)

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

    colorbar = fig.colorbar(
        image,
        ax=ax,
        boundaries=boundaries,
        ticks=np.arange(len(CLASS_LABELS)),
        spacing="uniform",
        pad=0.03,
    )
    colorbar.set_ticklabels(CLASS_LABELS)

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
        heat[:, j] = classify_detected_periods(detected, true_period)

    success_count = int(np.count_nonzero(heat == EXACT_ORDER))
    print(f"N={case.N}: success count {success_count} / {heat.size}")
    for code, label in enumerate(CLASS_LABELS):
        count = int(np.count_nonzero(heat == code))
        print(f"  {label}: {count} / {heat.size}")

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
