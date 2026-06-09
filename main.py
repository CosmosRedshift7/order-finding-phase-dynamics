from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from matplotlib.ticker import MaxNLocator
from scipy.io import loadmat
from scipy.signal import savgol_filter

plt.style.use(["science", "ieee", "no-latex"])


DEFAULT_LABEL_DX = 0.0
DEFAULT_LABEL_DY = 4.0

LABEL_ADJUSTMENTS_N55 = {
    0: {"dx": -3.0, "dy": 0.0},
    1: {"dx": -1.0, "dy": 0.0},
    7: {"dx": -5.0, "dy": -5.0},
    10: {"dx": 8.0, "dy": -3.0},
}

LABEL_ADJUSTMENTS_N135 = {
    0: {"dx": -3.0, "dy": -2.0},
    1: {"dx": -3.0, "dy": -2.0},
    4: {"dx": -6.0, "dy": -8.0},
    6: {"dx": -3.0, "dy": 0.0},
    7: {"dx": 2.0, "dy": -18.0},
    9: {"dx": -1.0, "dy": -2.0},
    10: {"dx": 8.0, "dy": -3.0},
    11: {"dx": -8.0, "dy": -8.0},
    12: {"dx": -9.0, "dy": -9.0},
    13: {"dx": -8.0, "dy": -2.0},
    14: {"dx": -3.0, "dy": -1.0},
    15: {"dx": 8.0, "dy": -8.0},
    16: {"dx": -4.0, "dy": 0.0},
    17: {"dx": 8.0, "dy": -8.0},
    18: {"dx": 8.0, "dy": -4.0},
}


@dataclass(frozen=True)
class Case:
    N: int
    a: int
    x0: int
    delta_deg: float
    label_adjustments: dict[int, dict[str, float]]


@dataclass(frozen=True)
class Config:
    mat_path: Path = Path("phases_1.mat")
    outdir: Path = Path("results")

    phase_key: str = "phi_12_unwrapped"
    phase_label: str = "12"
    phibits: tuple[int, ...] = (3, 4, 10, 11, 12, 13, 15)

    fmin_hz: float = 71_000.0
    fmax_hz: float = 73_000.0
    anchor_khz: float | None = None

    wraps_per_band: int = 8
    min_df_khz: float = 1e-3
    max_steps: int = 128
    min_period_steps: int = 1

    cases: tuple[Case, ...] = (
        Case(
            N=55,
            a=4,
            x0=1,
            delta_deg=7.0,
            label_adjustments=LABEL_ADJUSTMENTS_N55,
        ),
        Case(
            N=135,
            a=4,
            x0=1,
            delta_deg=10.0,
            label_adjustments=LABEL_ADJUSTMENTS_N135,
        ),
    )


def wrap_360(phi_deg: float | np.ndarray) -> float | np.ndarray:
    return np.mod(phi_deg, 360.0)


def wrap_pm180(phi_deg: float | np.ndarray) -> float | np.ndarray:
    return (phi_deg + 180.0) % 360.0 - 180.0


def target_phase_deg(x: int, N: int) -> float:
    return 360.0 * (x % N) / float(N)


def decode_phase_to_x(phi_deg_wrapped: float, N: int) -> int:
    grid = 360.0 * np.arange(N) / float(N)
    distances = np.abs(wrap_pm180(phi_deg_wrapped - grid))
    return int(np.argmin(distances))


def interp_at(x_grid: np.ndarray, y_grid: np.ndarray, x: float) -> float:
    return float(np.interp(x, x_grid, y_grid, left=y_grid[0], right=y_grid[-1]))


def axhspan_wrapped(
    ax,
    center_deg: float,
    half_width_deg: float,
    *,
    alpha: float = 0.12,
) -> None:
    center = float(wrap_360(center_deg))
    half_width = abs(float(half_width_deg))
    lo = center - half_width
    hi = center + half_width

    if 0.0 <= lo and hi <= 360.0:
        ax.axhspan(lo, hi, alpha=alpha)
    elif lo < 0.0:
        ax.axhspan(0.0, hi, alpha=alpha)
        ax.axhspan(360.0 + lo, 360.0, alpha=alpha)
    else:
        ax.axhspan(lo, 360.0, alpha=alpha)
        ax.axhspan(0.0, hi - 360.0, alpha=alpha)


@dataclass(frozen=True)
class Calibration:
    phase_label: str
    phibits: np.ndarray
    f_khz: np.ndarray
    f_min_khz: float
    f_max_khz: float
    anchor_khz: float
    wraps_per_band: int

    phi_cal_avg: np.ndarray
    slope_avg: float
    intercept_avg: float
    rmse_avg: float

    scale: np.ndarray
    shift: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    rmse: np.ndarray


def affine_scale_and_shift(
    f_khz: np.ndarray,
    phi_deg: np.ndarray,
    *,
    f_min_khz: float,
    f_max_khz: float,
    wraps_per_band: int,
    anchor_khz: float,
) -> tuple[float, float, float, float]:
    if wraps_per_band < 1:
        raise ValueError("wraps_per_band must be >= 1")

    a, b = np.polyfit(f_khz, phi_deg, deg=1)

    y_min = a * f_min_khz + b
    y_max = a * f_max_khz + b
    y_anchor = a * anchor_khz + b

    denom = y_max - y_min
    if np.isclose(denom, 0.0):
        scale = 1.0
    else:
        scale = 360.0 * wraps_per_band / denom

    shift = -scale * y_anchor
    slope_cal = scale * a
    intercept_cal = scale * b + shift

    return float(scale), float(shift), float(slope_cal), float(intercept_cal)


def calibrate_one_phibit(
    f_khz: np.ndarray,
    phi_deg: np.ndarray,
    *,
    f_min_khz: float,
    f_max_khz: float,
    wraps_per_band: int,
    anchor_khz: float,
) -> tuple[np.ndarray, float, float, float, float, float]:
    scale, shift, slope_cal, intercept_cal = affine_scale_and_shift(
        f_khz,
        phi_deg,
        f_min_khz=f_min_khz,
        f_max_khz=f_max_khz,
        wraps_per_band=wraps_per_band,
        anchor_khz=anchor_khz,
    )

    phi_cal = scale * phi_deg + shift

    fit_wrapped = wrap_360(slope_cal * f_khz + intercept_cal)
    data_wrapped = wrap_360(phi_cal)
    rmse = float(np.sqrt(np.mean(wrap_pm180(data_wrapped - fit_wrapped) ** 2)))

    return phi_cal, slope_cal, intercept_cal, rmse, scale, shift


def build_calibration(
    *,
    phase_label: str,
    phibits: np.ndarray,
    f_khz: np.ndarray,
    phi_mat: np.ndarray,
    f_min_khz: float,
    f_max_khz: float,
    wraps_per_band: int,
    anchor_khz: float,
) -> Calibration:
    calibrated_phases: list[np.ndarray] = []
    slopes: list[float] = []
    intercepts: list[float] = []
    rmses: list[float] = []
    scales: list[float] = []
    shifts: list[float] = []

    for phi_deg in phi_mat:
        phi_cal, slope, intercept, rmse, scale, shift = calibrate_one_phibit(
            f_khz,
            phi_deg,
            f_min_khz=f_min_khz,
            f_max_khz=f_max_khz,
            wraps_per_band=wraps_per_band,
            anchor_khz=anchor_khz,
        )

        calibrated_phases.append(phi_cal)
        slopes.append(slope)
        intercepts.append(intercept)
        rmses.append(rmse)
        scales.append(scale)
        shifts.append(shift)

    return Calibration(
        phase_label=phase_label,
        phibits=phibits.astype(int),
        f_khz=f_khz.astype(float),
        f_min_khz=float(f_min_khz),
        f_max_khz=float(f_max_khz),
        anchor_khz=float(anchor_khz),
        wraps_per_band=int(wraps_per_band),
        phi_cal_avg=np.mean(np.stack(calibrated_phases), axis=0).astype(float),
        slope_avg=float(np.mean(slopes)),
        intercept_avg=float(np.mean(intercepts)),
        rmse_avg=float(np.mean(rmses)),
        scale=np.array(scales, dtype=float),
        shift=np.array(shifts, dtype=float),
        slope=np.array(slopes, dtype=float),
        intercept=np.array(intercepts, dtype=float),
        rmse=np.array(rmses, dtype=float),
    )


def align_calibration_to_initial_state(
    cal: Calibration,
    *,
    N: int,
    x0: int,
) -> Calibration:
    phi_target = float(wrap_360(target_phase_deg(x0, N)))
    phi_measured = float(wrap_360(interp_at(cal.f_khz, cal.phi_cal_avg, cal.f_min_khz)))
    correction = float(wrap_pm180(phi_target - phi_measured))

    return replace(
        cal,
        phi_cal_avg=cal.phi_cal_avg + correction,
        intercept_avg=cal.intercept_avg + correction,
        intercept=cal.intercept + correction,
    )


def save_calibration_table(cal: Calibration, outpath: Path) -> None:
    lines = [
        "Per-phibit calibration parameters",
        f"phase_label = phi_{cal.phase_label}",
        f"band = [{cal.f_min_khz:.3f}, {cal.f_max_khz:.3f}] kHz",
        f"W = {cal.wraps_per_band}",
        f"anchor = {cal.anchor_khz:.3f} kHz",
        "",
        "reidx  orig_label   scale_s           shift_c           "
        "slope_cal(deg/kHz)   intercept_cal(deg)   rmse(deg)",
        "-" * 100,
    ]

    for j, label in enumerate(cal.phibits, start=1):
        i = j - 1
        lines.append(
            f"{j:5d}  {int(label):10d}  "
            f"{cal.scale[i]:16.8f}  "
            f"{cal.shift[i]:16.8f}  "
            f"{cal.slope[i]:18.8f}  "
            f"{cal.intercept[i]:20.8f}  "
            f"{cal.rmse[i]:10.6f}"
        )

    lines += [
        "",
        "Averaged calibration",
        f"slope_cal_avg = {cal.slope_avg:.8f} deg/kHz",
        f"intercept_cal_avg = {cal.intercept_avg:.8f} deg",
        f"rmse_avg = {cal.rmse_avg:.6f} deg",
    ]

    outpath.write_text("\n".join(lines), encoding="utf-8")


@dataclass(frozen=True)
class Simulation:
    N: int
    a: int
    x0: int

    x_true: np.ndarray
    x_hat: np.ndarray
    phi_target: np.ndarray
    phi_measured: np.ndarray
    freq_khz: np.ndarray
    phase_error_deg: np.ndarray
    wraps_used: np.ndarray

    period_data: int | None
    period_delta_deg: float
    min_period_steps: int

    f_grid_khz: np.ndarray
    phi_cal_avg: np.ndarray


def choose_wrap_index(
    *,
    phi_base_deg: float,
    slope: float,
    intercept: float,
    last_f_khz: float | None,
    min_df_khz: float,
    f_min_khz: float,
    f_max_khz: float,
) -> tuple[int, float, bool]:
    if np.isclose(slope, 0.0):
        return 0, float("nan"), False

    def frequency_for_wrap(n: int) -> float:
        return float((phi_base_deg + 360.0 * n - intercept) / slope)

    if slope > 0:
        n_min = (f_min_khz * slope + intercept - phi_base_deg) / 360.0
        n_max = (f_max_khz * slope + intercept - phi_base_deg) / 360.0
    else:
        n_min = (f_max_khz * slope + intercept - phi_base_deg) / 360.0
        n_max = (f_min_khz * slope + intercept - phi_base_deg) / 360.0

    n_lo = int(np.ceil(n_min - 1e-12))
    n_hi = int(np.floor(n_max + 1e-12))

    if n_lo > n_hi:
        return 0, float("nan"), False

    if last_f_khz is None:
        n_values = np.arange(n_lo, n_hi + 1, dtype=int)
        f_values = np.array([frequency_for_wrap(n) for n in n_values], dtype=float)
        f_mid = 0.5 * (f_min_khz + f_max_khz)
        best = int(np.argmin(np.abs(f_values - f_mid)))
        return int(n_values[best]), float(f_values[best]), True

    f_required = last_f_khz + min_df_khz
    n_required = (f_required * slope + intercept - phi_base_deg) / 360.0

    if slope > 0:
        n = max(int(np.ceil(n_required - 1e-12)), n_lo)
        ok = n <= n_hi
    else:
        n = min(int(np.floor(n_required + 1e-12)), n_hi)
        ok = n >= n_lo

    return n, frequency_for_wrap(n), bool(ok)


def run_experiment(
    cal: Calibration,
    *,
    N: int,
    a: int,
    x0: int,
    max_steps: int,
    min_df_khz: float,
    period_delta_deg: float,
    min_period_steps: int,
) -> Simulation:
    x_true: list[int] = []
    x_hat: list[int] = []
    phi_target: list[float] = []
    phi_measured: list[float] = []
    freq_khz: list[float] = []
    phase_error: list[float] = []
    wraps_used: list[int] = []

    last_f_khz: float | None = None

    def record_state(x_state: int, *, force_f_khz: float | None = None) -> bool:
        nonlocal last_f_khz

        base_phase = float(wrap_360(target_phase_deg(x_state, N)))

        if force_f_khz is None:
            n, f_khz, ok = choose_wrap_index(
                phi_base_deg=base_phase,
                slope=cal.slope_avg,
                intercept=cal.intercept_avg,
                last_f_khz=last_f_khz,
                min_df_khz=min_df_khz,
                f_min_khz=cal.f_min_khz,
                f_max_khz=cal.f_max_khz,
            )
            if not ok or not np.isfinite(f_khz):
                return False
        else:
            f_khz = float(np.clip(force_f_khz, cal.f_min_khz, cal.f_max_khz))
            n = int(
                np.rint(
                    (f_khz * cal.slope_avg + cal.intercept_avg - base_phase) / 360.0
                )
            )

        measured_unwrapped = interp_at(cal.f_khz, cal.phi_cal_avg, f_khz)
        target_wrapped = float(wrap_360(base_phase + 360.0 * n))
        measured_wrapped = float(wrap_360(measured_unwrapped))

        x_true.append(int(x_state))
        x_hat.append(decode_phase_to_x(measured_wrapped, N))
        phi_target.append(target_wrapped)
        phi_measured.append(measured_wrapped)
        freq_khz.append(float(f_khz))
        phase_error.append(float(wrap_pm180(measured_wrapped - target_wrapped)))
        wraps_used.append(int(n))

        last_f_khz = float(f_khz)
        return True

    x = int(x0)
    if not record_state(x, force_f_khz=cal.f_min_khz):
        raise RuntimeError("Failed to initialize the first state at f_min.")

    period_data: int | None = None
    phi0 = phi_measured[0]

    for _ in range(1, max_steps):
        x = (a * x) % N
        if not record_state(x):
            break

        return_dist = np.abs(wrap_pm180(np.array(phi_measured) - phi0))
        t = len(phi_measured) - 1

        if t >= max(min_period_steps, 1) and return_dist[t] <= period_delta_deg:
            period_data = int(t)
            break

    return Simulation(
        N=N,
        a=a,
        x0=x0,
        x_true=np.array(x_true, dtype=int),
        x_hat=np.array(x_hat, dtype=int),
        phi_target=np.array(phi_target, dtype=float),
        phi_measured=np.array(phi_measured, dtype=float),
        freq_khz=np.array(freq_khz, dtype=float),
        phase_error_deg=np.array(phase_error, dtype=float),
        wraps_used=np.array(wraps_used, dtype=int),
        period_data=period_data,
        period_delta_deg=float(period_delta_deg),
        min_period_steps=int(min_period_steps),
        f_grid_khz=cal.f_khz,
        phi_cal_avg=cal.phi_cal_avg,
    )


def load_selected_phase(cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(cfg.mat_path)

    f_hz = np.squeeze(data["freqs"]).astype(float)
    mask = (f_hz >= cfg.fmin_hz) & (f_hz <= cfg.fmax_hz)

    if not np.any(mask):
        raise RuntimeError("Selected frequency band is empty. Check fmin_hz/fmax_hz.")

    phibit_indices = np.array(cfg.phibits, dtype=int) - 1
    phi_full = np.asarray(data[cfg.phase_key], dtype=float)[phibit_indices]
    phi_full = smooth_rows(phi_full)

    return f_hz[mask] / 1000.0, phi_full[:, mask]


def smooth_rows(
    y: np.ndarray,
    *,
    max_window: int = 11,
    polyorder: int = 2,
) -> np.ndarray:
    n = y.shape[1]

    if n <= polyorder + 2:
        return y.copy()

    window = min(max_window, n if n % 2 == 1 else n - 1)
    window = max(window, polyorder + 2 + (polyorder % 2))

    if window % 2 == 0:
        window -= 1

    return savgol_filter(y, window_length=window, polyorder=polyorder, axis=-1)


def print_iteration_table(sim: Simulation) -> None:
    print("\n--- Period table ---")
    print(
        "Iter | x_true -> x_next | phi_target(deg) | phi_meas(deg) | "
        "err(deg) | f(kHz) | wrap | x_hat"
    )

    for t, x in enumerate(sim.x_true):
        x_next = (sim.a * int(x)) % sim.N

        print(
            f"{t:4d} | {int(x):5d} -> {x_next:5d} | "
            f"{sim.phi_target[t]:12.3f} | "
            f"{sim.phi_measured[t]:11.3f} | "
            f"{sim.phase_error_deg[t]:8.3f} | "
            f"{sim.freq_khz[t]:7.3f} | "
            f"{int(sim.wraps_used[t]):4d} | "
            f"{int(sim.x_hat[t]):5d}"
        )


def plot_result(
    sim: Simulation,
    outpdf: Path,
    *,
    label_adjustments: dict[int, dict[str, float]],
) -> None:
    f_fine = np.linspace(sim.f_grid_khz[0], sim.f_grid_khz[-1], 200)
    phi_fine = np.interp(f_fine, sim.f_grid_khz, sim.phi_cal_avg)

    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=False)

    ax = axes[0]
    ax.plot(f_fine, wrap_360(phi_fine), linewidth=1.0)
    axhspan_wrapped(ax, sim.phi_measured[0], sim.period_delta_deg, alpha=0.30)
    ax.grid(True)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Phase")
    ax.scatter(sim.freq_khz, sim.phi_measured, s=30, marker="x", linewidths=1.2)

    for i, (f, phi) in enumerate(zip(sim.freq_khz, sim.phi_measured)):
        adj = label_adjustments.get(i, {})
        dx = DEFAULT_LABEL_DX + adj.get("dx", 0.0)
        dy = DEFAULT_LABEL_DY + adj.get("dy", 0.0)

        ax.annotate(
            str(i),
            xy=(f, phi),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax = axes[1]
    iterations = np.arange(len(sim.x_true))
    ax.plot(iterations, sim.phase_error_deg, marker="o", markersize=2, linewidth=1)
    ax.axhline(0.0, linewidth=1)

    if sim.period_data is not None and 0 <= int(sim.period_data) < len(iterations):
        ax.axvline(int(sim.period_data), linewidth=1)

    ax.grid(True)
    ax.set_ylabel("Error")
    ax.set_xlabel("Iteration")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    fig.savefig(outpdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config()
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    f_khz, phi_mat = load_selected_phase(cfg)

    f_min_khz = cfg.fmin_hz / 1000.0
    f_max_khz = cfg.fmax_hz / 1000.0
    anchor_khz = f_min_khz if cfg.anchor_khz is None else cfg.anchor_khz

    cal = build_calibration(
        phase_label=cfg.phase_label,
        phibits=np.array(cfg.phibits, dtype=int),
        f_khz=f_khz,
        phi_mat=phi_mat,
        f_min_khz=f_min_khz,
        f_max_khz=f_max_khz,
        wraps_per_band=cfg.wraps_per_band,
        anchor_khz=anchor_khz,
    )

    print(
        f"Using phase channel: phi_{cal.phase_label} "
        f"averaged over phibits {list(cfg.phibits)}"
    )
    print(
        f"Calibration: W={cal.wraps_per_band}, "
        f"anchor={cal.anchor_khz:.3f} kHz, "
        f"slope_avg={cal.slope_avg:.3g} deg/kHz, "
        f"RMSE_avg={cal.rmse_avg:.3f} deg"
    )

    calib_path = cfg.outdir / f"calibration_params_phi_{cal.phase_label}.txt"
    save_calibration_table(cal, calib_path)
    print(f"Saved calibration table: {calib_path}")

    for case in cfg.cases:
        cal_case = align_calibration_to_initial_state(cal, N=case.N, x0=case.x0)

        sim = run_experiment(
            cal_case,
            N=case.N,
            a=case.a,
            x0=case.x0,
            max_steps=cfg.max_steps,
            min_df_khz=cfg.min_df_khz,
            period_delta_deg=case.delta_deg,
            min_period_steps=cfg.min_period_steps,
        )

        print("\n=== Simulation parameters ===")
        print(f"N={case.N}, a={case.a}, x0={case.x0}")

        if sim.period_data is None:
            print(f"No return detected (Delta={sim.period_delta_deg:.2f} deg)")
        else:
            print(
                f"Detected period = {sim.period_data} "
                f"(Delta={sim.period_delta_deg:.2f} deg, "
                f"min_t={sim.min_period_steps})"
            )

        print_iteration_table(sim)

        outpdf = (
            cfg.outdir
            / f"N{case.N}_a{case.a}_W{cfg.wraps_per_band}_Delta{int(case.delta_deg)}.pdf"
        )

        plot_result(
            sim,
            outpdf,
            label_adjustments=case.label_adjustments,
        )
        print(f"Saved: {outpdf}")

    print(f"\nAll outputs in: {cfg.outdir}")


if __name__ == "__main__":
    main()
