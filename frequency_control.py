"""Sequential affine frequency control in degrees and kHz (Eqs. 27–29)."""

from fractions import Fraction
from math import ceil


def frequency_steps(slope, intercept, f_min, f_max, min_df, multiplier,
                    initial_phase, updates):
    """Yield frequency, wrapped model target and branch, including initialization."""
    # Repeated multiplication amplifies floating-point roundoff. Keep the
    # model frequency exact and convert to float only for readout/reporting.
    s, c = Fraction(float(slope)), Fraction(float(intercept))
    if s <= 0 or min_df <= 0:
        raise ValueError("Frequency control requires positive slope and increment.")
    frequency = Fraction(float(f_min))
    upper = Fraction(float(f_max))
    increment = Fraction(float(min_df))
    period = 360 / s
    yield float(frequency), float(initial_phase), round((s * frequency + c - initial_phase) / 360)

    for step in range(updates):
        if step == 0:
            base = (multiplier * initial_phase - c) / s
        else:
            base = multiplier * frequency + (multiplier - 1) * c / s
        branch = ceil((frequency + increment - base) / period)
        frequency = base + branch * period
        if frequency > upper:
            break
        wraps, target = divmod(s * frequency + c, 360)
        yield float(frequency), float(target), wraps
