"""Media transformations used in marketing mix models.

Two effects make media different from an ordinary regression input:

* Carryover (adstock): an ad seen this week still influences sales in later
  weeks. Modelled here as geometric decay.
* Diminishing returns (saturation): each extra euro buys less extra effect.
  Modelled here as a Hill curve.

These are hand-written on purpose. The baseline model uses them directly,
and they let the simulator plant effects with known parameters.
"""

from __future__ import annotations

import numpy as np


def geometric_adstock(x: np.ndarray, decay: float, max_lag: int | None = None,
                      normalise: bool = True) -> np.ndarray:
    """Apply geometric carryover to a 1-D series of weekly media values.

    adstock[t] = sum_{l=0..L} decay**l * x[t-l]

    Args:
        x: weekly spend or impressions, oldest first.
        decay: share of the effect that carries into the next week, in [0, 1).
        max_lag: how many past weeks to include. Defaults to the full series.
        normalise: divide by the sum of weights so the output stays on the
            same scale as the input. Meridian also normalises its adstock.

    Returns:
        Array of the same length as x.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D; apply per channel and per geo")
    if not 0.0 <= decay < 1.0:
        raise ValueError("decay must be in [0, 1)")

    n = len(x)
    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    weights = decay ** np.arange(max_lag + 1)

    out = np.zeros(n)
    for lag, w in enumerate(weights):
        out[lag:] += w * x[: n - lag]

    if normalise:
        out /= weights.sum()
    return out


def hill_saturation(x: np.ndarray, half_saturation: float,
                    slope: float = 1.0) -> np.ndarray:
    """Hill curve: maps media to an effect between 0 and 1.

    effect = x**slope / (x**slope + half_saturation**slope)

    Args:
        x: non-negative media values (usually already adstocked).
        half_saturation: the media level at which the effect reaches 0.5.
        slope: steepness. slope <= 1 is concave (diminishing from the start);
            slope > 1 is S-shaped (a threshold before media starts working).
    """
    x = np.asarray(x, dtype=float)
    if np.any(x < 0):
        raise ValueError("media values must be non-negative")
    if half_saturation <= 0 or slope <= 0:
        raise ValueError("half_saturation and slope must be positive")

    xs = x ** slope
    return xs / (xs + half_saturation ** slope)
