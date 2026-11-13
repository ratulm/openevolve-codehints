"""Problem 4: robust (neighbourhood-averaged) minimisation of a 3-variable landscape.

The landscape itself is visible to both arms; the averaging rule below is not."""
import numpy as np

NAME = "p04_flat_basin"
DESCRIPTION = (
    "Choose one point x in the box [-4, 4]^3 for a given multimodal function g of 3 variables "
    "(a constant 13, minus five Gaussian wells of different depths and widths, plus a small "
    "sine/cosine ripple and a weak quadratic bowl). The point is NOT judged by g(x). Its cost is "
    "the robust value R(x): the mean of g over 27 fixed points x + d, where d runs over the "
    "3x3x3 grid of offsets {-0.5, 0, 0.5}^3. The score is R_ref / R(x) with R_ref about 6.25, "
    "so a broad, moderately low basin beats the narrow well that holds the global minimum of g."
)
TIME_LIMIT_S = 30
DIM = 3
LOW, HIGH = -4.0, 4.0
TOL = 1e-9
STEP = 0.5
OFFSETS = np.array([[i, j, k] for i in (-STEP, 0.0, STEP) for j in (-STEP, 0.0, STEP)
                    for k in (-STEP, 0.0, STEP)])
R_REF = 6.251663935017363  # robust value of the reference point (3 x 60 polished starts), see reference.py

WELL_CENTERS = np.array([
    [1.8, -1.6, 1.2],
    [-2.0, 1.5, -1.0],
    [-1.85, 1.4, -0.9],
    [2.5, 2.4, -2.6],
    [-2.6, -2.4, 2.2],
])
WELL_DEPTHS = np.array([9.0, 6.5, 4.5, 6.0, 4.0])
WELL_WIDTHS = np.array([1.0, 0.8, 0.15, 0.3, 0.8])


def landscape(points):
    """The function both programs carry, vectorised over rows of an (n, 3) array."""
    x = np.atleast_2d(np.asarray(points, dtype=float))
    d = np.sum((x[:, None, :] - WELL_CENTERS[None, :, :]) ** 2, axis=2)
    wells = np.sum(WELL_DEPTHS * np.exp(-d / (2.0 * WELL_WIDTHS ** 2)), axis=1)
    ripple = (0.5 * np.sin(2.3 * x[:, 0] + 1.1 * x[:, 1]) * np.cos(1.7 * x[:, 2] - 0.6 * x[:, 0])
              + 0.3 * np.cos(2.9 * x[:, 1] - 1.3 * x[:, 2]))
    bowl = 0.02 * np.sum(x ** 2, axis=1)
    wall = 50.0 * np.sum(np.maximum(np.abs(x) - 4.0, 0.0), axis=1)
    return 13.0 - wells + ripple + bowl + wall


def robust_value(point):
    """Mean of the landscape over the 27 fixed neighbours of `point`."""
    return float(np.mean(landscape(np.asarray(point, dtype=float)[None, :] + OFFSETS)))


def score(output):
    try:
        x = np.asarray(output, dtype=float)
    except Exception:
        return 0.0, "run() must return a sequence of 3 numbers (the chosen point)."
    if x.shape != (DIM,):
        return 0.0, f"Expected a point of shape ({DIM},); got an array of shape {x.shape}."
    if not np.all(np.isfinite(x)):
        i = int(np.argmin(np.isfinite(x)))
        return 0.0, f"Coordinate {i} of the point is {x[i]}; all coordinates must be finite."
    outside = np.maximum(LOW - x, x - HIGH)
    if np.any(outside > TOL):
        i = int(np.argmax(outside))
        return 0.0, (f"Coordinate {i} of the point is {x[i]:.6f}, which is outside the box "
                     f"[{LOW}, {HIGH}] by {outside[i]:.3e}.")
    value = robust_value(x)
    if not np.isfinite(value) or value <= 0:
        return 0.0, f"The neighbourhood average at the point is {value}; it must be positive and finite."
    return float(R_REF / value), None
