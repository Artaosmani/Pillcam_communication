"""Shared colours and matplotlib style for the thesis visualizers."""
import matplotlib as mpl

# Categorical slots in fixed order (validated default palette, light mode).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#a9a8a2"

# Latency stages, in pipeline order, each with a fixed colour.
STAGES = ["capture", "detection", "network", "mapping", "lookup", "decision"]
STAGE_COLORS = dict(zip(STAGES, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]))
PHYS_COLOR = VIOLET          # t_half_s, physical delivery time

# Particle states (engine order 0..3).
STATE_NAMES = ["free", "bound", "internalized", "degraded"]
STATE_COLORS = [BLUE, ORANGE, AQUA, MUTED]


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "hatch.color": SURFACE, "hatch.linewidth": 1.2,
    })
