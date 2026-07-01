"""
Estilo de figuras
"""
import matplotlib as mpl
import matplotlib.pyplot as plt


PALETTE = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
BLUE, ORANGE, GREEN, RED = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3]


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman",
                       "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "lines.linewidth": 1.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
