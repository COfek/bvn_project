"""One fixed colour per setup for the winner maps (dct_parameter_space.py,
winner_map_tcalc_sensitivity.py), so a setup has the same colour in every figure.

The nine colours were chosen from the Okabe-Ito and Tol palettes to stay apart under
deuteranopia, protanopia and tritanopia (Machado et al. 2009, full severity): every pair
differs by at least 18 CIELAB units under each of the three and under normal vision, and none
is darker than L* = 35, so the black guide lines stay visible on every region. The setup
names are also written on the regions themselves (label_regions), so the maps do not rely on
colour alone.
"""
from __future__ import annotations

import numpy as np

COLORS = {
    "Hungarian BvN":        "#DDDDDD",
    "Hungarian Euler d1":   "#99DDFF",
    "Hungarian Euler d2":   "#CC79A7",
    "Hungarian Euler d3":   "#F0E442",
    "Hungarian Euler d4":   "#0072B2",
    "GW Dynamic Euler d4":  "#CC3311",
    "GW Static Euler d4":   "#AAAA00",
    "GW Static B4":         "#AA3377",
    "GW Static B2":         "#EE6677",
}
FALLBACK = "#777777"


def color_of(label: str) -> str:
    return COLORS.get(label, FALLBACK)


def _dark(hexcol: str) -> bool:
    r, g, b = (int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45


def label_regions(ax, codes, X, Y, names, min_share=0.03, fontsize=6, short=None):
    """Write each region's setup name on it.

    codes : 2-D array of region indices (same shape as X, Y; log-spaced grids)
    names : names[i] is the setup drawn with index i
    Regions below min_share of the plane are left to the legend. The text is set vertically
    for a strip spanning most of the height, and smaller and horizontal for a region
    under 3% of the plane.
    """
    lx, ly = np.log10(X), np.log10(Y)
    x0, x1, y0, y1 = lx.min(), lx.max(), ly.min(), ly.max()
    for i in np.unique(codes):
        mask = codes == i
        if mask.mean() < min_share:
            continue
        fx = (lx[mask] - x0) / (x1 - x0)
        fy = (ly[mask] - y0) / (y1 - y0)
        cx, cy = fx.mean(), fy.mean()
        # the centroid of a non-convex region can fall outside it: take its nearest cell
        k = np.argmin((fx - cx) ** 2 + (fy - cy) ** 2)
        cx, cy = fx[k], fy[k]
        tall = (fy.max() - fy.min()) > 0.6      # a full-height strip: vertical text
        small = mask.mean() < 0.03                 # a corner region: smaller, horizontal
        name = names[i] if short is None else short(names[i])
        if small:                                  # start at the region's left edge
            cx = fx.min() + 0.01
        ax.text(cx, cy, name, transform=ax.transAxes, ha="left" if small else "center",
                va="center",
                rotation=90 if tall else 0, fontsize=fontsize * (0.8 if small else 1),
                color="white" if _dark(color_of(names[i])) else "black")
