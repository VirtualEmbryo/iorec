"""Display colours for labels and interfaces, the same in every exported file.

Each label gets a hue from the golden-angle sequence, so that any number of labels
receive distinct colours and a label keeps its colour from one file to the next.
"""

from __future__ import annotations

import colorsys

_GOLDEN_FRACTION = 0.6180339887498949
_SATURATION = 0.55
_VALUE = 0.92


def label_color(label: int) -> tuple[float, float, float]:
    """RGB colour of a label, each channel in [0, 1]."""
    hue = (0.11 + label * _GOLDEN_FRACTION) % 1.0
    return colorsys.hsv_to_rgb(hue, _SATURATION, _VALUE)


def interface_color(
    label_a: int,
    label_b: int,
    exterior_label: int | None,
) -> tuple[float, float, float]:
    """RGB colour of the interface between two labels.

    A cell surface facing the exterior takes the colour of the cell; an interface
    between two cells takes the mean of their colours.
    """
    if label_a == exterior_label:
        return label_color(label_b)
    if label_b == exterior_label:
        return label_color(label_a)
    color_a, color_b = label_color(label_a), label_color(label_b)
    r, g, b = ((x + y) / 2 for x, y in zip(color_a, color_b))
    return r, g, b


def to_bytes(color: tuple[float, float, float]) -> tuple[int, int, int]:
    """Convert an RGB colour in [0, 1] to three integers in [0, 255]."""
    r, g, b = (round(255 * channel) for channel in color)
    return r, g, b


def to_hex(color: tuple[float, float, float]) -> str:
    """Convert an RGB colour in [0, 1] to `#RRGGBB`."""
    return "#{:02X}{:02X}{:02X}".format(*to_bytes(color))
