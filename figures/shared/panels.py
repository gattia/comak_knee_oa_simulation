"""Panel letters (A, B, C ...) for multi-panel figures.

Placement is in *offset points* from each axes' top-left corner rather than in
axes fractions, so a letter lands in the same visual spot no matter how wide the
y tick labels are. With axes-fraction placement the letters in a multi-row figure
drift horizontally row by row, which is obvious once the rows have different
y-axis magnitudes.

    from shared import panels

    i = panels.letter_axes(top_row)            # A B C
    i = panels.letter_axes(bottom_row, start=i)  # D E F
    panels.add_panel_letter(ax, "G")             # one-off
"""

from __future__ import annotations

import string

__all__ = ["LETTERS", "DEFAULT_FONTSIZE", "DEFAULT_OFFSET",
           "add_panel_letter", "letter_axes"]

LETTERS = list(string.ascii_uppercase)

DEFAULT_FONTSIZE = 22
# The +6 pt rise matches matplotlib's default title pad, so on rows that have a
# title the letter's baseline sits inline with it.
DEFAULT_OFFSET = (-48, 6)


def add_panel_letter(ax, letter, fontsize=DEFAULT_FONTSIZE, offset=DEFAULT_OFFSET,
                     **kwargs):
    """Bold panel letter at the axes' top-left corner."""
    return ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                       xytext=offset, textcoords="offset points",
                       ha="left", va="baseline", fontsize=fontsize,
                       weight="bold", annotation_clip=False, **kwargs)


def letter_axes(axes, start=0, **kwargs):
    """Letter a sequence of axes consecutively; returns the next unused index."""
    for i, ax in enumerate(axes):
        add_panel_letter(ax, LETTERS[start + i], **kwargs)
    return start + len(axes)
