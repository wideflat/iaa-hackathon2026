"""Index-adjust nominal features that inflate with the calendar year (written_premium +3%/yr, sum_insured +3%/yr).

A tree or net trained on 2021-24 has never seen the 2025 level of a drifting nominal column, so rows are pushed into the top bins by inflation alone
while the severity trend already carries inflation. Dividing a column by its median in the row's own year (train and test alike, no outcome used)
makes it stationary. Columns absent from a line's frame are skipped.
"""
import pandas as pd


def deflate_frames(frames: dict, cols) -> dict:
    out = {}
    for line, df in frames.items():
        present = [c for c in cols if c in df.columns]
        out[line] = df.assign(**{c: df[c] / df.groupby("year")[c].transform("median") for c in present})
    return out
