"""Cross-line covariates of the same customer in the same year (label-free: policy attributes only, no claim or cost, so train, validation and test rows may be computed together).

For every row: the customer's written premium in each of the three lines that year (NaN = not held), the sum insured of the buildings and contents policies, the vehicle value group
of the motor policy, and the number of lines held. A customer-year with two policies of one line (at most two) contributes its larger value.

add_crossline(frames) returns new frames; the inputs are not modified.
"""
import pandas as pd

from .config import LINES

_KEYS = ["customer_id", "year"]


def customer_year_table(frames: dict) -> pd.DataFrame:
    parts = []
    for line in LINES:
        d = frames[line]
        cols = {"written_premium": f"x_prem_{line}"}
        if line in ("buildings", "contents"):
            cols["sum_insured"] = f"x_si_{line}"
        if line == "motor":
            cols["vehicle_value_group"] = "x_vvg_motor"
        g = d.groupby(_KEYS)[list(cols)].max().rename(columns=cols)
        parts.append(g)
    table = pd.concat(parts, axis=1)
    table["x_lines_held"] = table[[f"x_prem_{line}" for line in LINES]].notna().sum(axis=1).astype("int8")
    return table.reset_index()


def add_crossline(frames: dict) -> dict:
    table = customer_year_table(frames)
    out = {}
    for line, d in frames.items():
        merged = d.merge(table, on=_KEYS, how="left", validate="many_to_one")
        assert len(merged) == len(d), "cross-line merge changed the row count"
        out[line] = merged
    return out
