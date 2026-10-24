"""Office-premium build-up for the 20 reference policies (report exhibit), from a prediction file and the emerging `expenses` table.

  python scripts/p2_office_premium_exhibit.py --candidate submissions/p2_t3_robust_gbm_tabm.csv

Writes docs/part2/office_premium_reference.csv and a Markdown table. The risk premium of a row is its predicted claim cost (already for the row's exposure);
see iaa.office_premium for the formula and assumptions (fixed expense pro-rated by exposure; no comparison-site policy is among the 20).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd  # noqa: E402
from iaa import office_premium as op  # noqa: E402

EM = ROOT / "data" / "part2" / "emerging"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    a = ap.parse_args()
    ref = pd.read_parquet(EM / "reference_policies.parquet")
    pred = pd.read_csv(ROOT / a.candidate, dtype={"row_id": str}).set_index("row_id")["prediction"]
    exp = pd.read_parquet(EM / "expenses.parquet")
    assump = op.assumptions(exp, "emerging")
    t = op.office_premium(ref["row_id"].map(pred), ref["line"], ref["channel"], ref["exposure"], assump)
    t.insert(0, "row_id", ref["row_id"].to_numpy())
    t.insert(1, "line", ref["line"].to_numpy())
    t.insert(2, "channel", ref["channel"].to_numpy())
    t.insert(3, "exposure", ref["exposure"].to_numpy())
    t["written_premium"] = ref["written_premium"].to_numpy()
    t["office_vs_written"] = t["office_premium"] / t["written_premium"]
    t.to_csv(ROOT / "docs" / "part2" / "office_premium_reference.csv", index=False)
    show = t[["row_id", "line", "channel", "exposure", "risk_premium", "cat_reinsurance", "fixed_expense", "variable_expense", "commission", "premium_tax", "profit_margin", "office_premium", "written_premium", "office_vs_written"]]
    print(show.round(2).to_string(index=False))
    print("\nassumptions:", {k: v for k, v in assump.items() if k != "commission_comparison_site"})


if __name__ == "__main__":
    main()
