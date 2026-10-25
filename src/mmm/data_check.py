"""Evening-1 readiness check for a real MMM dataset.

Answers the questions that decide how the project is framed, before any
modelling:

1. Is it a clean weekly panel (one row per geo per week, no gaps)?
2. Which media columns are spend and which are exposure (impressions/views)?
   Without spend there is no direct ROI.
3. Is there a population or size column per geo? Meridian's geo model uses it.
4. Which control variables exist (price, promotions, holidays)?

Usage:
    python -m mmm.data_check data/raw/<file>.csv --geo Division --date Calendar_Week --kpi Sales
"""

from __future__ import annotations

import argparse
import re

import pandas as pd

SPEND_PAT = re.compile(r"spend|cost|investment|budget", re.I)
EXPOSURE_PAT = re.compile(r"impression|view|click|reach|grp|imps", re.I)
CONTROL_PAT = re.compile(r"price|promo|discount|holiday|event|competitor", re.I)
POP_PAT = re.compile(r"population|pop|size|households", re.I)


def check(df: pd.DataFrame, geo: str, date: str, kpi: str) -> None:
    df = df.copy()
    df[date] = pd.to_datetime(df[date], errors="coerce", dayfirst=True)
    bad_dates = df[date].isna().sum()

    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Unparseable dates: {bad_dates}")
    print(f"Geos: {df[geo].nunique()}   Weeks: {df[date].nunique()}   "
          f"Range: {df[date].min():%Y-%m-%d} to {df[date].max():%Y-%m-%d}")

    # Panel completeness
    counts = df.groupby(geo)[date].nunique()
    full = df[date].nunique()
    incomplete = counts[counts < full]
    dupes = df.duplicated([geo, date]).sum()
    print(f"\nDuplicate geo-week rows: {dupes}")
    print(f"Geos missing some weeks: {len(incomplete)} of {len(counts)}")
    if len(incomplete):
        print(incomplete.sort_values().head(10).to_string())

    # Weekly spacing
    gaps = pd.Series(sorted(df[date].dropna().unique())).diff().dt.days.value_counts()
    print(f"\nSpacing between dates (days -> count):\n{gaps.to_string()}")

    # Column roles
    numeric = [c for c in df.select_dtypes("number").columns if c != kpi]
    spend = [c for c in numeric if SPEND_PAT.search(c)]
    exposure = [c for c in numeric if EXPOSURE_PAT.search(c) and c not in spend]
    controls = [c for c in df.columns if CONTROL_PAT.search(c)]
    pop = [c for c in df.columns if POP_PAT.search(c)]
    other = [c for c in numeric if c not in spend + exposure + controls + pop]

    print("\nColumn roles (by name, verify by eye):")
    print(f"  Spend:      {spend or 'NONE FOUND'}")
    print(f"  Exposure:   {exposure or 'none'}")
    print(f"  Controls:   {controls or 'NONE FOUND'}")
    print(f"  Population: {pop or 'NONE FOUND'}")
    print(f"  Unassigned numeric: {other}")

    print("\nDecisions this implies:")
    if not spend:
        print("  - No spend columns: ROI can't be computed directly. Either impute spend\n"
              "    from a stated CPM assumption, or report effect per 1,000 impressions.")
    if not pop:
        print("  - No population column: Meridian's geo model needs one. Use a proxy\n"
              "    (e.g. average KPI per geo) or equal weights, and state the assumption.")
    if not controls:
        print("  - No controls: media may absorb price/promo/holiday effects. Add a\n"
              "    holiday calendar at minimum and note the limitation.")

    # Quick summaries
    cols = [kpi] + spend + exposure
    print("\nSummary of KPI and media:")
    print(df[cols].describe().T[["mean", "std", "min", "max"]].round(1).to_string())
    print("\nShare of zero weeks per media column:")
    print((df[spend + exposure] == 0).mean().round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--geo", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--kpi", required=True)
    a = p.parse_args()
    check(pd.read_csv(a.path), a.geo, a.date, a.kpi)
