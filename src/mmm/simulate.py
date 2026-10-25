"""Generate weekly geo-level MMM data with known ground truth.

This is the validation set. Because every channel's true decay, saturation
and effect size are planted, you can check whether a fitted model recovers
them. Predictive fit alone cannot tell you that: a model can fit sales well
and still split the credit between channels wrongly.

Two spend regimes are supported:

* exogenous: spend is planned independently of sales (the easy case most
  tutorials assume).
* endogenous: part of each week's spend reacts to last week's sales, as real
  budgets often do. This correlates spend with demand and is the usual reason
  MMMs over-credit media. Fitting both regimes shows how much it matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .transforms import geometric_adstock, hill_saturation


@dataclass
class ChannelTruth:
    name: str
    mean_weekly_spend: float      # average spend per geo per week
    decay: float                  # adstock carryover, in [0, 1)
    half_saturation: float        # in units of adstocked spend
    slope: float                  # Hill slope
    beta: float                   # sales at full saturation, per geo per week


@dataclass
class SimConfig:
    n_geos: int = 20
    n_weeks: int = 156
    base_sales: float = 50_000.0
    seasonality_amplitude: float = 0.15
    trend_per_year: float = 0.03
    noise_sd: float = 0.05            # multiplicative noise on sales
    endogenous_share: float = 0.0     # 0 = exogenous; e.g. 0.3 = 30% of spend chases last week's sales
    seed: int = 42
    channels: list[ChannelTruth] = field(default_factory=lambda: [
        ChannelTruth("tv", 8_000, decay=0.6, half_saturation=8_000, slope=2.0, beta=18_000),
        ChannelTruth("search", 4_000, decay=0.1, half_saturation=3_000, slope=1.0, beta=14_000),
        ChannelTruth("social", 3_000, decay=0.3, half_saturation=4_000, slope=1.0, beta=6_000),
        ChannelTruth("display", 2_000, decay=0.2, half_saturation=3_000, slope=1.0, beta=1_000),
    ])


def simulate(cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (observed, truth).

    observed: what a practitioner sees — geo, week, population, spend per
        channel, price index, promo flag, sales.
    truth: per-channel planted parameters plus the realised incremental
        sales and true ROI over the whole simulation.
    """
    rng = np.random.default_rng(cfg.seed)
    weeks = pd.date_range("2023-01-02", periods=cfg.n_weeks, freq="W-MON")
    t = np.arange(cfg.n_weeks)

    rows = []
    incremental = {c.name: 0.0 for c in cfg.channels}
    total_spend = {c.name: 0.0 for c in cfg.channels}

    for g in range(cfg.n_geos):
        population = rng.integers(200_000, 2_000_000)
        size = population / 1_000_000  # scales baseline, spend and effects

        season = 1 + cfg.seasonality_amplitude * np.sin(2 * np.pi * t / 52)
        trend = 1 + cfg.trend_per_year * t / 52
        price = 1 + 0.05 * rng.standard_normal(cfg.n_weeks)
        promo = (rng.random(cfg.n_weeks) < 0.1).astype(int)
        baseline = (cfg.base_sales * size * season * trend
                    * price ** -1.5 * (1 + 0.12 * promo))

        # Planned spend: smooth, with flighting (some weeks dark)
        planned = {}
        for c in cfg.channels:
            on = rng.random(cfg.n_weeks) > 0.15
            level = c.mean_weekly_spend * size * rng.lognormal(0, 0.35, cfg.n_weeks)
            planned[c.name] = level * on

        # Build sales week by week so endogenous spend can react to last week
        spend = {c.name: np.zeros(cfg.n_weeks) for c in cfg.channels}
        sales = np.zeros(cfg.n_weeks)
        effects = {c.name: np.zeros(cfg.n_weeks) for c in cfg.channels}
        expected_base = baseline.mean()

        for w in range(cfg.n_weeks):
            for c in cfg.channels:
                s = planned[c.name][w]
                if cfg.endogenous_share > 0 and w > 0:
                    ratio = sales[w - 1] / expected_base
                    s = s * (1 - cfg.endogenous_share) + s * cfg.endogenous_share * ratio
                spend[c.name][w] = s

            media_total = 0.0
            for c in cfg.channels:
                ad = geometric_adstock(spend[c.name][: w + 1], c.decay, max_lag=12)[-1]
                eff = c.beta * size * hill_saturation(np.array([ad]),
                                                      c.half_saturation * size,
                                                      c.slope)[0]
                effects[c.name][w] = eff
                media_total += eff

            noise = rng.normal(1, cfg.noise_sd)
            sales[w] = (baseline[w] + media_total) * noise

        for c in cfg.channels:
            incremental[c.name] += effects[c.name].sum()
            total_spend[c.name] += spend[c.name].sum()

        for w in range(cfg.n_weeks):
            row = {"geo": f"geo_{g:02d}", "week": weeks[w], "population": population,
                   "price_index": price[w], "promo": promo[w], "sales": sales[w]}
            for c in cfg.channels:
                row[f"spend_{c.name}"] = spend[c.name][w]
            rows.append(row)

    observed = pd.DataFrame(rows)
    truth = pd.DataFrame([
        {"channel": c.name, "decay": c.decay, "half_saturation_per_million_pop": c.half_saturation,
         "slope": c.slope, "beta_per_million_pop": c.beta,
         "total_spend": total_spend[c.name],
         "incremental_sales": incremental[c.name],
         "true_roi": incremental[c.name] / total_spend[c.name]}
        for c in cfg.channels
    ])
    return observed, truth


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Write a synthetic MMM dataset with known truth.")
    p.add_argument("--endogenous-share", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = p.parse_args()

    cfg = SimConfig(endogenous_share=args.endogenous_share)
    obs, truth = simulate(cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = "endogenous" if args.endogenous_share > 0 else "exogenous"
    obs.to_csv(args.out / f"sim_{tag}.csv", index=False)
    truth.to_csv(args.out / f"sim_{tag}_truth.csv", index=False)
    print(truth.round(3).to_string(index=False))
