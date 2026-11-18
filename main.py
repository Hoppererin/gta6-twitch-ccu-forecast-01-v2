"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01) — v2.

Reads one case /input/<id>.in and writes /output/<id>.out:
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]} for h = 2..24.

Method (standard library only, deterministic, no network):
  The case ships ~40 past big launches ("comparables") with hourly Twitch
  category averages from 14 days before to 3 days after their release day.
  For each comparable we take its release-day curve (hours 1..24 from its
  day_start) and the log-ratio r_h = ln(viewers_h / viewers_1).

  The dominant regularity in these curves is saturation: the bigger a
  launch's hour-1 print, the smaller its later ratios (across the
  comparables, corr(ln(viewers_1), ln(peak/viewers_1)) is about -0.94).
  v1 fitted one global line of r_h on ln(viewers_1) per hour. v2 fits a
  LOCAL line instead: per hour h, a Gaussian-kernel weighted least squares
  of r_h on ln(viewers_1) around the target's own ln(hour-1 print), so the
  forecast is driven by the launches closest in size to the target — the
  regime that matters, since GTA VI's hour 1 will sit at or beyond the top
  of the training range, where the global line is pulled around by tiny
  launches that grew 100x.

  Only clean comparables are used: each comparable's unlock note carries a
  "Tier:" tag, and early-access-contaminated launches (whose release-day
  curve is depressed because the peak happened during early access days)
  and weak analogues are left out. The target is a clean console launch
  with no early access, so its analogues are the clean launches.

  The spread is a normal spread, not empirical quantiles: per hour, sigma
  is the local residuals' interquartile range / 1.349, shrunk 70% toward
  0.30. With only ~26 clean launches, empirical 5%/95% quantiles are noisy
  and too wide (v1's 90% coverage was 0.96-1.00, wider than the honest
  0.90); the shrunk normal spread is better calibrated on every check we
  ran: practice, augmented practice (training plus the other practice
  launches, as the final's comparables hold them), and leave-one-out over
  the clean console launches.

  The forecast for the target case is then
      q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h),
  with p1 the case's hour-1 print and z_tau the standard normal's quantile
  at each level. Medians and slopes are clipped to sane ranges only as
  insurance against wild extrapolation at an hour-1 print beyond the
  training range; the clips do not bind anywhere in the measured range.

  If the case carries no usable comparables, we fall back to the house
  persistence baseline (every hour stays at hour 1's print, spread widening
  like a random walk in log space).

Local practice score (mean pinball loss on the log scale over the four
practice launches, scored against their real release-day curves fetched
from the same public source the kit names, SullyGnome): 0.141, vs 0.220
for v1 and 0.387 for the house baseline on the same cases. Augmented
practice (training also holds the other three practice launches, as the
final's comparables do): 0.139. Leave-one-out over the clean console
launches: 0.178; over clean launches with hour-1 >= 100k: 0.186.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (we ship none, but keep the line
# harmless for parity with the starter).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
     1.6448536269514722]
# Fallback spread, in natural-log units per square-root hour (house baseline).
SIGMA = 0.15
# Local-linear kernel bandwidth, in ln(viewers) units around the target.
BANDWIDTH = 1.0
# The residual sigma is shrunk this far toward SIGMA_TARGET.
SIGMA_SHRINK = 0.7
SIGMA_TARGET = 0.30
SIGMA_FLOOR = 0.05
# Insurance clips for the fitted median log-ratio and slope; they do not
# bind anywhere in the measured range.
MU_CLIP = (-1.6, 2.6)
B_CLIP = (-1.5, 0.5)
# Kernel weights below this are dropped from the local fit.
MIN_WEIGHT = 1e-3


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(s):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an ascending list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def day_curve(comp):
    """A comparable's release-day hourly viewers, hours 1..24 (None if missing).

    Returns None when the curve has no hour-1 print or fewer than 12 of the
    day's 24 hours (the GTA VI pre-release curve has no release day yet).
    """
    try:
        day_start = parse_time(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_time = {}
    for row in comp.get("hourly") or []:
        try:
            by_time[parse_time(row["from"])] = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
    hours = []
    for h in range(24):
        v = by_time.get(day_start + timedelta(hours=h))
        hours.append(v if isinstance(v, (int, float)) and v > 0 else None)
    if not hours[0]:
        return None
    if sum(1 for v in hours[1:] if v is not None) < 12:
        return None
    return hours


def tier_of(comp):
    """A comparable's tier from its unlock note: 'ea' (early-access
    contaminated), 'weak', or 'comp' (clean). Anything without an explicit
    early-access or weak tag counts as clean: the practice launches the
    final ships as comparables are clean console launches."""
    unlock = comp.get("unlock") or ""
    if "early-access-contaminated" in unlock:
        return "ea"
    if "Tier: weak" in unlock:
        return "weak"
    return "comp"


def weighted_least_squares(xs, ys, ws):
    """Weighted least squares of y on x; returns (intercept, slope)."""
    total = sum(ws)
    if total == 0:
        return 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / total
    my = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    return my - b * mx, b


def local_fit(curves, x0):
    """Per-hour local-linear fit of the log-ratio on ln(hour-1 print).

    `curves` is a list of release-day hour lists (hours 1..24) from clean
    comparables only. Returns a dict hour -> (mu, sigma): the fitted median
    log-ratio at x0 and the shrunk normal spread around it.
    """
    per_hour = {}
    for h in range(2, 25):
        xs = []
        ys = []
        ws = []
        for hours in curves:
            v1 = hours[0]
            v = hours[h - 1]
            if v1 and v:
                x = math.log(v1)
                w = math.exp(-0.5 * ((x - x0) / BANDWIDTH) ** 2)
                if w < MIN_WEIGHT:
                    continue
                xs.append(x)
                ys.append(math.log(v / v1))
                ws.append(w)
        if len(xs) < 3:
            # Too few launches near the target: fall back to an unweighted
            # fit over all clean curves for this hour.
            xs = []
            ys = []
            for hours in curves:
                v1 = hours[0]
                v = hours[h - 1]
                if v1 and v:
                    xs.append(math.log(v1))
                    ys.append(math.log(v / v1))
            ws = [1.0] * len(xs)
        if not xs:
            per_hour[h] = (0.0, SIGMA_TARGET)
            continue
        a, b = weighted_least_squares(xs, ys, ws)
        b = min(max(b, B_CLIP[0]), B_CLIP[1])
        mu = min(max(a + b * x0, MU_CLIP[0]), MU_CLIP[1])
        residuals = sorted(y - (a + b * x) for x, y in zip(xs, ys))
        iqr = quantile(residuals, 0.75) - quantile(residuals, 0.25)
        sigma = max(SIGMA_FLOOR, iqr / 1.349)
        sigma = (1.0 - SIGMA_SHRINK) * sigma + SIGMA_SHRINK * SIGMA_TARGET
        per_hour[h] = (mu, sigma)
    return per_hour


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    if len(levels) != 5:
        levels = LEVELS
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    else:
        # No hour-1 print: fall back to the last pre-checkpoint point.
        history = case.get("history") or []
        p1 = max(1.0, float(history[-1]["viewers"])) if history else 1.0

    curves = []
    for comp in case.get("comparables") or []:
        if tier_of(comp) != "comp":
            continue
        hours = day_curve(comp)
        if hours is None:
            continue
        curves.append(hours)

    hours_out = []
    if len(curves) >= 4:
        fit = local_fit(curves, math.log(p1))
        for row in case.get("forecast") or []:
            h = row["hour"]
            mu, sigma = fit[h]
            qs = [mu + z * sigma for z in Z]
            qs = sorted(qs)
            hours_out.append(
                {"hour": h, "q": [round2(max(1.0, p1 * math.exp(q))) for q in qs]}
            )
    if not hours_out:
        # Fallback: house persistence baseline.
        for row in case.get("forecast") or []:
            h = row["hour"]
            hours_out.append(
                {
                    "hour": h,
                    "q": [
                        round2(max(1.0, p1 * math.exp(z * SIGMA * math.sqrt(h - 1))))
                        for z in Z
                    ],
                }
            )
    if not hours_out:
        # No forecast rows in the case at all: emit hours 2..24 from the baseline.
        for h in range(2, 25):
            hours_out.append(
                {
                    "hour": h,
                    "q": [
                        round2(max(1.0, p1 * math.exp(z * SIGMA * math.sqrt(h - 1))))
                        for z in Z
                    ],
                }
            )
    hours_out.sort(key=lambda r: r["hour"])
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
