# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/geishakampret/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/geishakampret/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), after the first hour — as five quantiles an hour, for the lowest pinball loss on the log scale.

## How it works (v2)

`main.py` (Python 3.13, standard library only, no network) reads the one case it is given and forecasts hours 2–24 from the case's own data:

- Every case ships ~40 past big launches ("comparables") with hourly Twitch category averages around their release day. For each one we take its release-day curve and the log-ratio `r_h = ln(viewers_h / viewers_1)`.
- The dominant regularity is saturation: the bigger a launch's hour-1 print, the smaller its later ratios (across the comparables, `corr(ln(viewers_1), ln(peak/viewers_1)) ≈ -0.94`). So per hour `h` we fit a **local-linear** model of `r_h` on `ln(viewers_1)`: a Gaussian-kernel weighted least squares around the target's own hour-1 print (bandwidth 1.0 in log units). The forecast is driven by the launches closest in size to the target — the regime that matters, since GTA VI's hour 1 will sit at or beyond the top of the training range, where a global line is pulled around by tiny launches that grew 100×.
- Only **clean comparables** are used: each comparable's unlock note carries a "Tier:" tag, and early-access-contaminated launches (whose release-day curve is depressed because the peak happened during early-access days) and weak analogues are left out. The target is a clean console launch with no early access, so its analogues are the clean launches.
- The spread is a **normal spread, not empirical quantiles**: per hour, sigma is the local residuals' interquartile range / 1.349, shrunk 70% toward 0.30. With only ~26 clean launches, empirical 5%/95% quantiles are noisy and too wide (v1's 90% coverage was 0.96–1.00, wider than the honest 0.90); the shrunk normal spread is better calibrated on every check we ran.
- If a case carries no usable comparables, the program falls back to the house persistence baseline (every hour stays at hour 1's print, spread widening like a random walk in log space).

Local practice score (mean pinball loss on the log scale over the four practice launches, scored against their real release-day curves from the same public source the kit names, SullyGnome): **0.141**, vs 0.220 for v1 and 0.387 for the house baseline on the same cases. Augmented practice (training also holds the other three practice launches, as the final's comparables do): 0.139. Leave-one-out over the clean console launches: 0.178; over clean launches with hour-1 ≥ 100k: 0.186. Two runs on the same case write byte-identical output, in ~0.1 s per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` with `/input/<id>.in` and expects `/output/<id>.out`: `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2–24.

---

<sub>Entered on [make.fun](https://make.fun/?ref=dhj53q2r&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
