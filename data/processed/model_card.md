# Model Card

## Intended Use

This package estimates the May 26, 2026 Texas Republican U.S. Senate runoff as a probabilistic research forecast. It is intended for election-analysis review and showcase presentation, not as a guarantee or financial advice.

## Headline

- As of: `2026-05-12`
- Paxton fair probability: `60.8%`
- Cornyn fair probability: `39.2%`
- Mean Paxton margin: `+2.34` points
- 80% interval: `-8.61` to `+13.30` points
- Release status: `withheld`

## Inputs And Adjustments

- Polling baseline: direct runoff polls weighted by recency, sample size, likely-voter population, partisan/internal status, and same-day internal clustering. Clustered internal poll groups: `1`.
- Primary prior: first-round votes plus Hunt-voter transfer evidence. Baseline transfer prior margin: `+1.00` points.
- Candidate strength: capped favorability and supporter-commitment adjustment of `+0.47` margin points.
- Money/media: capped cash and ad-spending adjustment of `-1.89` margin points.
- Uncertainty: base sigma `8.53` points with scenario-specific widening.

## Scenario Outputs

- polling_only: Paxton 71.7%, mean margin 5.2 points.
- polling_plus_primary: Paxton 69.1%, mean margin 4.3 points.
- full_model_mid_turnout: Paxton 60.8%, mean margin 2.3 points.
- low_turnout: Paxton 65.0%, mean margin 3.6 points.
- high_turnout: Paxton 54.5%, mean margin 1.1 points.
- repeat_primary_poll_error: Paxton 42.5%, mean margin -1.8 points.

## Turnout Prior

- low_turnout: 758,418 projected voters, +1.25 Paxton margin points.
- mid_turnout: 1,083,455 projected voters, +0.00 Paxton margin points.
- high_turnout: 1,408,492 projected voters, -1.25 Paxton margin points.

## Hunt Transfer Stress

- cornyn_resolves_undecided: 57.4% Hunt-to-Paxton transfer, prior margin +0.5.
- baseline: 59.1% Hunt-to-Paxton transfer, prior margin +1.0.
- paxton_resolves_undecided: 60.8% Hunt-to-Paxton transfer, prior margin +1.5.

## Known Limits

County-level candidate rows remain a release gate until a complete, reconciled county file replaces the statewide placeholder. Historical calibration history is tracked as a reliability penalty rather than a hard publication blocker when current source, duplicate, and quality gates pass.
