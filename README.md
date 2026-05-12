# 2026 Texas GOP Senate Primary Predictor

Reproducible data science project for the May 26, 2026 Texas Republican U.S. Senate runoff between Ken Paxton and John Cornyn.

The project combines public polling, March primary results, Hunt-voter transfer evidence, campaign finance/ad signals, turnout context, source-check metadata, and prediction-market snapshots into a transparent win-probability model. It is built for auditability: every input row has a source URL, the model is implemented in a repeatable Python pipeline, and the pipeline now emits a release-gated forecast package, source manifest, immutable raw snapshot manifest, model card, audit report, and static showcase dashboard for each refresh.

This is an election-forecasting research project, not financial advice. Prediction markets and political betting are risky, volatile, and jurisdiction-dependent.

## Headline Result

As of May 12, 2026, the internal full model estimate is:

| Candidate | Internal Fair Probability |
| --- | ---: |
| Ken Paxton | 60.8% |
| John Cornyn | 39.2% |

The model mean is Paxton +2.3 points, with an intentionally wide 80% interval from Cornyn +8.6 to Paxton +13.3. Scenario outputs range from Cornyn 57.5% in a repeat-primary-poll-error stress test to Paxton 71.7% in a polling-only view. The forecast package currently marks the run as `withheld` because county coverage is incomplete; historical calibration limitations now reduce reliability rather than automatically blocking publication.

## What This Demonstrates

- End-to-end election data pipeline design with raw and processed datasets.
- Transparent model construction using weighted polling, priors, scenario analysis, and uncertainty intervals.
- Source reconciliation against reported primary results.
- Sensitivity testing across pollsters and partisan/internal polls.
- Clear separation between model estimates and market prices.
- Wager-risk tooling with default no-bet behavior, explicit block reasons, market-friction buffers, shock tests, and capped stake-sizing math.
- Release gating via calibration, source freshness, county-coverage, and market-safety checks.
- Immutable raw snapshot manifest and source registry status for each build.
- Static `showcase/index.html` dashboard for polished local presentation.
- Reproducible analysis suitable for review in a notebook or command line.

## Repository Structure

```text
data/
  raw/                 Original hand-entered public-source snapshots
  processed/           Generated normalized tables and model outputs
notebooks/
  tx_gop_senate_runoff_model.ipynb
scripts/
  build_dataset.py     Validation, normalization, model, and output pipeline
  build_showcase.py    Static HTML dashboard generator
showcase/
  index.html           Generated pitch-ready forecast dashboard
tests/
  test_dataset.py      Reproducibility and schema checks
docs/
  pre_election_checklist.md
```

## Quick Start

Install dependencies:

```powershell
python3 -m pip install -r requirements.txt
```

Rebuild the processed data and model output:

```powershell
python3 scripts/build_dataset.py --as-of 2026-05-12
python3 scripts/build_showcase.py
```

Run tests:

```powershell
python3 -m pytest
```

Review `data/processed/model_output.json`, `data/processed/audit_report.md`, `data/processed/model_card.md`, and `showcase/index.html` first. The notebook remains exploratory and secondary to the generated forecast package.

## Data Snapshot

As of May 12, 2026:

- March 3 Republican primary: Cornyn 910,382 votes / 42.0%; Paxton 878,564 / 40.5%; Hunt 293,250 / 13.5%. These statewide candidate totals reconcile to the TXElectionResults 2,166,910 GOP ballot total.
- Poll file contains 39 rows: 12 direct post-primary runoff polls, 22 primary polls, and 5 pre-primary Paxton-Cornyn head-to-head/context rows.
- Core runoff polls include TPOR, PPP, Change Research, Impact Research, Quantus, GQR, Peak Insights, co/efficient, University of Houston Hobby School, and Remington Research.
- Expanded signals include McLaughlin second-choice data, Change Research crosstabs, TPOR and UH Hunt-voter transfer, UH/TSU and UH Hobby subgroup rows, turnout-market ranges, and general-election/electability polling.
- Market comparison includes Kalshi, Polymarket, TexPolls, and Covers/Kalshi snapshots.
- Market time series includes timestamp, bid/ask/last where available, normalized implied probabilities, stale-price flags, settlement warnings, and liquidity/volume fields, including a May 11 Polymarket API snapshot.
- Money/media inputs include FEC-style campaign finance reporting from Texas Tribune plus AdImpact ad-spending reporting.
- County features now cover all 254 Texas counties with TXElectionResults party lean and historical vote-volume context. Candidate-level county results still reconcile through a `statewide_unallocated` placeholder until complete county rows are imported from a reliable source.
- Poll-miss diagnostics compare final pre-primary polling against the March 3 result, including Cornyn's primary overperformance and a repeat-miss stress test.

## Model Summary

The model is intentionally transparent rather than black-box:

- Polling baseline: recency-, sample-, likely-voter-, partisan-, and same-day internal-cluster-weighted average of direct post-primary runoff polls.
- Primary prior: March first-round result blended with Hunt-voter transfer evidence.
- Hunt-transfer aggregate: multiple sources estimate Hunt voters at about 47% Paxton, 32% Cornyn, and 21% unresolved; unresolved Hunt voters are modeled as slightly Paxton-leaning.
- Turnout scenarios: low turnout modestly benefits Paxton; high turnout modestly benefits Cornyn.
- Candidate-strength adjustment: small capped Paxton boost from favorability, retention, and supporter commitment signals.
- Money/media adjustment: capped Cornyn boost for cash and ad-spending advantage.
- Market comparison: prediction markets are excluded from the core model and used only as an external price check.
- Wager decision layer: defaults to `no_bet_zone` unless forecast publication, calibration, freshness, liquidity, settlement, and scenario-durability gates all pass. Fractional Kelly and flat bankroll settings are included as research math, not as a recommendation.
- Shock model: stress-tests late Trump endorsement, Hunt endorsement, major new poll, repeat primary polling miss, turnout collapse/surge, money saturation, market-liquidity, and campaign/legal-event variance.

## Wager-Readiness Layer

The project now produces additional outputs for disciplined price comparison:

| Output | Purpose |
| --- | --- |
| `data/processed/poll_diagnostics.csv` | Effective poll count, weighted standard error, MoE-derived uncertainty, partisan/nonpartisan splits. |
| `data/processed/poll_miss_diagnostics.csv` | Final pre-primary polling average vs. actual March 3 result, including repeat-miss stress input. |
| `data/processed/margin_distribution.csv` | Monte Carlo margin distribution by scenario. |
| `data/processed/shock_model.csv` | Stress-test probabilities under late endorsement, poll, or campaign-event shocks. |
| `data/processed/market_timeseries.csv` | Normalized market snapshots with settlement, stale-price, and liquidity warnings. |
| `data/processed/wager_value_table.csv` | Safety-gated market decision table with default no-bet behavior and explicit block reasons. |
| `data/processed/county_turnout_model.csv` | County-turnout framework using primary totals, Hunt transfer, and low/mid/high retention assumptions. |
| `data/processed/runoff_county_projection.csv` | Low/mid/high runoff vote projection by loaded county row, with early-vote availability flags. |
| `data/processed/early_vote_turnout.csv` | Daily early-vote table ready for May 18-22 county updates. |
| `data/processed/turnout_prior.csv` | Explicit low/mid/high runoff retention assumptions before early vote starts. |
| `data/processed/hunt_transfer_scenarios.csv` | Cornyn/base/Paxton unresolved-Hunt-voter transfer stress table. |
| `data/processed/data_quality_report.csv` | Row counts, source freshness, missing-source checks, stale flags, and model-use status by signal group. |
| `data/processed/source_manifest.csv` | Canonical manifest of every raw dataset used in the run, including file hashes. |
| `data/processed/source_checks.csv` | Source-check log for cases where public sources were checked but no new model row exists. |
| `data/processed/source_registry_status.csv` | Source freshness and parse-status table derived from `data/raw/source_registry.csv`. |
| `data/processed/calibration_summary.csv` | Calibration and reliability summary; historical insufficiency is a penalty, not a hard source gate. |
| `data/processed/model_ablation.csv` | Bounded component-by-component impact on the headline forecast. |
| `data/processed/release_status.csv` | Publish/withhold decision, reliability score, and blocking issues. |
| `data/processed/snapshot_manifest.csv` | Immutable raw snapshot inventory for the run. |
| `data/processed/audit_report.md` | Human-readable audit report for the forecast package. |
| `data/processed/model_card.md` | Concise model-card explanation of intended use, inputs, adjustments, scenarios, and limits. |
| `showcase/index.html` | Self-contained static dashboard for local demos and showcase review. |

Current limitation: the county turnout framework is structurally ready, but it is not yet a true county-by-county projection because the raw county candidate rows have not been imported. Until `data/raw/county_primary_results.csv` is replaced with official county rows, county output should be treated as a scaffold and validation target, not a local geographic signal. Under the current conservative release rules, that limitation is enough to keep the forecast package in a withheld state.

## Current Scenario Output

| Scenario | Paxton | Cornyn | Mean Margin |
| --- | ---: | ---: | ---: |
| Polling only | 71.7% | 28.3% | Paxton +5.2 |
| Polling + primary prior | 69.1% | 30.9% | Paxton +4.3 |
| Full model, mid turnout | 60.8% | 39.2% | Paxton +2.3 |
| Low turnout | 65.0% | 35.0% | Paxton +3.6 |
| High turnout | 54.5% | 45.5% | Paxton +1.1 |
| Repeat primary polling miss | 42.5% | 57.5% | Cornyn +1.8 |

## Source List

- TXElectionResults primary results: https://txelectionresults.com/primaries/2026
- TexPolls/AP results and market tracker: https://texpolls.com/races/us-senate-2026
- 270toWin polling table: https://www.270towin.com/2026-senate-polls/texas
- RealClearPolling GOP primary table: https://www.realclearpolitics.com/epolls/2026/senate/tx/2026_texas_senate_republican_primary-8724.html
- University of Houston Hobby School report: https://www.uh.edu/hobby/primary2026/senate.pdf
- UH/TSU Texas Trends 2025 report: https://www.uh.edu/hobby/txtrends/election2026.pdf
- Emerson/Nexstar January poll: https://emersoncollegepolling.com/texas-2026-poll/
- Stratus Intelligence/DDHQ memo: https://data.ddhq.io/polls/2025/11/24/Stratus%20Intelligence/Wesley%20Hunt-Texas-2025-11-21-2025-11-22
- J.L. Partners December memo: https://static1.squarespace.com/static/663253f0bc87b7070102f41f/t/693335eb1a3f392a5295aa2a/1764963819245/Texas%2BGOP%2BPrimary%2Bpolling%2B-%2BPRESS%2BRELEASE.pdf
- Chism/Blueprint polling reports: https://crm.capitolinside.com/polls226x.html
- Peak Insights runoff report reference: https://capitolinside.com/poll418x.html
- Polymarket GOP primary winner market: https://polymarket.com/event/texas-republican-senate-primary-winner/will-ken-paxton-win-the-2026-republican-primary
- GQR/Senate Majority Project memo: https://senatemajority.com/wp-content/uploads/SMP-GQR-Texas-Republican-Primary-Memo.pdf
- Texas Tribune fundraising report: https://www.texastribune.org/2026/04/15/john-cornyn-ken-paxton-runoff-first-quarter-fundraising/
- AdImpact ad analysis: https://adimpact.com/blogs/tx-senate-ad-analysis-the-most-expensive-senate-primary-on-record
- Covers/Kalshi odds writeup: https://www.covers.com/politics/texas-senate-odds
- AP CPAC/Paxton event context: https://apnews.com/article/075d6eff33890921319ac73bd853986b
- Change Research Texas memo: https://changeresearch.com/wp-content/uploads/2026/03/Texas-Memo-_-March-2026.pdf
- KERA/TPOR April runoff report: https://www.keranews.org/elections-2026/2026-04-17/paxton-cornyn-poll-republican-primary-runoff-texas-senate-race
- UH Hobby School May runoff report: https://www.texastribune.org/wp-content/uploads/2026/05/UH-GOP-runoff-poll.pdf
- Public polling table for late Remington/Peak runoff polls: https://en.wikipedia.org/wiki/2026_United_States_Senate_election_in_Texas
- Spectrum Hunt-voter report: https://spectrumlocalnews.com/tx/austin/news/2026/04/23/texas-poll-hunt-senate-primary-runoff-senate
- McLaughlin/DDHQ second-choice data: https://data.ddhq.io/polls/2026/03/01/McLaughlin-%26-Associates-Texas
- Impact Research/Talarico poll report: https://www.newsmax.com/politics/talarico-texas-senate/2026/03/20/id/1250261/
- PPP/Senate Majority PAC electability memo: https://senatemajority.com/wp-content/uploads/signal-2026-03-06-122044.pdf
- TXElectionResults primary turnout/results: https://txelectionresults.com/primaries/2026
- Polymarket event API: https://gamma-api.polymarket.com/events?slug=texas-republican-senate-primary-winner

## Refresh Notes

When a new poll, market snapshot, or no-new-data source check appears:

1. Add the source row to the relevant `data/raw/*.csv`.
2. If a source was checked and has no new usable row, add the check to `data/raw/source_checks.csv`.
3. Keep `source_url` and `notes` populated.
4. Run `python3 scripts/build_dataset.py --as-of 2026-05-12`.
5. Run `python3 scripts/build_showcase.py`.
6. Run `python3 -m pytest`.
7. Review `data/processed/release_status.csv`, `data/processed/calibration_summary.csv`, and `data/processed/data_quality_report.csv`.
8. Review `data/processed/model_output.json`, `data/processed/audit_report.md`, `data/processed/model_card.md`, and `showcase/index.html`.
9. Reopen or rerun the notebook only if you need exploratory analysis.

If President Trump endorses either candidate, add it to `data/raw/endorsements_events.csv`; do not change the quantitative model until there is evidence from polling or observed market movement.

For the full daily workflow, see `docs/pre_election_checklist.md`.

## License And Data Attribution

Original code in this repository is released under the MIT License. Public election, polling, finance, market, and media data remains attributable to the original sources listed above and in the CSV `source_url` fields.
