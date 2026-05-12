# Pre-Election Refresh Checklist

Use this checklist daily through the May 26, 2026 runoff, and immediately after any new poll, endorsement, finance/ad update, or early-vote report.

1. Update `data/raw/polls.csv` with any new public runoff poll. Include sample, dates, population, mode, sponsor, source URL, and partisan/internal flag.
2. Update `data/raw/market_timeseries.csv` with timestamped bid, ask, last price, volume, liquidity, fee assumptions, and settlement-rule notes.
3. Update `data/raw/finance_ads.csv` with campaign finance, outside spending, and ad-support changes.
4. Update `data/raw/endorsements_events.csv` with endorsements, withdrawals, major campaign events, and model notes.
5. Record checked-but-no-new-data sources in `data/raw/source_checks.csv`; this is how freshness passes without inventing a new data row.
6. Starting May 18, update `data/raw/early_vote_turnout.csv` daily by county and party.
7. Replace the `statewide_unallocated` placeholder in `data/raw/county_primary_results.csv` once complete county-by-county candidate results are available and reconcile to statewide totals.
8. Run `python3 scripts/build_dataset.py --as-of YYYY-MM-DD`.
9. Run `python3 scripts/build_showcase.py`.
10. Run `python3 -m pytest`.
11. Review `data/processed/release_status.csv`, `data/processed/calibration_summary.csv`, and `data/processed/data_quality_report.csv` before looking at any market output.
12. Review `data/processed/model_output.json`, `data/processed/audit_report.md`, `data/processed/model_card.md`, `data/processed/source_manifest.csv`, and `showcase/index.html`.
13. Re-execute `notebooks/tx_gop_senate_runoff_model.ipynb` only if you need supporting exploratory analysis.
14. Review `data/processed/wager_value_table.csv`, `data/processed/poll_diagnostics.csv`, `data/processed/poll_miss_diagnostics.csv`, `data/processed/runoff_county_projection.csv`, `data/processed/sensitivity.csv`, and `data/processed/shock_model.csv`.

Do not treat a value flag as a standalone recommendation. It only means the model edge cleared the configured spread, fee, and uncertainty buffers.
