# Pre-Election Refresh Checklist

Use this checklist daily through the May 26, 2026 runoff, and immediately after any new poll, endorsement, finance/ad update, or early-vote report.

1. Update `data/raw/polls.csv` with any new public runoff poll. Include sample, dates, population, mode, sponsor, source URL, and partisan/internal flag.
2. Update `data/raw/market_timeseries.csv` with timestamped bid, ask, last price, volume, liquidity, fee assumptions, and settlement-rule notes.
3. Update `data/raw/finance_ads.csv` with campaign finance, outside spending, and ad-support changes.
4. Update `data/raw/endorsements_events.csv` with endorsements, withdrawals, major campaign events, and model notes.
5. Starting May 18, update `data/raw/early_vote_turnout.csv` daily by county and party.
6. Replace the `statewide_unallocated` placeholder in `data/raw/county_primary_results.csv` once full county-by-county candidate results are available.
7. Run `python scripts/build_dataset.py`.
8. Run `python -m pytest`.
9. Re-execute `notebooks/tx_gop_senate_runoff_model.ipynb`.
10. Review `data/processed/data_quality_report.csv` first for stale sources or missing metadata.
11. Review `data/processed/wager_value_table.csv`, `data/processed/poll_diagnostics.csv`, `data/processed/poll_miss_diagnostics.csv`, `data/processed/runoff_county_projection.csv`, `data/processed/sensitivity.csv`, and `data/processed/shock_model.csv`.

Do not treat a value flag as a standalone recommendation. It only means the model edge cleared the configured spread, fee, and uncertainty buffers.
