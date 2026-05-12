# Forecast Audit Report

- Run ID: `txgop-20260512`
- As of: `2026-05-12`
- Release status: `withheld`
- Forecast publishable: `False`
- Betting eligible: `False`
- Reliability: `low` (32.1/100)

## Headline Forecast

- Paxton fair probability: `0.608`
- Cornyn fair probability: `0.392`
- Mean Paxton margin: `2.34`
- 80% margin interval: `-8.61` to `13.30`
- Forecast withheld: `True`

## Blocking Issues

- county_coverage_incomplete
- county_results_quality_gate_failed

## Reliability Penalties

- calibration:historical_brier_score:insufficient_history
- calibration:historical_calibration_error:insufficient_history
- calibration:historical_interval_coverage:insufficient_history
- calibration:historical_log_loss:insufficient_history
- calibration:primary_poll_margin_abs_error:fail

These reduce confidence but do not automatically withhold the forecast unless a current data gate fails.

## Model Story

The headline is a probabilistic forecast, not a guaranteed call. It blends clustered direct runoff polling, the first-round result, Hunt-voter transfer evidence, bounded candidate-strength signals, and a capped Cornyn money/media adjustment. Markets are used only as an external price check and never as a core model input.

## Required Data Quality Gates

- polls: stale=False, missing_sources=0, missing_notes=0
- county_results: stale=True, missing_sources=0, missing_notes=0
- market_prices: stale=False, missing_sources=0, missing_notes=0
- finance_ads: stale=False, missing_sources=0, missing_notes=0
- endorsements_events: stale=False, missing_sources=0, missing_notes=0
- turnout_signals: stale=False, missing_sources=0, missing_notes=0

## Betting Decision Summary

- Kalshi: no_bet_zone (edge_below_required_buffer;edge_not_durable_across_scenarios;forecast_withheld;liquidity_unknown;stale_market_price;verify_contract_terms)
- Covers: no_bet_zone (edge_below_required_buffer;edge_not_durable_across_scenarios;forecast_withheld;liquidity_unknown;stale_market_price;verify_contract_terms)
- Polymarket: no_bet_zone (edge_below_required_buffer;edge_not_durable_across_scenarios;forecast_withheld)
