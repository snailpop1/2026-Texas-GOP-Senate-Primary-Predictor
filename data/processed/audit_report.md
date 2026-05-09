# Forecast Audit Report

        - Run ID: `txgop-20260509`
        - As of: `2026-05-09`
        - Release status: `withheld`
        - Forecast publishable: `False`
        - Betting eligible: `False`
        - Reliability: `low` (0.0/100)

        ## Headline Forecast

        - Paxton fair probability: `0.616`
        - Cornyn fair probability: `0.384`
        - Mean Paxton margin: `2.58`
        - 80% margin interval: `-8.67` to `13.76`
        - Forecast withheld: `True`

        ## Blocking Issues

        - calibration:historical_brier_score:insufficient_history
- calibration:historical_calibration_error:insufficient_history
- calibration:historical_interval_coverage:insufficient_history
- calibration:historical_log_loss:insufficient_history
- calibration:primary_poll_margin_abs_error:fail
- county_coverage_incomplete
- county_results_quality_gate_failed
- source_registry_stale:early_vote_turnout
- source_registry_stale:endorsements_events
- source_registry_stale:turnout_signals

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
