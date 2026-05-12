from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.build_dataset import AS_OF, main
from scripts.build_showcase import main as build_showcase


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def test_raw_poll_schema_and_counts() -> None:
    polls = pd.read_csv(RAW_DIR / "polls.csv")
    required_columns = {
        "race",
        "stage",
        "model_group",
        "pollster",
        "start_date",
        "end_date",
        "release_date",
        "sample_size",
        "population",
        "paxton_pct",
        "cornyn_pct",
        "source_url",
    }
    assert required_columns.issubset(polls.columns)
    assert len(polls) >= 35
    assert len(polls[(polls["stage"] == "runoff") & (polls["model_group"] == "core")]) >= 9

    core = polls[polls["model_group"] == "core"]
    assert core[list(required_columns)].notna().all().all()
    assert core["paxton_pct"].between(0, 100).all()
    assert core["cornyn_pct"].between(0, 100).all()


def test_source_registry_covers_all_raw_inputs() -> None:
    registry = pd.read_csv(RAW_DIR / "source_registry.csv")
    assert {
        "dataset_name",
        "source_type",
        "retrieval_method",
        "freshness_sla_days",
        "trust_tier",
        "citation_url",
        "refresh_instructions",
        "notes",
    }.issubset(registry.columns)
    raw_datasets = {path.stem for path in RAW_DIR.glob("*.csv")} - {"source_registry"}
    assert raw_datasets.issubset(set(registry["dataset_name"]))


def test_primary_results_reconcile_to_source_totals() -> None:
    primary = pd.read_csv(RAW_DIR / "primary_results.csv")
    statewide = primary[primary["statewide_or_county"] == "statewide"]
    votes = dict(zip(statewide["candidate"], statewide["votes"]))

    assert votes["John Cornyn"] == 910_382
    assert votes["Ken Paxton"] == 878_564
    assert votes["Wesley Hunt"] == 293_250
    assert 99.5 < statewide["pct"].sum() < 100.5


def test_county_primary_placeholder_reconciles_to_statewide_totals() -> None:
    primary = pd.read_csv(RAW_DIR / "primary_results.csv")
    county = pd.read_csv(RAW_DIR / "county_primary_results.csv")
    features = pd.read_csv(RAW_DIR / "county_features.csv")
    statewide = primary[primary["statewide_or_county"] == "statewide"]
    votes = dict(zip(statewide["candidate"], statewide["votes"]))
    minor_total = statewide.loc[
        ~statewide["candidate"].isin(["John Cornyn", "Ken Paxton", "Wesley Hunt"]), "votes"
    ].sum()
    county_votes = county.groupby("candidate")["votes"].sum()

    assert len(features) == 254
    assert features["county"].nunique() == 254
    assert county_votes["John Cornyn"] == votes["John Cornyn"]
    assert county_votes["Ken Paxton"] == votes["Ken Paxton"]
    assert county_votes["Wesley Hunt"] == votes["Wesley Hunt"]
    assert county_votes["Minor candidates"] == minor_total


def test_pipeline_outputs_valid_probabilities_and_release_contract() -> None:
    result = main()
    scenarios = result["scenarios"]
    full = scenarios[scenarios["scenario"] == "full_model_mid_turnout"].iloc[0]
    release = result["release_status"].iloc[0]

    assert len(scenarios) == 6
    assert 0 <= full["paxton_win_probability"] <= 1
    assert 0 <= full["cornyn_win_probability"] <= 1
    assert abs(full["paxton_win_probability"] + full["cornyn_win_probability"] - 1) < 1e-9
    assert full["margin_80pct_low"] < full["mean_paxton_margin_points"] < full["margin_80pct_high"]
    assert (
        scenarios["paxton_win_probability"] + scenarios["cornyn_win_probability"]
    ).sub(1).abs().lt(1e-9).all()
    assert release["release_status"] in {"published", "withheld"}
    assert 0 <= release["reliability_score"] <= 100
    assert release["reliability_class"] in {"low", "medium", "high"}


def test_required_processed_outputs_exist() -> None:
    required_files = [
        "polls.csv",
        "primary_results.csv",
        "finance_ads.csv",
        "markets.csv",
        "market_timeseries.csv",
        "wager_settings.csv",
        "hunt_transfer.csv",
        "subgroup_signals.csv",
        "candidate_strength.csv",
        "turnout_signals.csv",
        "general_election_polls.csv",
        "shock_scenarios.csv",
        "county_primary_results.csv",
        "county_features.csv",
        "early_vote_turnout.csv",
        "source_checks.csv",
        "turnout_prior.csv",
        "hunt_transfer_scenarios.csv",
        "model_scenarios.csv",
        "market_comparison.csv",
        "wager_value_table.csv",
        "poll_diagnostics.csv",
        "poll_miss_diagnostics.csv",
        "margin_distribution.csv",
        "shock_model.csv",
        "county_turnout_model.csv",
        "county_model_status.csv",
        "runoff_county_projection.csv",
        "data_quality_report.csv",
        "sensitivity.csv",
        "source_manifest.csv",
        "source_registry_status.csv",
        "poll_canonicalization_report.csv",
        "model_ablation.csv",
        "calibration_summary.csv",
        "release_status.csv",
        "snapshot_manifest.csv",
        "model_output.json",
        "audit_report.md",
        "model_card.md",
    ]
    for filename in required_files:
        path = PROCESSED_DIR / filename
        assert path.exists(), f"Missing processed output: {filename}"
        assert path.stat().st_size > 0, f"Empty processed output: {filename}"

    output = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))
    assert output["version"] == "2.0"
    assert output["as_of"] == str(AS_OF.date())
    assert "release_status" in output
    assert "headline_forecast" in output
    assert "model_health" in output
    assert "calibration_metrics" in output
    assert "source_freshness" in output
    assert "betting_decision_summary" in output
    assert "scenario_range" in output


def test_processed_csvs_include_last_updated() -> None:
    result = main()
    assert "wager_value_table" in result
    for path in PROCESSED_DIR.glob("*.csv"):
        frame = pd.read_csv(path)
        assert "last_updated" in frame.columns, f"{path.name} is missing last_updated"
        assert set(frame["last_updated"].dropna()) == {str(AS_OF.date())}


def test_market_and_wager_outputs_are_normalized_and_conservative() -> None:
    main()
    markets = pd.read_csv(PROCESSED_DIR / "market_timeseries.csv")
    total = markets["normalized_paxton_prob"] + markets["normalized_cornyn_prob"] + markets["normalized_other_prob"]
    assert total.between(0.999, 1.001).all()
    assert {"stale_price_flag", "liquidity_warning", "settlement_warning"}.issubset(markets.columns)

    value = pd.read_csv(PROCESSED_DIR / "wager_value_table.csv")
    assert set(value["value_flag"]).issubset({"paxton_value", "cornyn_value", "no_bet_zone"})
    assert (value["required_edge"] > 0).all()
    assert (value["capped_exposure_fraction"] <= 0.05).all()
    assert {"stale_price_flag", "liquidity_warning", "settlement_warning", "block_reasons", "decision_eligible"}.issubset(
        value.columns
    )
    assert (value["value_flag"] == "no_bet_zone").all()


def test_source_metadata_and_quality_outputs() -> None:
    main()
    for path in RAW_DIR.glob("*.csv"):
        frame = pd.read_csv(path)
        if path.name in {"wager_settings.csv", "source_registry.csv"}:
            continue
        assert "source_url" in frame.columns, f"{path.name} is missing source_url"
        assert frame["source_url"].fillna("").astype(str).str.strip().ne("").all()
        note_column = "notes" if "notes" in frame.columns else "model_note" if "model_note" in frame.columns else None
        assert note_column is not None, f"{path.name} is missing notes/model_note"
        assert frame[note_column].fillna("").astype(str).str.strip().ne("").all()

    quality = pd.read_csv(PROCESSED_DIR / "data_quality_report.csv")
    assert {
        "polls",
        "county_results",
        "early_vote",
        "market_prices",
        "finance_ads",
        "endorsements_events",
        "turnout_signals",
        "poll_miss",
    }.issubset(set(quality["signal_group"]))
    assert (quality["missing_source_count"] == 0).all()
    assert "required_for_publish" in quality.columns


def test_release_and_calibration_outputs_reflect_withholding() -> None:
    main()
    release = pd.read_csv(PROCESSED_DIR / "release_status.csv")
    row = release.iloc[0]
    assert not row["forecast_publishable"]
    assert not row["betting_eligible"]
    assert "county_coverage_incomplete" in row["blocking_issues"]
    assert "historical_brier_score" not in row["blocking_issues"]
    assert "historical_brier_score" in row["reliability_penalties"]

    calibration = pd.read_csv(PROCESSED_DIR / "calibration_summary.csv")
    assert {"metric", "threshold", "pass", "status"}.issubset(calibration.columns)
    assert (calibration["status"] == "insufficient_history").any()

    source_status = pd.read_csv(PROCESSED_DIR / "source_registry_status.csv")
    assert {
        "dataset_name",
        "freshness_pass",
        "parse_status",
        "freshness_basis",
        "latest_source_check_date",
    }.issubset(source_status.columns)
    polls = source_status[source_status["dataset_name"] == "polls"].iloc[0]
    assert polls["freshness_pass"]
    assert polls["freshness_basis"] == "source_check"


def test_poll_miss_and_early_vote_layers_are_explicit() -> None:
    main()
    poll_miss = pd.read_csv(PROCESSED_DIR / "poll_miss_diagnostics.csv")
    assert poll_miss["cornyn_poll_error_points"].iloc[0] > 0
    assert poll_miss["repeat_miss_paxton_margin_adjustment"].iloc[0] < 0

    early_vote = pd.read_csv(PROCESSED_DIR / "early_vote_turnout.csv")
    assert set(early_vote["data_status"]) == {"not_started"}
    assert not early_vote["available_for_model"].any()

    projection = pd.read_csv(PROCESSED_DIR / "runoff_county_projection.csv")
    assert set(projection["scenario"]) == {"low", "mid", "high"}
    assert not projection["early_vote_available"].any()


def test_poll_clustering_and_auxiliary_model_outputs() -> None:
    main()
    polls = pd.read_csv(PROCESSED_DIR / "polls.csv")
    assert {"poll_cluster_id", "cluster_size", "cluster_weight", "sponsor_alignment"}.issubset(polls.columns)
    assert (polls.loc[polls["cluster_size"] > 1, "cluster_weight"] < 1).all()

    turnout = pd.read_csv(PROCESSED_DIR / "turnout_prior.csv")
    assert set(turnout["scenario"]) == {"low_turnout", "mid_turnout", "high_turnout"}
    assert (turnout["projected_runoff_votes"] > 0).all()

    hunt = pd.read_csv(PROCESSED_DIR / "hunt_transfer_scenarios.csv")
    assert set(hunt["scenario"]) == {"cornyn_resolves_undecided", "baseline", "paxton_resolves_undecided"}
    assert hunt["paxton_transfer_share"].between(0, 1).all()


def test_as_of_build_is_deterministic_and_showcase_exists() -> None:
    first = main(as_of="2026-05-12")["forecast_package"]["headline_forecast"]
    second = main(as_of="2026-05-12")["forecast_package"]["headline_forecast"]
    assert first == second

    build_showcase()
    showcase = ROOT / "showcase" / "index.html"
    assert showcase.exists()
    text = showcase.read_text(encoding="utf-8")
    assert "Texas GOP Senate Runoff Forecast" in text
    assert "Paxton leads" in text


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
