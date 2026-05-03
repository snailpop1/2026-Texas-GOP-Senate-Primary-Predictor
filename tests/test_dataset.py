from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.build_dataset import main


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
    assert core[list(required_columns - {"hunt_pct"})].notna().all().all()
    assert core["paxton_pct"].between(0, 100).all()
    assert core["cornyn_pct"].between(0, 100).all()


def test_primary_results_reconcile_to_source_totals() -> None:
    primary = pd.read_csv(RAW_DIR / "primary_results.csv")
    statewide = primary[primary["statewide_or_county"] == "statewide"]
    votes = dict(zip(statewide["candidate"], statewide["votes"]))

    assert votes["John Cornyn"] == 907_416
    assert votes["Ken Paxton"] == 881_192
    assert votes["Wesley Hunt"] == 292_702
    assert statewide["pct"].sum() > 99.5
    assert statewide["pct"].sum() < 100.5


def test_county_primary_placeholder_reconciles_to_statewide_totals() -> None:
    primary = pd.read_csv(RAW_DIR / "primary_results.csv")
    county = pd.read_csv(RAW_DIR / "county_primary_results.csv")
    statewide = primary[primary["statewide_or_county"] == "statewide"]
    votes = dict(zip(statewide["candidate"], statewide["votes"]))
    minor_total = statewide.loc[
        ~statewide["candidate"].isin(["John Cornyn", "Ken Paxton", "Wesley Hunt"]), "votes"
    ].sum()
    county_votes = county.groupby("candidate")["votes"].sum()

    assert county_votes["John Cornyn"] == votes["John Cornyn"]
    assert county_votes["Ken Paxton"] == votes["Ken Paxton"]
    assert county_votes["Wesley Hunt"] == votes["Wesley Hunt"]
    assert county_votes["Minor candidates"] == minor_total


def test_pipeline_outputs_valid_probabilities() -> None:
    result = main()
    scenarios = result["scenarios"]
    full = scenarios[scenarios["scenario"] == "full_model_mid_turnout"].iloc[0]

    assert len(scenarios) == 5
    assert 0 <= full["paxton_win_probability"] <= 1
    assert 0 <= full["cornyn_win_probability"] <= 1
    assert abs(full["paxton_win_probability"] + full["cornyn_win_probability"] - 1) < 1e-9
    assert full["margin_80pct_low"] < full["mean_paxton_margin_points"] < full["margin_80pct_high"]


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
        "model_scenarios.csv",
        "market_comparison.csv",
        "wager_value_table.csv",
        "poll_diagnostics.csv",
        "margin_distribution.csv",
        "shock_model.csv",
        "county_turnout_model.csv",
        "county_model_status.csv",
        "sensitivity.csv",
        "model_output.json",
    ]
    for filename in required_files:
        path = PROCESSED_DIR / filename
        assert path.exists(), f"Missing processed output: {filename}"
        assert path.stat().st_size > 0, f"Empty processed output: {filename}"

    output = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))
    headline = output["headline"]
    assert headline["as_of"] == "2026-05-03"
    assert 0 <= headline["paxton_fair_probability"] <= 1
    assert 0 <= headline["cornyn_fair_probability"] <= 1


def test_processed_csvs_include_last_updated() -> None:
    result = main()
    assert "wager_value_table" in result
    for path in PROCESSED_DIR.glob("*.csv"):
        frame = pd.read_csv(path)
        assert "last_updated" in frame.columns, f"{path.name} is missing last_updated"
        assert set(frame["last_updated"].dropna()) == {"2026-05-03"}


def test_market_and_wager_outputs_are_normalized() -> None:
    main()
    markets = pd.read_csv(PROCESSED_DIR / "market_timeseries.csv")
    total = markets["normalized_paxton_prob"] + markets["normalized_cornyn_prob"] + markets["normalized_other_prob"]
    assert total.between(0.999, 1.001).all()

    value = pd.read_csv(PROCESSED_DIR / "wager_value_table.csv")
    assert set(value["value_flag"]).issubset({"paxton_value", "cornyn_value", "no_bet_zone"})
    assert (value["required_edge"] > 0).all()
    assert (value["capped_exposure_fraction"] <= 0.05).all()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__]))
