from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.build_dataset import main


ROOT = Path(__file__).resolve().parents[1]
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
        "hunt_transfer.csv",
        "subgroup_signals.csv",
        "candidate_strength.csv",
        "turnout_signals.csv",
        "general_election_polls.csv",
        "model_scenarios.csv",
        "market_comparison.csv",
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
