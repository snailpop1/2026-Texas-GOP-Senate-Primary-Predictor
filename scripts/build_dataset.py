from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import textwrap
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
AS_OF = pd.Timestamp("2026-05-09")
RUNOFF_ELECTION_DATE = pd.Timestamp("2026-05-26")
EARLY_VOTE_START = pd.Timestamp("2026-05-18")
STALE_MARKET_DAYS = 2
RNG_SEED = 20260509
FORECAST_PACKAGE_VERSION = "2.0"


def read_raw() -> Dict[str, pd.DataFrame]:
    return {
        "polls": pd.read_csv(RAW_DIR / "polls.csv"),
        "primary_results": pd.read_csv(RAW_DIR / "primary_results.csv"),
        "markets": pd.read_csv(RAW_DIR / "markets.csv"),
        "finance_ads": pd.read_csv(RAW_DIR / "finance_ads.csv"),
        "endorsements_events": pd.read_csv(RAW_DIR / "endorsements_events.csv"),
        "hunt_transfer": pd.read_csv(RAW_DIR / "hunt_transfer.csv"),
        "subgroup_signals": pd.read_csv(RAW_DIR / "subgroup_signals.csv"),
        "candidate_strength": pd.read_csv(RAW_DIR / "candidate_strength.csv"),
        "turnout_signals": pd.read_csv(RAW_DIR / "turnout_signals.csv"),
        "general_election_polls": pd.read_csv(RAW_DIR / "general_election_polls.csv"),
        "market_timeseries": pd.read_csv(RAW_DIR / "market_timeseries.csv"),
        "wager_settings": pd.read_csv(RAW_DIR / "wager_settings.csv"),
        "shock_scenarios": pd.read_csv(RAW_DIR / "shock_scenarios.csv"),
        "county_primary_results": pd.read_csv(RAW_DIR / "county_primary_results.csv"),
        "county_features": pd.read_csv(RAW_DIR / "county_features.csv"),
        "early_vote_turnout": pd.read_csv(RAW_DIR / "early_vote_turnout.csv"),
        "source_registry": pd.read_csv(RAW_DIR / "source_registry.csv"),
    }


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def latest_date_value(frame: pd.DataFrame, columns: Iterable[str]) -> str:
    dates = []
    for column in columns:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            if parsed.notna().any():
                dates.append(parsed.max())
    if not dates:
        return ""
    return str(max(dates).date())


def setting_value(settings: pd.Series, name: str, default: float) -> float:
    value = settings.get(name, default)
    if pd.isna(value):
        return default
    return float(value)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.astype(float), weights=weights.astype(float)))


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    average = weighted_mean(values, weights)
    variance = np.average((values.astype(float) - average) ** 2, weights=weights.astype(float))
    return float(math.sqrt(max(variance, 0.0)))


def effective_sample_count(weights: pd.Series) -> float:
    w = weights.astype(float)
    denom = float((w**2).sum())
    if denom == 0:
        return 0.0
    return float((w.sum() ** 2) / denom)


def validate_source_registry(raw: Dict[str, pd.DataFrame]) -> None:
    registry = raw["source_registry"]
    required = [
        "dataset_name",
        "source_type",
        "retrieval_method",
        "freshness_sla_days",
        "trust_tier",
        "citation_url",
        "refresh_instructions",
        "notes",
    ]
    require_columns(registry, required, "source_registry")
    expected = set(raw) - {"source_registry"}
    actual = set(registry["dataset_name"])
    missing = expected - actual
    if missing:
        raise AssertionError(f"source_registry missing datasets: {sorted(missing)}")


def validate_source_metadata(raw: Dict[str, pd.DataFrame]) -> None:
    required_sources = [name for name in raw if name not in {"wager_settings", "source_registry"}]
    for name in required_sources:
        frame = raw[name]
        if "source_url" not in frame.columns:
            raise ValueError(f"{name} is missing source_url")
        source_missing = frame["source_url"].fillna("").astype(str).str.strip().eq("")
        if source_missing.any():
            raise AssertionError(f"{name} has {int(source_missing.sum())} rows missing source_url")

        note_column = "notes" if "notes" in frame.columns else "model_note" if "model_note" in frame.columns else None
        if note_column is None:
            raise ValueError(f"{name} is missing notes or model_note")
        notes_missing = frame[note_column].fillna("").astype(str).str.strip().eq("")
        if notes_missing.any():
            raise AssertionError(f"{name} has {int(notes_missing.sum())} rows missing {note_column}")


def validate_primary_results(primary: pd.DataFrame) -> None:
    require_columns(primary, ["candidate", "votes", "pct"], "primary_results")
    statewide = primary[primary["statewide_or_county"] == "statewide"].copy()
    expected_votes = {
        "John Cornyn": 907416,
        "Ken Paxton": 881192,
        "Wesley Hunt": 292702,
    }
    for candidate, votes in expected_votes.items():
        actual = int(statewide.loc[statewide["candidate"] == candidate, "votes"].iloc[0])
        if actual != votes:
            raise AssertionError(f"{candidate} votes changed: expected {votes}, got {actual}")
    pct_sum = statewide["pct"].sum()
    if not 99.5 <= pct_sum <= 100.5:
        raise AssertionError(f"Statewide primary percentages sum to {pct_sum:.2f}, expected about 100.")


def validate_polls(polls: pd.DataFrame) -> None:
    required = [
        "pollster",
        "sponsor",
        "start_date",
        "end_date",
        "sample_size",
        "population",
        "paxton_pct",
        "cornyn_pct",
        "source_url",
        "model_group",
    ]
    require_columns(polls, required, "polls")
    core = polls[polls["model_group"] == "core"]
    missing = core[required].isna().any(axis=1)
    if missing.any():
        bad = core.loc[missing, ["pollster", "release_date"]].to_dict("records")
        raise AssertionError(f"Core polls have missing required fields: {bad}")
    if len(core) < 5:
        raise AssertionError("Expected at least five core runoff polls.")


def validate_hunt_transfer(hunt_transfer: pd.DataFrame) -> None:
    required = [
        "pollster",
        "start_date",
        "end_date",
        "sample_size",
        "population",
        "paxton_pct",
        "cornyn_pct",
        "usable_in_model",
        "source_url",
    ]
    require_columns(hunt_transfer, required, "hunt_transfer")
    usable = hunt_transfer[hunt_transfer["usable_in_model"].map(to_bool)]
    missing = usable[required].isna().any(axis=1)
    if missing.any():
        bad = usable.loc[missing, ["pollster", "release_date"]].to_dict("records")
        raise AssertionError(f"Usable Hunt-transfer rows have missing required fields: {bad}")
    if len(usable) < 4:
        raise AssertionError("Expected at least four usable Hunt-transfer data points.")


def validate_county_primary_results(county_results: pd.DataFrame, primary: pd.DataFrame) -> None:
    required = ["county", "candidate", "party", "votes", "data_quality", "source_url"]
    require_columns(county_results, required, "county_primary_results")
    votes = county_results.copy()
    votes["votes"] = pd.to_numeric(votes["votes"], errors="coerce").fillna(0)
    county_totals = votes.groupby("candidate")["votes"].sum()

    statewide = primary[primary["statewide_or_county"] == "statewide"].copy()
    statewide_totals = dict(zip(statewide["candidate"], statewide["votes"]))
    minor_total = statewide.loc[
        ~statewide["candidate"].isin(["John Cornyn", "Ken Paxton", "Wesley Hunt"]), "votes"
    ].sum()
    expected = {
        "John Cornyn": statewide_totals["John Cornyn"],
        "Ken Paxton": statewide_totals["Ken Paxton"],
        "Wesley Hunt": statewide_totals["Wesley Hunt"],
        "Minor candidates": minor_total,
    }
    for candidate, expected_votes in expected.items():
        actual = int(county_totals.get(candidate, 0))
        if actual != int(expected_votes):
            raise AssertionError(
                f"county_primary_results does not reconcile for {candidate}: "
                f"expected {int(expected_votes)}, got {actual}"
            )


def process_polls(polls: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    out = polls.copy()
    for column in ["start_date", "end_date", "release_date"]:
        out[column] = pd.to_datetime(out[column])

    out["poll_midpoint"] = out["start_date"] + (out["end_date"] - out["start_date"]) / 2
    out["days_old"] = (as_of - out["poll_midpoint"]).dt.days.clip(lower=0)
    out["two_candidate_paxton_share"] = out["paxton_pct"] / (out["paxton_pct"] + out["cornyn_pct"])
    out["two_candidate_margin"] = 200 * out["two_candidate_paxton_share"] - 100
    out["canonical_poll_id"] = (
        out["pollster"].astype(str).str.strip().str.lower()
        + "|"
        + out["sponsor"].astype(str).str.strip().str.lower()
        + "|"
        + out["start_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + out["end_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + out["population"].astype(str).str.strip().str.upper()
    )

    recency_half_life_days = 21.0
    out["recency_weight"] = np.exp(-np.log(2) * out["days_old"] / recency_half_life_days)
    out["sample_weight"] = np.sqrt(out["sample_size"] / 600.0)
    out["population_weight"] = np.where(out["population"].str.upper().eq("LV"), 1.0, 0.75)
    out["partisan_weight"] = np.where(out["partisan_or_internal"].map(to_bool), 0.70, 1.0)
    out["model_weight"] = (
        out["recency_weight"] * out["sample_weight"] * out["population_weight"] * out["partisan_weight"]
    )
    return out


def process_hunt_transfer(hunt_transfer: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    out = hunt_transfer.copy()
    for column in ["start_date", "end_date", "release_date"]:
        out[column] = pd.to_datetime(out[column])
    out["poll_midpoint"] = out["start_date"] + (out["end_date"] - out["start_date"]) / 2
    out["days_old"] = (as_of - out["poll_midpoint"]).dt.days.clip(lower=0)
    out["recency_weight"] = np.exp(-np.log(2) * out["days_old"] / 35.0)
    out["sample_weight"] = np.sqrt(out["sample_size"] / 600.0)
    out["population_weight"] = np.where(out["population"].str.upper().eq("LV"), 1.0, 0.75)
    out["partisan_weight"] = np.where(out["partisan_or_internal"].map(to_bool), 0.70, 1.0)
    out["model_weight"] = (
        out["recency_weight"] * out["sample_weight"] * out["population_weight"] * out["partisan_weight"]
    )
    out.loc[~out["usable_in_model"].map(to_bool), "model_weight"] = 0.0
    out["explicit_two_candidate_paxton_share"] = out["paxton_pct"] / (out["paxton_pct"] + out["cornyn_pct"])
    return out


def process_market_timeseries(market_timeseries: pd.DataFrame) -> pd.DataFrame:
    out = market_timeseries.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    numeric_columns = [
        "paxton_bid",
        "paxton_ask",
        "paxton_last",
        "paxton_implied_prob",
        "cornyn_bid",
        "cornyn_ask",
        "cornyn_last",
        "cornyn_implied_prob",
        "other_implied_prob",
        "fee_assumption_pct",
        "volume",
        "liquidity",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["paxton_mid"] = np.where(
        out[["paxton_bid", "paxton_ask"]].notna().all(axis=1),
        (out["paxton_bid"] + out["paxton_ask"]) / 2,
        np.nan,
    )
    out["cornyn_mid"] = np.where(
        out[["cornyn_bid", "cornyn_ask"]].notna().all(axis=1),
        (out["cornyn_bid"] + out["cornyn_ask"]) / 2,
        np.nan,
    )
    out["paxton_market_prob"] = out["paxton_mid"].fillna(out["paxton_last"]).fillna(out["paxton_implied_prob"])
    out["cornyn_market_prob"] = out["cornyn_mid"].fillna(out["cornyn_last"]).fillna(out["cornyn_implied_prob"])
    out["other_implied_prob"] = out["other_implied_prob"].fillna(0.0)
    total = out["paxton_market_prob"] + out["cornyn_market_prob"] + out["other_implied_prob"]
    out["normalized_paxton_prob"] = out["paxton_market_prob"] / total
    out["normalized_cornyn_prob"] = out["cornyn_market_prob"] / total
    out["normalized_other_prob"] = out["other_implied_prob"] / total
    out["paxton_spread"] = out["paxton_ask"] - out["paxton_bid"]
    out["cornyn_spread"] = out["cornyn_ask"] - out["cornyn_bid"]
    out["average_spread"] = out[["paxton_spread", "cornyn_spread"]].mean(axis=1)
    out["days_stale"] = (AS_OF - out["timestamp"]).dt.days.clip(lower=0)
    out["stale_price_flag"] = out["days_stale"] > STALE_MARKET_DAYS
    out["liquidity_warning"] = np.select(
        [
            out["liquidity"].notna() & (out["liquidity"] < 10_000),
            out["liquidity"].isna() & out["volume"].isna(),
        ],
        ["thin_liquidity", "liquidity_unknown"],
        default="ok",
    )
    out["settlement_warning"] = np.where(
        out["settlement_notes"].fillna("").astype(str).str.lower().str.contains("verify"),
        "verify_contract_terms",
        "ok",
    )
    return out


def process_early_vote_turnout(early_vote: pd.DataFrame) -> pd.DataFrame:
    out = early_vote.copy()
    for column in ["election_date", "report_date"]:
        out[column] = pd.to_datetime(out[column])
    for column in ["early_in_person_votes", "mail_votes", "total_early_votes"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    calculated_total = out["early_in_person_votes"].fillna(0) + out["mail_votes"].fillna(0)
    out["total_early_votes"] = out["total_early_votes"].fillna(calculated_total.where(calculated_total > 0))
    out["days_until_early_vote"] = (EARLY_VOTE_START - AS_OF).days
    out["available_for_model"] = AS_OF >= EARLY_VOTE_START
    out["data_status"] = np.select(
        [
            out["total_early_votes"].notna(),
            AS_OF < EARLY_VOTE_START,
        ],
        ["reported", "not_started"],
        default="missing_after_start",
    )
    return out


def aggregate_hunt_transfer(hunt_transfer: pd.DataFrame) -> Dict[str, float]:
    usable = hunt_transfer[hunt_transfer["model_weight"] > 0].copy()
    paxton_pct = weighted_mean(usable["paxton_pct"], usable["model_weight"])
    cornyn_pct = weighted_mean(usable["cornyn_pct"], usable["model_weight"])
    undecided_pct = weighted_mean(usable["undecided_other_pct"], usable["model_weight"])
    explicit_two_way = weighted_mean(usable["explicit_two_candidate_paxton_share"], usable["model_weight"])

    unresolved_to_paxton = 0.55
    paxton_transfer = paxton_pct + unresolved_to_paxton * undecided_pct
    cornyn_transfer = cornyn_pct + (1.0 - unresolved_to_paxton) * undecided_pct
    total = paxton_transfer + cornyn_transfer
    paxton_transfer_share = paxton_transfer / total

    return {
        "hunt_paxton_explicit_pct": float(paxton_pct),
        "hunt_cornyn_explicit_pct": float(cornyn_pct),
        "hunt_undecided_other_pct": float(undecided_pct),
        "hunt_explicit_two_candidate_paxton_share": float(explicit_two_way),
        "hunt_unresolved_to_paxton_assumption": unresolved_to_paxton,
        "hunt_final_paxton_transfer_share": float(paxton_transfer_share),
        "hunt_final_cornyn_transfer_share": float(1.0 - paxton_transfer_share),
    }


def primary_transfer_prior(primary: pd.DataFrame, hunt_transfer: pd.DataFrame) -> Dict[str, float]:
    statewide = primary[primary["statewide_or_county"] == "statewide"].copy()
    votes = dict(zip(statewide["candidate"], statewide["votes"]))
    paxton = votes["Ken Paxton"]
    cornyn = votes["John Cornyn"]
    hunt = votes["Wesley Hunt"]
    transfer = aggregate_hunt_transfer(hunt_transfer)
    minor = statewide.loc[
        ~statewide["candidate"].isin(["Ken Paxton", "John Cornyn", "Wesley Hunt"]), "votes"
    ].sum()
    paxton_runoff = paxton + transfer["hunt_final_paxton_transfer_share"] * hunt + 0.50 * minor
    cornyn_runoff = cornyn + transfer["hunt_final_cornyn_transfer_share"] * hunt + 0.50 * minor
    paxton_share = paxton_runoff / (paxton_runoff + cornyn_runoff)
    return {
        "paxton_share": float(paxton_share),
        "two_candidate_margin": float(200 * paxton_share - 100),
        "top_two_primary_margin": float(200 * (paxton / (paxton + cornyn)) - 100),
        **transfer,
    }


def candidate_strength_adjustment(candidate_strength: pd.DataFrame) -> Dict[str, float]:
    pivot = candidate_strength.pivot_table(
        index=["source", "pollster", "release_date", "population"],
        columns=["metric", "candidate"],
        values="value_pct",
        aggfunc="first",
    )

    net_diffs = []
    for _, row in pivot.iterrows():
        try:
            paxton_net = row[("favorable", "Paxton")] - row[("unfavorable", "Paxton")]
            cornyn_net = row[("favorable", "Cornyn")] - row[("unfavorable", "Cornyn")]
        except KeyError:
            continue
        if pd.notna(paxton_net) and pd.notna(cornyn_net):
            net_diffs.append(float(paxton_net - cornyn_net))
    avg_net_diff = float(np.mean(net_diffs)) if net_diffs else 0.0

    turnout_metrics = candidate_strength[
        candidate_strength["metric"].isin(
            ["very_likely_to_vote_supporters", "certain_to_vote_supporters", "definite_choice", "strong_supporter"]
        )
    ]
    turnout_diffs = []
    for (_, metric), group in turnout_metrics.groupby(["source", "metric"]):
        values = dict(zip(group["candidate"], group["value_pct"]))
        if "Paxton" in values and "Cornyn" in values:
            turnout_diffs.append(float(values["Paxton"] - values["Cornyn"]))
    avg_turnout_diff = float(np.mean(turnout_diffs)) if turnout_diffs else 0.0

    favorability_component = np.clip(avg_net_diff / 60.0, -0.60, 0.60)
    turnout_component = np.clip(avg_turnout_diff / 40.0, -0.40, 0.40)
    total_adjustment = float(np.clip(favorability_component + turnout_component, -0.90, 0.90))
    return {
        "avg_paxton_minus_cornyn_net_favorability": avg_net_diff,
        "avg_paxton_minus_cornyn_commitment_metric": avg_turnout_diff,
        "candidate_strength_margin_adjustment": total_adjustment,
    }


def money_media_adjustment(finance_ads: pd.DataFrame) -> Dict[str, float]:
    runoff = finance_ads[finance_ads["stage"] == "runoff"].copy()
    cash = runoff.groupby("candidate_affinity")["cash_on_hand_m"].sum(min_count=1)
    cornyn_cash = float(cash.get("Cornyn", 0.0))
    paxton_cash = float(cash.get("Paxton", 0.0))
    cash_ratio = (cornyn_cash + 0.25) / (paxton_cash + 0.25)
    cornyn_cash_edge = -min(1.5, 0.80 * math.log(cash_ratio))

    ad_spend = runoff.groupby("candidate_affinity")["ad_spending_m"].sum(min_count=1)
    cornyn_runoff_air = float(ad_spend.get("Cornyn", 0.0))
    paxton_runoff_air = float(ad_spend.get("Paxton", 0.0))
    air_ratio = (cornyn_runoff_air + 0.05) / (paxton_runoff_air + 0.05)
    cornyn_air_edge = -min(0.9, 0.30 * math.log(air_ratio))

    total_adjustment = cornyn_cash_edge + cornyn_air_edge
    return {
        "cornyn_cash_m": cornyn_cash,
        "paxton_cash_m": paxton_cash,
        "cash_ratio_cornyn_to_paxton": cash_ratio,
        "cornyn_runoff_air_m": cornyn_runoff_air,
        "paxton_runoff_air_m": paxton_runoff_air,
        "money_media_margin_adjustment": float(total_adjustment),
    }


def simulate_probability(mean_margin: float, sigma: float, draws: int = 200_000) -> Dict[str, float]:
    rng = np.random.default_rng(RNG_SEED + int(round((mean_margin + 100) * 100)))
    sims = rng.normal(mean_margin, sigma, size=draws)
    paxton_prob = float((sims > 0).mean())
    lo, hi = np.quantile(sims, [0.10, 0.90])
    return {
        "paxton_win_probability": paxton_prob,
        "cornyn_win_probability": 1.0 - paxton_prob,
        "mean_paxton_margin_points": float(mean_margin),
        "margin_80pct_low": float(lo),
        "margin_80pct_high": float(hi),
        "sigma_margin_points": float(sigma),
    }


def build_scenarios(
    polls: pd.DataFrame,
    primary: pd.DataFrame,
    finance_ads: pd.DataFrame,
    hunt_transfer: pd.DataFrame,
    candidate_strength: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    core = polls[polls["model_group"] == "core"].copy()
    poll_margin = weighted_mean(core["two_candidate_margin"], core["model_weight"])
    poll_dispersion = weighted_std(core["two_candidate_margin"], core["model_weight"])
    prior = primary_transfer_prior(primary, hunt_transfer)
    money = money_media_adjustment(finance_ads)
    strength = candidate_strength_adjustment(candidate_strength)
    poll_miss = build_poll_miss_diagnostics(polls, primary)

    base_sigma = max(6.5, poll_dispersion + 3.0)
    polling_plus_primary_margin = 0.78 * poll_margin + 0.22 * prior["two_candidate_margin"]
    full_margin = (
        0.72 * poll_margin
        + 0.18 * prior["two_candidate_margin"]
        + 0.10 * prior["top_two_primary_margin"]
        + strength["candidate_strength_margin_adjustment"]
        + money["money_media_margin_adjustment"]
    )
    repeat_miss_adjustment = (
        float(poll_miss["repeat_miss_paxton_margin_adjustment"].iloc[0])
        if not poll_miss.empty
        else 0.0
    )

    scenario_specs = [
        ("polling_only", poll_margin, base_sigma + 0.5, "Only post-primary direct runoff polls."),
        (
            "polling_plus_primary",
            polling_plus_primary_margin,
            base_sigma,
            "Runoff polls blended with first-round result plus Hunt transfer evidence.",
        ),
        (
            "full_model_mid_turnout",
            full_margin,
            base_sigma,
            "Core estimate: polls, primary prior, Paxton enthusiasm, and Cornyn money/media edge.",
        ),
        (
            "low_turnout",
            full_margin + 1.25,
            base_sigma + 0.75,
            "Lower-turnout runoff scenario; assumes more ideologically intense electorate.",
        ),
        (
            "high_turnout",
            full_margin - 1.25,
            base_sigma + 0.75,
            "Higher-turnout runoff scenario; assumes Cornyn reaches more infrequent/moderate GOP voters.",
        ),
        (
            "repeat_primary_poll_error",
            full_margin + repeat_miss_adjustment,
            base_sigma + 1.0,
            "Stress test only: applies half of the March primary polling-margin miss toward Cornyn.",
        ),
    ]

    rows = []
    for scenario, mean, sigma, note in scenario_specs:
        result = simulate_probability(mean, sigma)
        result["scenario"] = scenario
        result["note"] = note
        rows.append(result)
    scenario_df = pd.DataFrame(rows)[
        [
            "scenario",
            "paxton_win_probability",
            "cornyn_win_probability",
            "mean_paxton_margin_points",
            "margin_80pct_low",
            "margin_80pct_high",
            "sigma_margin_points",
            "note",
        ]
    ]

    diagnostics = {
        "as_of": str(AS_OF.date()),
        "weighted_poll_margin_points": poll_margin,
        "weighted_poll_dispersion_points": poll_dispersion,
        "primary_transfer_prior_margin_points": prior["two_candidate_margin"],
        "top_two_primary_margin_points": prior["top_two_primary_margin"],
        "repeat_primary_poll_error_margin_adjustment": repeat_miss_adjustment,
        **{key: value for key, value in prior.items() if key.startswith("hunt_")},
        **strength,
        **money,
        "base_sigma_points": base_sigma,
    }
    return scenario_df, diagnostics


def process_county_turnout_model(
    county_results: pd.DataFrame,
    county_features: pd.DataFrame,
    hunt_transfer: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    transfer = aggregate_hunt_transfer(hunt_transfer)
    county = county_results.copy()
    county["votes"] = pd.to_numeric(county["votes"], errors="coerce").fillna(0)
    features = county_features.copy()

    rows = []
    candidate_columns = {
        "John Cornyn": "cornyn_primary_votes",
        "Ken Paxton": "paxton_primary_votes",
        "Wesley Hunt": "hunt_primary_votes",
        "Minor candidates": "minor_primary_votes",
    }
    for county_name, group in county.groupby("county", dropna=False):
        row = {"county": county_name}
        for candidate, output_column in candidate_columns.items():
            row[output_column] = float(group.loc[group["candidate"] == candidate, "votes"].sum())
        row["primary_votes"] = sum(row[column] for column in candidate_columns.values())
        first = group.iloc[0]
        row["county_type"] = first.get("county_type", "")
        row["region_group"] = first.get("region_group", "")
        row["metro_area"] = first.get("metro_area", "")
        row["data_quality"] = first.get("data_quality", "")
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.merge(
            features.rename(
                columns={
                    "county_type": "feature_county_type",
                    "region_group": "feature_region_group",
                    "metro_area": "feature_metro_area",
                }
            ),
            on="county",
            how="left",
        )
        out["county_type"] = out["county_type"].replace("", np.nan).fillna(out["feature_county_type"])
        out["region_group"] = out["region_group"].replace("", np.nan).fillna(out["feature_region_group"])
        out["metro_area"] = out["metro_area"].replace("", np.nan).fillna(out["feature_metro_area"])
        out = out.drop(columns=["feature_county_type", "feature_region_group", "feature_metro_area"])

    transfer_paxton = transfer["hunt_final_paxton_transfer_share"]
    transfer_cornyn = transfer["hunt_final_cornyn_transfer_share"]
    out["paxton_transfer_equivalent_votes"] = (
        out["paxton_primary_votes"] + transfer_paxton * out["hunt_primary_votes"] + 0.50 * out["minor_primary_votes"]
    )
    out["cornyn_transfer_equivalent_votes"] = (
        out["cornyn_primary_votes"] + transfer_cornyn * out["hunt_primary_votes"] + 0.50 * out["minor_primary_votes"]
    )
    equivalent_total = out["paxton_transfer_equivalent_votes"] + out["cornyn_transfer_equivalent_votes"]
    out["paxton_transfer_share"] = out["paxton_transfer_equivalent_votes"] / equivalent_total
    out["cornyn_transfer_share"] = out["cornyn_transfer_equivalent_votes"] / equivalent_total
    out["paxton_primary_share"] = out["paxton_primary_votes"] / out["primary_votes"]
    out["cornyn_primary_share"] = out["cornyn_primary_votes"] / out["primary_votes"]
    out["hunt_primary_share"] = out["hunt_primary_votes"] / out["primary_votes"]
    out["gop_share_of_statewide_primary"] = out["primary_votes"] / out["primary_votes"].sum()
    if "runoff_retention_baseline" not in out.columns:
        out["runoff_retention_baseline"] = np.nan

    retention = {"low": 0.35, "mid": 0.50, "high": 0.65}
    for scenario, rate in retention.items():
        baseline = out["runoff_retention_baseline"].fillna(rate)
        out[f"{scenario}_retention_rate"] = baseline
        out[f"{scenario}_turnout_votes"] = out["primary_votes"] * rate
        out[f"{scenario}_paxton_projected_votes"] = out[f"{scenario}_turnout_votes"] * out["paxton_transfer_share"]
        out[f"{scenario}_cornyn_projected_votes"] = out[f"{scenario}_turnout_votes"] * out["cornyn_transfer_share"]
        out[f"{scenario}_paxton_margin_votes"] = (
            out[f"{scenario}_paxton_projected_votes"] - out[f"{scenario}_cornyn_projected_votes"]
        )

    detailed_counties = out[~out["county"].astype(str).eq("statewide_unallocated")]
    statewide_placeholder = out["county"].astype(str).eq("statewide_unallocated").any()
    total_county_rows = int(len(out))
    detailed_count = int(len(detailed_counties))
    coverage_pct = float(detailed_count / 254.0) if total_county_rows else 0.0
    status = pd.DataFrame(
        [
            {
                "county_detail_available": bool(detailed_count > 0),
                "county_rows_with_candidate_votes": detailed_count,
                "candidate_vote_coverage_pct": coverage_pct,
                "statewide_placeholder_present": bool(statewide_placeholder),
                "county_forecast_eligible": bool(detailed_count >= 254 and not statewide_placeholder),
                "current_limitation": (
                    "County-level projection is ready, but only loaded counties should be treated as local signals. "
                    "A statewide placeholder means full county candidate rows are still unavailable."
                ),
                "next_refresh_priority": "Replace statewide_unallocated with full official county rows and add daily early vote once available.",
            }
        ]
    )
    return out, status


def build_runoff_county_projection(county_turnout_model: pd.DataFrame, early_vote: pd.DataFrame) -> pd.DataFrame:
    early = early_vote.copy()
    reported = early[early["data_status"].eq("reported")]
    early_totals = reported.groupby("county", as_index=False)["total_early_votes"].sum()
    early_totals = early_totals.rename(columns={"total_early_votes": "reported_early_votes"})

    rows = []
    for _, county in county_turnout_model.iterrows():
        for scenario in ["low", "mid", "high"]:
            projected_turnout = float(county[f"{scenario}_turnout_votes"])
            rows.append(
                {
                    "county": county["county"],
                    "scenario": scenario,
                    "county_type": county.get("county_type", ""),
                    "region_group": county.get("region_group", ""),
                    "metro_area": county.get("metro_area", ""),
                    "primary_votes": float(county["primary_votes"]),
                    "runoff_retention_rate": float(county[f"{scenario}_retention_rate"]),
                    "projected_turnout_votes": projected_turnout,
                    "projected_paxton_votes": float(county[f"{scenario}_paxton_projected_votes"]),
                    "projected_cornyn_votes": float(county[f"{scenario}_cornyn_projected_votes"]),
                    "projected_paxton_margin_votes": float(county[f"{scenario}_paxton_margin_votes"]),
                    "paxton_transfer_share": float(county["paxton_transfer_share"]),
                    "cornyn_transfer_share": float(county["cornyn_transfer_share"]),
                    "early_vote_available": bool(AS_OF >= EARLY_VOTE_START),
                }
            )
    out = pd.DataFrame(rows)
    if early_totals.empty:
        out["reported_early_votes"] = np.nan
        out["early_vote_share_of_projection"] = np.nan
    else:
        out = out.merge(early_totals, on="county", how="left")
        out["early_vote_share_of_projection"] = out["reported_early_votes"] / out["projected_turnout_votes"]
    return out


def build_poll_diagnostics(polls: pd.DataFrame) -> pd.DataFrame:
    core = polls[polls["model_group"] == "core"].copy()
    weights = core["model_weight"].astype(float)
    margin = core["two_candidate_margin"].astype(float)
    eff_n = effective_sample_count(weights)
    dispersion = weighted_std(margin, weights)
    weighted_se = dispersion / math.sqrt(max(eff_n, 1.0))

    moe = pd.to_numeric(core.get("moe"), errors="coerce")
    moe_margin_sigma = math.sqrt(2.0) * (moe / 1.96)
    normalized_weights = weights / weights.sum()
    moe_derived_se = float(
        np.sqrt(((normalized_weights**2) * (moe_margin_sigma.fillna(moe_margin_sigma.mean()) ** 2)).sum())
    )

    rows = [
        {
            "diagnostic": "core_poll_count",
            "value": float(len(core)),
            "notes": "Direct runoff polls included in the core polling baseline.",
        },
        {
            "diagnostic": "effective_poll_count",
            "value": eff_n,
            "notes": "Weight-adjusted poll count after recency, sample, population, and partisan penalties.",
        },
        {
            "diagnostic": "weighted_mean_paxton_margin",
            "value": weighted_mean(margin, weights),
            "notes": "Two-candidate Paxton margin in points.",
        },
        {
            "diagnostic": "weighted_poll_standard_deviation",
            "value": dispersion,
            "notes": "Weighted spread of core poll margins.",
        },
        {
            "diagnostic": "weighted_standard_error",
            "value": weighted_se,
            "notes": "Polling-only standard error using effective poll count.",
        },
        {
            "diagnostic": "moe_derived_standard_error",
            "value": moe_derived_se,
            "notes": "Approximate margin uncertainty derived from reported poll MoE.",
        },
    ]

    partisan = core[core["partisan_or_internal"].map(to_bool)]
    non_partisan = core[~core["partisan_or_internal"].map(to_bool)]
    if not partisan.empty:
        rows.append(
            {
                "diagnostic": "partisan_internal_weighted_mean_margin",
                "value": weighted_mean(partisan["two_candidate_margin"], partisan["model_weight"]),
                "notes": "Core polling margin using only partisan/internal polls.",
            }
        )
    if not non_partisan.empty:
        rows.append(
            {
                "diagnostic": "nonpartisan_weighted_mean_margin",
                "value": weighted_mean(non_partisan["two_candidate_margin"], non_partisan["model_weight"]),
                "notes": "Core polling margin excluding partisan/internal polls.",
            }
        )
    rows.append(
        {
            "diagnostic": "latest_core_poll_release_date",
            "value": np.nan,
            "notes": str(core["release_date"].max().date()),
        }
    )
    return pd.DataFrame(rows)


def build_poll_miss_diagnostics(polls: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    final_primary = polls[
        (polls["stage"].eq("primary"))
        & (polls["release_date"] < pd.Timestamp("2026-03-03"))
        & (polls["release_date"] >= pd.Timestamp("2026-02-13"))
    ].copy()
    if final_primary.empty:
        return pd.DataFrame()

    statewide = primary[primary["statewide_or_county"].eq("statewide")].copy()
    actual = dict(zip(statewide["candidate"], statewide["pct"]))
    weighted_paxton = weighted_mean(final_primary["paxton_pct"], final_primary["sample_size"])
    weighted_cornyn = weighted_mean(final_primary["cornyn_pct"], final_primary["sample_size"])
    poll_two_way_paxton = weighted_paxton / (weighted_paxton + weighted_cornyn)
    actual_two_way_paxton = actual["Ken Paxton"] / (actual["Ken Paxton"] + actual["John Cornyn"])
    poll_margin = 200 * poll_two_way_paxton - 100
    actual_margin = 200 * actual_two_way_paxton - 100
    margin_miss = actual_margin - poll_margin

    return pd.DataFrame(
        [
            {
                "diagnostic": "final_pre_primary_poll_average",
                "poll_count": int(len(final_primary)),
                "poll_window_start": str(final_primary["release_date"].min().date()),
                "poll_window_end": str(final_primary["release_date"].max().date()),
                "poll_average_paxton_pct": float(weighted_paxton),
                "poll_average_cornyn_pct": float(weighted_cornyn),
                "actual_paxton_pct": float(actual["Ken Paxton"]),
                "actual_cornyn_pct": float(actual["John Cornyn"]),
                "paxton_poll_error_points": float(actual["Ken Paxton"] - weighted_paxton),
                "cornyn_poll_error_points": float(actual["John Cornyn"] - weighted_cornyn),
                "poll_two_candidate_paxton_margin": float(poll_margin),
                "actual_two_candidate_paxton_margin": float(actual_margin),
                "two_candidate_margin_miss_points": float(margin_miss),
                "repeat_miss_fraction_used": 0.50,
                "repeat_miss_paxton_margin_adjustment": float(0.50 * margin_miss),
                "source_url": "https://www.270towin.com/2026-senate-polls/texas",
                "notes": (
                    "Weighted final primary polling average compared with March 3 primary result. "
                    "Negative margin miss means Cornyn overperformed the polling margin."
                ),
            }
        ]
    )


def build_sensitivity(
    polls: pd.DataFrame,
    primary: pd.DataFrame,
    finance_ads: pd.DataFrame,
    hunt_transfer: pd.DataFrame,
    candidate_strength: pd.DataFrame,
) -> pd.DataFrame:
    core = polls[polls["model_group"] == "core"].copy()
    rows = []
    for pollster in sorted(core["pollster"].unique()):
        reduced = polls[~((polls["model_group"] == "core") & (polls["pollster"] == pollster))].copy()
        if len(reduced[reduced["model_group"] == "core"]) < 3:
            continue
        scenarios, _ = build_scenarios(reduced, primary, finance_ads, hunt_transfer, candidate_strength)
        full = scenarios[scenarios["scenario"] == "full_model_mid_turnout"].iloc[0]
        rows.append(
            {
                "removed_pollster": pollster,
                "paxton_win_probability": full["paxton_win_probability"],
                "mean_paxton_margin_points": full["mean_paxton_margin_points"],
            }
        )

    non_partisan = polls[
        ~((polls["model_group"] == "core") & polls["partisan_or_internal"].map(to_bool))
    ]
    scenarios, _ = build_scenarios(non_partisan, primary, finance_ads, hunt_transfer, candidate_strength)
    full = scenarios[scenarios["scenario"] == "full_model_mid_turnout"].iloc[0]
    rows.append(
        {
            "removed_pollster": "all partisan/internal core polls",
            "paxton_win_probability": full["paxton_win_probability"],
            "mean_paxton_margin_points": full["mean_paxton_margin_points"],
        }
    )
    return pd.DataFrame(rows)


def build_market_comparison(markets: pd.DataFrame, scenario_df: pd.DataFrame) -> pd.DataFrame:
    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    paxton_fair = float(full["paxton_win_probability"])
    cornyn_fair = 1.0 - paxton_fair
    out = markets.copy()
    out["paxton_model_edge_vs_market"] = paxton_fair - out["paxton_implied_prob"]
    out["cornyn_model_edge_vs_market"] = cornyn_fair - out["cornyn_implied_prob"]
    out["paxton_value_below_with_5pt_buffer"] = paxton_fair - 0.05
    out["cornyn_value_below_with_5pt_buffer"] = cornyn_fair - 0.05
    return out


def build_margin_distribution(scenario_df: pd.DataFrame, draws: int = 50_000) -> pd.DataFrame:
    rows = []
    bins = np.arange(-30, 32, 2)
    for _, scenario in scenario_df.iterrows():
        scenario_seed = sum(ord(char) for char in str(scenario["scenario"]))
        rng = np.random.default_rng(RNG_SEED + scenario_seed)
        sims = rng.normal(
            float(scenario["mean_paxton_margin_points"]),
            float(scenario["sigma_margin_points"]),
            size=draws,
        )
        counts, edges = np.histogram(sims, bins=bins)
        for count, low, high in zip(counts, edges[:-1], edges[1:]):
            rows.append(
                {
                    "scenario": scenario["scenario"],
                    "margin_bin_low": float(low),
                    "margin_bin_high": float(high),
                    "draw_count": int(count),
                    "density": float(count / draws),
                }
            )
    return pd.DataFrame(rows)


def build_shock_model(scenario_df: pd.DataFrame, shock_scenarios: pd.DataFrame) -> pd.DataFrame:
    base = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    rows = []
    for _, shock in shock_scenarios.iterrows():
        mean = float(base["mean_paxton_margin_points"]) + float(shock["paxton_margin_adjustment"])
        sigma = max(1.0, float(base["sigma_margin_points"]) + float(shock["sigma_adjustment"]))
        result = simulate_probability(mean, sigma, draws=80_000)
        rows.append(
            {
                "shock": shock["shock"],
                "base_mean_paxton_margin_points": float(base["mean_paxton_margin_points"]),
                "paxton_margin_adjustment": float(shock["paxton_margin_adjustment"]),
                "sigma_adjustment": float(shock["sigma_adjustment"]),
                "shock_mean_paxton_margin_points": mean,
                "shock_sigma_margin_points": sigma,
                "paxton_win_probability": result["paxton_win_probability"],
                "cornyn_win_probability": result["cornyn_win_probability"],
                "margin_80pct_low": result["margin_80pct_low"],
                "margin_80pct_high": result["margin_80pct_high"],
                "probability_note": shock["probability_note"],
                "source_url": shock["source_url"],
                "notes": shock["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_poll_canonicalization_report(polls: pd.DataFrame) -> pd.DataFrame:
    core = polls[polls["model_group"] == "core"].copy()
    duplicate_counts = core.groupby("canonical_poll_id").size().reset_index(name="row_count")
    duplicates = duplicate_counts[duplicate_counts["row_count"] > 1]

    overlap_rows = []
    for _, group in core.groupby(["pollster", "sponsor", "population"], dropna=False):
        group = group.sort_values("start_date")
        for i in range(1, len(group)):
            prior = group.iloc[i - 1]
            current = group.iloc[i]
            overlaps = current["start_date"] <= prior["end_date"]
            if overlaps:
                overlap_rows.append(
                    {
                        "pollster": current["pollster"],
                        "sponsor": current["sponsor"],
                        "population": current["population"],
                        "prior_release_date": str(prior["release_date"].date()),
                        "current_release_date": str(current["release_date"].date()),
                    }
                )

    rows = [
        {
            "check": "core_duplicate_canonical_ids",
            "value": int(len(duplicates)),
            "status": "fail" if len(duplicates) else "pass",
            "notes": "Duplicate core poll rows after canonical key normalization.",
        },
        {
            "check": "core_overlapping_same_house_polls",
            "value": int(len(overlap_rows)),
            "status": "warn" if len(overlap_rows) else "pass",
            "notes": "Potential overlapping field periods from the same pollster/sponsor/population grouping.",
        },
    ]
    return pd.DataFrame(rows)


def build_model_ablation(
    diagnostics: Dict[str, float],
    scenario_df: pd.DataFrame,
) -> pd.DataFrame:
    base_sigma = float(diagnostics["base_sigma_points"])
    poll_margin = float(diagnostics["weighted_poll_margin_points"])
    primary_margin = float(diagnostics["primary_transfer_prior_margin_points"])
    top_two_margin = float(diagnostics["top_two_primary_margin_points"])
    strength = float(diagnostics["candidate_strength_margin_adjustment"])
    money = float(diagnostics["money_media_margin_adjustment"])

    stages = [
        ("polling_only_component", poll_margin, "Polling baseline only."),
        ("plus_primary_transfer_prior", 0.78 * poll_margin + 0.22 * primary_margin, "Adds transfer prior blend."),
        (
            "plus_top_two_primary_structure",
            0.72 * poll_margin + 0.18 * primary_margin + 0.10 * top_two_margin,
            "Adds top-two primary structural prior.",
        ),
        (
            "plus_candidate_strength",
            0.72 * poll_margin + 0.18 * primary_margin + 0.10 * top_two_margin + strength,
            "Adds bounded candidate-strength adjustment.",
        ),
        (
            "plus_money_media",
            0.72 * poll_margin + 0.18 * primary_margin + 0.10 * top_two_margin + strength + money,
            "Adds bounded money/media adjustment.",
        ),
    ]

    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    rows = []
    prior_prob = None
    for stage, mean_margin, note in stages:
        sim = simulate_probability(mean_margin, base_sigma, draws=120_000)
        prob = float(sim["paxton_win_probability"])
        rows.append(
            {
                "ablation_stage": stage,
                "mean_paxton_margin_points": mean_margin,
                "paxton_win_probability": prob,
                "delta_vs_previous_probability": np.nan if prior_prob is None else prob - prior_prob,
                "delta_vs_full_model_probability": prob - float(full["paxton_win_probability"]),
                "notes": note,
            }
        )
        prior_prob = prob
    return pd.DataFrame(rows)


def build_calibration_summary(
    poll_diagnostics: pd.DataFrame,
    poll_miss: pd.DataFrame,
    sensitivity: pd.DataFrame,
    scenario_df: pd.DataFrame,
) -> pd.DataFrame:
    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    rows = []

    effective_count = float(
        poll_diagnostics.loc[poll_diagnostics["diagnostic"] == "effective_poll_count", "value"].iloc[0]
    )
    rows.append(
        {
            "metric": "effective_poll_count",
            "value": effective_count,
            "threshold": ">= 6.0",
            "pass": effective_count >= 6.0,
            "status": "ready" if effective_count >= 6.0 else "fail",
            "notes": "Minimum effective poll count for a publishable forecast.",
        }
    )

    interval_width = float(full["margin_80pct_high"] - full["margin_80pct_low"])
    rows.append(
        {
            "metric": "forecast_interval_width",
            "value": interval_width,
            "threshold": "<= 24.0",
            "pass": interval_width <= 24.0,
            "status": "ready" if interval_width <= 24.0 else "fail",
            "notes": "Wide intervals imply lower reliability and can block publication.",
        }
    )

    if not sensitivity.empty:
        full_prob = float(full["paxton_win_probability"])
        max_prob_shift = float((sensitivity["paxton_win_probability"] - full_prob).abs().max())
    else:
        max_prob_shift = np.nan
    rows.append(
        {
            "metric": "pollster_sensitivity_max_shift",
            "value": max_prob_shift,
            "threshold": "<= 0.10",
            "pass": bool(pd.notna(max_prob_shift) and max_prob_shift <= 0.10),
            "status": "ready" if pd.notna(max_prob_shift) and max_prob_shift <= 0.10 else "fail",
            "notes": "Probability should not swing too sharply when one pollster is removed.",
        }
    )

    if not poll_miss.empty:
        primary_abs_error = float(abs(poll_miss["two_candidate_margin_miss_points"].iloc[0]))
        primary_pass = primary_abs_error <= 6.0
        rows.append(
            {
                "metric": "primary_poll_margin_abs_error",
                "value": primary_abs_error,
                "threshold": "<= 6.0",
                "pass": primary_pass,
                "status": "ready" if primary_pass else "fail",
                "notes": "Historical analog error proxy from the March primary polling miss.",
            }
        )
    else:
        rows.append(
            {
                "metric": "primary_poll_margin_abs_error",
                "value": np.nan,
                "threshold": "<= 6.0",
                "pass": False,
                "status": "insufficient_history",
                "notes": "No usable pre-primary analog window was available.",
            }
        )

    for metric_name, note in [
        ("historical_brier_score", "Not enough comparable historical forecast runs are stored in-repo."),
        ("historical_log_loss", "Not enough comparable historical forecast runs are stored in-repo."),
        ("historical_calibration_error", "Not enough comparable historical forecast runs are stored in-repo."),
        ("historical_interval_coverage", "Not enough comparable historical forecast runs are stored in-repo."),
    ]:
        rows.append(
            {
                "metric": metric_name,
                "value": np.nan,
                "threshold": "historical benchmark required",
                "pass": False,
                "status": "insufficient_history",
                "notes": note,
            }
        )

    return pd.DataFrame(rows)


def build_source_manifest(raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name, frame in raw.items():
        if dataset_name == "source_registry":
            continue
        file_path = RAW_DIR / f"{dataset_name}.csv"
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        source_urls = frame["source_url"].dropna().astype(str).str.strip() if "source_url" in frame.columns else pd.Series(dtype=str)
        rows.append(
            {
                "dataset_name": dataset_name,
                "file_name": file_path.name,
                "row_count": int(len(frame)),
                "unique_source_url_count": int(source_urls.nunique()) if not source_urls.empty else 0,
                "file_sha256": file_hash,
                "latest_source_date": latest_date_value(
                    frame,
                    [
                        "release_date",
                        "end_date",
                        "date_end",
                        "date",
                        "signal_date",
                        "timestamp",
                        "election_date",
                        "report_date",
                    ],
                ),
            }
        )
    return pd.DataFrame(rows)


def build_data_quality_report(
    raw: Dict[str, pd.DataFrame],
    processed: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    groups = [
        ("polls", raw["polls"], ["release_date", "end_date"], "baseline", True),
        ("county_results", raw["county_primary_results"], ["election_date"], "projection_input", True),
        ("early_vote", processed["early_vote_turnout"], ["report_date"], "inactive_until_early_vote", AS_OF >= EARLY_VOTE_START),
        ("market_prices", processed["market_timeseries"], ["timestamp"], "external_price_check", True),
        ("finance_ads", raw["finance_ads"], ["date_end", "date_start"], "baseline_adjustment", True),
        ("endorsements_events", raw["endorsements_events"], ["date"], "qualitative_shock_tracking", True),
        ("turnout_signals", raw["turnout_signals"], ["signal_date"], "scenario_context", True),
        ("poll_miss", processed["poll_miss_diagnostics"], ["poll_window_end"], "stress_test_only", False),
    ]
    rows = []
    for name, frame, date_columns, model_use, required_for_publish in groups:
        latest = latest_date_value(frame, date_columns)
        latest_ts = pd.Timestamp(latest) if latest else pd.NaT
        days_stale = int((AS_OF - latest_ts).days) if pd.notna(latest_ts) else None
        note_column = "notes" if "notes" in frame.columns else "model_note" if "model_note" in frame.columns else None
        missing_notes = int(frame[note_column].fillna("").astype(str).str.strip().eq("").sum()) if note_column else len(frame)
        missing_sources = (
            int(frame["source_url"].fillna("").astype(str).str.strip().eq("").sum())
            if "source_url" in frame.columns
            else len(frame)
        )
        if name == "market_prices":
            stale_flag = bool(processed["market_timeseries"]["stale_price_flag"].all())
        elif name == "early_vote":
            stale_flag = bool(required_for_publish and not processed["early_vote_turnout"]["available_for_model"].any())
        else:
            stale_flag = bool(days_stale is not None and days_stale > 21)
        rows.append(
            {
                "signal_group": name,
                "row_count": int(len(frame)),
                "latest_source_date": latest,
                "days_stale": days_stale,
                "missing_source_count": missing_sources,
                "missing_notes_count": missing_notes,
                "stale_flag": stale_flag,
                "required_for_publish": bool(required_for_publish),
                "model_use_status": model_use,
            }
        )
    return pd.DataFrame(rows)


def build_source_registry_status(
    raw: Dict[str, pd.DataFrame],
    processed: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    registry = raw["source_registry"].copy()
    rows = []
    date_columns = {
        "polls": ["release_date", "end_date"],
        "primary_results": ["election_date"],
        "markets": ["timestamp"],
        "market_timeseries": ["timestamp"],
        "finance_ads": ["date_end", "date_start"],
        "endorsements_events": ["date"],
        "hunt_transfer": ["release_date", "end_date"],
        "subgroup_signals": ["release_date"],
        "candidate_strength": ["release_date"],
        "turnout_signals": ["signal_date"],
        "general_election_polls": ["release_date", "end_date"],
        "shock_scenarios": [],
        "county_primary_results": ["election_date"],
        "county_features": [],
        "early_vote_turnout": ["report_date"],
        "wager_settings": [],
    }
    for _, row in registry.iterrows():
        dataset_name = row["dataset_name"]
        frame = raw[dataset_name]
        latest = latest_date_value(frame, date_columns.get(dataset_name, []))
        latest_ts = pd.Timestamp(latest) if latest else pd.NaT
        days_stale = int((AS_OF - latest_ts).days) if pd.notna(latest_ts) else None
        freshness_sla_days = int(row["freshness_sla_days"])
        freshness_pass = bool(days_stale is None or days_stale <= freshness_sla_days)
        parse_status = "ok" if len(frame) > 0 else "empty"
        rows.append(
            {
                "dataset_name": dataset_name,
                "source_type": row["source_type"],
                "retrieval_method": row["retrieval_method"],
                "trust_tier": row["trust_tier"],
                "freshness_sla_days": freshness_sla_days,
                "latest_source_date": latest,
                "days_stale": days_stale,
                "freshness_pass": freshness_pass,
                "parse_status": parse_status,
                "row_count": int(len(frame)),
                "citation_url": row["citation_url"],
                "notes": row["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_release_status(
    scenario_df: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    data_quality_report: pd.DataFrame,
    county_model_status: pd.DataFrame,
    poll_canonicalization_report: pd.DataFrame,
    source_registry_status: pd.DataFrame,
) -> pd.DataFrame:
    blocking_issues = []

    required_quality = data_quality_report[data_quality_report["required_for_publish"]]
    for _, row in required_quality.iterrows():
        if bool(row["stale_flag"]) or int(row["missing_source_count"]) > 0 or int(row["missing_notes_count"]) > 0:
            blocking_issues.append(f"{row['signal_group']}_quality_gate_failed")

    calibration_failures = calibration_summary[~calibration_summary["pass"]]
    for _, row in calibration_failures.iterrows():
        blocking_issues.append(f"calibration:{row['metric']}:{row['status']}")

    county_row = county_model_status.iloc[0]
    if not bool(county_row["county_forecast_eligible"]):
        blocking_issues.append("county_coverage_incomplete")

    duplicate_fail = poll_canonicalization_report[poll_canonicalization_report["status"].eq("fail")]
    if not duplicate_fail.empty:
        blocking_issues.append("duplicate_core_poll_rows")

    registry_fails = source_registry_status[
        (~source_registry_status["freshness_pass"]) & source_registry_status["dataset_name"].ne("wager_settings")
    ]
    for _, row in registry_fails.iterrows():
        blocking_issues.append(f"source_registry_stale:{row['dataset_name']}")

    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    interval_width = float(full["margin_80pct_high"] - full["margin_80pct_low"])
    reliability_score = 100.0
    reliability_score -= 18.0 * len(set(blocking_issues))
    reliability_score -= min(20.0, max(0.0, interval_width - 20.0))
    reliability_score = max(0.0, min(100.0, reliability_score))
    if reliability_score >= 80:
        reliability_class = "high"
    elif reliability_score >= 60:
        reliability_class = "medium"
    else:
        reliability_class = "low"

    forecast_publishable = len(blocking_issues) == 0
    betting_eligible = False
    release_status = "published" if forecast_publishable else "withheld"
    rows = [
        {
            "as_of": str(AS_OF.date()),
            "run_id": f"txgop-{AS_OF.strftime('%Y%m%d')}",
            "release_status": release_status,
            "forecast_publishable": forecast_publishable,
            "betting_eligible": betting_eligible,
            "reliability_score": reliability_score,
            "reliability_class": reliability_class,
            "blocking_issue_count": len(set(blocking_issues)),
            "blocking_issues": ";".join(sorted(set(blocking_issues))) if blocking_issues else "",
            "county_forecast_eligible": bool(county_row["county_forecast_eligible"]),
            "county_detail_available": bool(county_row["county_detail_available"]),
            "candidate_vote_coverage_pct": float(county_row["candidate_vote_coverage_pct"]),
        }
    ]
    return pd.DataFrame(rows)


def binary_share_kelly_fraction(fair_probability: float, market_price: float) -> float:
    if not 0 < market_price < 1:
        return 0.0
    return float((fair_probability - market_price) / (1.0 - market_price))


def build_wager_value_table(
    market_timeseries: pd.DataFrame,
    scenario_df: pd.DataFrame,
    wager_settings: pd.DataFrame,
    release_status: pd.DataFrame,
    sensitivity: pd.DataFrame,
    shock_model: pd.DataFrame,
) -> pd.DataFrame:
    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    scenario_probs = {
        row["scenario"]: float(row["paxton_win_probability"]) for _, row in scenario_df.iterrows()
    }
    paxton_fair = float(full["paxton_win_probability"])
    cornyn_fair = float(full["cornyn_win_probability"])
    settings = wager_settings.iloc[0].copy()
    bankroll = float(settings["bankroll"])
    max_exposure_pct = float(settings["max_exposure_pct"])
    fractional_kelly = float(settings["fractional_kelly"])
    base_required_edge = (
        float(settings["edge_buffer_pct"])
        + float(settings["fee_buffer_pct"])
        + float(settings["spread_buffer_pct"])
    )

    model_risk_buffer = 0.03
    scenario_robustness_buffer = 0.02
    slippage_buffer = 0.01
    required_edge = base_required_edge + model_risk_buffer + scenario_robustness_buffer + slippage_buffer
    release_row = release_status.iloc[0]

    if not sensitivity.empty:
        sensitivity_shift = float((sensitivity["paxton_win_probability"] - paxton_fair).abs().max())
    else:
        sensitivity_shift = np.nan

    adverse_paxton_prob = min(
        scenario_probs.get("polling_only", paxton_fair),
        scenario_probs.get("full_model_mid_turnout", paxton_fair),
        scenario_probs.get("low_turnout", paxton_fair),
        scenario_probs.get("high_turnout", paxton_fair),
        scenario_probs.get("repeat_primary_poll_error", paxton_fair),
        float(shock_model["paxton_win_probability"].min()) if not shock_model.empty else paxton_fair,
    )
    adverse_cornyn_prob = 1.0 - max(
        scenario_probs.get("polling_only", paxton_fair),
        scenario_probs.get("full_model_mid_turnout", paxton_fair),
        scenario_probs.get("low_turnout", paxton_fair),
        scenario_probs.get("high_turnout", paxton_fair),
        scenario_probs.get("repeat_primary_poll_error", paxton_fair),
        float(shock_model["paxton_win_probability"].max()) if not shock_model.empty else paxton_fair,
    )

    latest = (
        market_timeseries.sort_values("timestamp")
        .groupby(["platform", "contract"], as_index=False)
        .tail(1)
        .copy()
    )
    rows = []
    for _, market in latest.iterrows():
        paxton_buy_price = market["paxton_ask"]
        if pd.isna(paxton_buy_price):
            paxton_buy_price = market["paxton_last"]
        if pd.isna(paxton_buy_price):
            paxton_buy_price = market["paxton_market_prob"]

        cornyn_buy_price = market["cornyn_ask"]
        if pd.isna(cornyn_buy_price):
            cornyn_buy_price = market["cornyn_last"]
        if pd.isna(cornyn_buy_price):
            cornyn_buy_price = market["cornyn_market_prob"]

        paxton_edge = paxton_fair - float(paxton_buy_price)
        cornyn_edge = cornyn_fair - float(cornyn_buy_price)
        paxton_robust_edge = adverse_paxton_prob - float(paxton_buy_price)
        cornyn_robust_edge = adverse_cornyn_prob - float(cornyn_buy_price)
        paxton_kelly = max(0.0, binary_share_kelly_fraction(paxton_fair, float(paxton_buy_price)))
        cornyn_kelly = max(0.0, binary_share_kelly_fraction(cornyn_fair, float(cornyn_buy_price)))
        paxton_capped = min(max_exposure_pct, paxton_kelly * fractional_kelly)
        cornyn_capped = min(max_exposure_pct, cornyn_kelly * fractional_kelly)

        block_reasons = []
        if not bool(release_row["forecast_publishable"]):
            block_reasons.append("forecast_withheld")
        if bool(market["stale_price_flag"]):
            block_reasons.append("stale_market_price")
        if str(market["liquidity_warning"]) != "ok":
            block_reasons.append(str(market["liquidity_warning"]))
        if str(market["settlement_warning"]) != "ok":
            block_reasons.append(str(market["settlement_warning"]))
        if pd.notna(sensitivity_shift) and sensitivity_shift > 0.10:
            block_reasons.append("single_pollster_sensitivity_too_high")

        value_flag = "no_bet_zone"
        research_side = "None"
        capped_fraction = 0.0

        if (
            not block_reasons
            and paxton_edge > required_edge
            and paxton_robust_edge > 0
            and paxton_edge > cornyn_edge
        ):
            value_flag = "paxton_value"
            research_side = "Paxton"
            capped_fraction = paxton_capped
        elif (
            not block_reasons
            and cornyn_edge > required_edge
            and cornyn_robust_edge > 0
            and cornyn_edge > paxton_edge
        ):
            value_flag = "cornyn_value"
            research_side = "Cornyn"
            capped_fraction = cornyn_capped
        else:
            if paxton_edge <= required_edge and cornyn_edge <= required_edge:
                block_reasons.append("edge_below_required_buffer")
            if paxton_robust_edge <= 0 and cornyn_robust_edge <= 0:
                block_reasons.append("edge_not_durable_across_scenarios")

        rows.append(
            {
                "platform": market["platform"],
                "contract": market["contract"],
                "timestamp": market["timestamp"],
                "paxton_model_probability": paxton_fair,
                "cornyn_model_probability": cornyn_fair,
                "paxton_buy_price": float(paxton_buy_price),
                "cornyn_buy_price": float(cornyn_buy_price),
                "average_spread": float(market["average_spread"]) if pd.notna(market["average_spread"]) else np.nan,
                "stale_price_flag": bool(market["stale_price_flag"]),
                "liquidity_warning": market["liquidity_warning"],
                "settlement_warning": market["settlement_warning"],
                "paxton_edge": paxton_edge,
                "cornyn_edge": cornyn_edge,
                "paxton_robust_edge": paxton_robust_edge,
                "cornyn_robust_edge": cornyn_robust_edge,
                "required_edge": required_edge,
                "value_flag": value_flag,
                "research_side": research_side,
                "decision_eligible": value_flag != "no_bet_zone",
                "block_reasons": ";".join(sorted(set(block_reasons))) if block_reasons else "",
                "fractional_kelly": fractional_kelly,
                "capped_exposure_fraction": capped_fraction,
                "illustrative_stake": bankroll * capped_fraction,
                "max_loss": bankroll * capped_fraction,
                "research_note": "Research output only; default to no-bet unless forecast publication and market-safety gates both pass.",
            }
        )
    return pd.DataFrame(rows)


def add_last_updated(frame: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    out = frame.copy()
    out["last_updated"] = str(as_of.date())
    return out


def snapshot_raw_inputs(run_id: str) -> pd.DataFrame:
    destination = SNAPSHOT_DIR / run_id / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        target = destination / path.name
        shutil.copy2(path, target)
        rows.append(
            {
                "run_id": run_id,
                "file_name": path.name,
                "snapshot_path": str(target.relative_to(ROOT)),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def build_forecast_package(
    scenario_df: pd.DataFrame,
    diagnostics: Dict[str, float],
    release_status: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    data_quality_report: pd.DataFrame,
    source_registry_status: pd.DataFrame,
    wager_value_table: pd.DataFrame,
    model_ablation: pd.DataFrame,
) -> Dict[str, object]:
    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    release = release_status.iloc[0]
    interval_width = float(full["margin_80pct_high"] - full["margin_80pct_low"])
    eligible_rows = wager_value_table[wager_value_table["decision_eligible"]]
    blocking_issues = str(release["blocking_issues"]).split(";") if str(release["blocking_issues"]).strip() else []
    blocking_issues = [issue for issue in blocking_issues if issue]

    calibration_ready = calibration_summary["pass"].all()
    source_freshness = source_registry_status[
        ["dataset_name", "freshness_pass", "days_stale", "latest_source_date", "trust_tier"]
    ].to_dict("records")
    scenario_summary = []
    for _, row in scenario_df.iterrows():
        scenario_summary.append(
            {
                "scenario": row["scenario"],
                "paxton_win_probability": float(row["paxton_win_probability"]),
                "cornyn_win_probability": float(row["cornyn_win_probability"]),
                "mean_paxton_margin_points": float(row["mean_paxton_margin_points"]),
            }
        )

    return {
        "version": FORECAST_PACKAGE_VERSION,
        "as_of": diagnostics["as_of"],
        "run_id": str(release["run_id"]),
        "release_status": {
            "state": str(release["release_status"]),
            "forecast_publishable": bool(release["forecast_publishable"]),
            "betting_eligible": bool(release["betting_eligible"]),
            "blocking_issues": blocking_issues,
        },
        "headline_forecast": {
            "paxton_fair_probability": float(full["paxton_win_probability"]),
            "cornyn_fair_probability": float(full["cornyn_win_probability"]),
            "mean_paxton_margin_points": float(full["mean_paxton_margin_points"]),
            "forecast_withheld": not bool(release["forecast_publishable"]),
        },
        "uncertainty": {
            "margin_80pct_interval": [float(full["margin_80pct_low"]), float(full["margin_80pct_high"])],
            "sigma_margin_points": float(full["sigma_margin_points"]),
            "interval_width_points": interval_width,
            "reliability_class": str(release["reliability_class"]),
        },
        "model_health": {
            "reliability_score": float(release["reliability_score"]),
            "reliability_class": str(release["reliability_class"]),
            "county_forecast_eligible": bool(release["county_forecast_eligible"]),
            "candidate_vote_coverage_pct": float(release["candidate_vote_coverage_pct"]),
            "calibration_ready": bool(calibration_ready),
        },
        "calibration_metrics": calibration_summary.to_dict("records"),
        "source_freshness": source_freshness,
        "blocking_issues": blocking_issues,
        "scenario_summary": scenario_summary,
        "betting_decision_summary": {
            "eligible_market_count": int(len(eligible_rows)),
            "no_bet_default": True,
            "recommended_markets": eligible_rows[["platform", "contract", "research_side"]].to_dict("records"),
        },
        "headline": {
            "as_of": diagnostics["as_of"],
            "paxton_fair_probability": float(full["paxton_win_probability"]),
            "cornyn_fair_probability": float(full["cornyn_win_probability"]),
            "mean_paxton_margin_points": float(full["mean_paxton_margin_points"]),
            "margin_80pct_interval": [float(full["margin_80pct_low"]), float(full["margin_80pct_high"])],
        },
        "diagnostics": diagnostics,
        "data_quality_summary": data_quality_report[
            ["signal_group", "stale_flag", "required_for_publish", "days_stale"]
        ].to_dict("records"),
        "model_ablation_summary": model_ablation.to_dict("records"),
    }


def build_audit_report(
    forecast_package: Dict[str, object],
    release_status: pd.DataFrame,
    data_quality_report: pd.DataFrame,
    wager_value_table: pd.DataFrame,
) -> str:
    release = release_status.iloc[0]
    headline = forecast_package["headline_forecast"]
    uncertainty = forecast_package["uncertainty"]
    blocked = forecast_package["blocking_issues"]
    blocked_text = "\n".join(f"- {issue}" for issue in blocked) if blocked else "- none"
    required_quality = data_quality_report[data_quality_report["required_for_publish"]]
    quality_text = "\n".join(
        f"- {row.signal_group}: stale={row.stale_flag}, missing_sources={row.missing_source_count}, missing_notes={row.missing_notes_count}"
        for row in required_quality.itertuples()
    )
    wager_text = "\n".join(
        f"- {row.platform}: {row.value_flag} ({row.block_reasons or 'no block reasons'})"
        for row in wager_value_table.itertuples()
    )
    return textwrap.dedent(
        f"""
        # Forecast Audit Report

        - Run ID: `{release['run_id']}`
        - As of: `{forecast_package['as_of']}`
        - Release status: `{release['release_status']}`
        - Forecast publishable: `{bool(release['forecast_publishable'])}`
        - Betting eligible: `{bool(release['betting_eligible'])}`
        - Reliability: `{release['reliability_class']}` ({float(release['reliability_score']):.1f}/100)

        ## Headline Forecast

        - Paxton fair probability: `{headline['paxton_fair_probability']:.3f}`
        - Cornyn fair probability: `{headline['cornyn_fair_probability']:.3f}`
        - Mean Paxton margin: `{headline['mean_paxton_margin_points']:.2f}`
        - 80% margin interval: `{uncertainty['margin_80pct_interval'][0]:.2f}` to `{uncertainty['margin_80pct_interval'][1]:.2f}`
        - Forecast withheld: `{headline['forecast_withheld']}`

        ## Blocking Issues

        {blocked_text}

        ## Required Data Quality Gates

        {quality_text}

        ## Betting Decision Summary

        {wager_text}
        """
    ).strip() + "\n"


def write_outputs(
    raw: Dict[str, pd.DataFrame],
    frames: Dict[str, pd.DataFrame],
    scenario_df: pd.DataFrame,
    forecast_package: Dict[str, object],
    audit_report: str,
    run_id: str,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        add_last_updated(frame).to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    add_last_updated(scenario_df).to_csv(PROCESSED_DIR / "model_scenarios.csv", index=False)
    (PROCESSED_DIR / "model_output.json").write_text(json.dumps(forecast_package, indent=2), encoding="utf-8")
    (PROCESSED_DIR / "audit_report.md").write_text(audit_report, encoding="utf-8")
    snapshot_manifest = snapshot_raw_inputs(run_id)
    add_last_updated(snapshot_manifest).to_csv(PROCESSED_DIR / "snapshot_manifest.csv", index=False)


def main() -> Dict[str, object]:
    raw = read_raw()
    validate_source_registry(raw)
    validate_source_metadata(raw)
    validate_primary_results(raw["primary_results"])
    validate_polls(raw["polls"])
    validate_hunt_transfer(raw["hunt_transfer"])
    validate_county_primary_results(raw["county_primary_results"], raw["primary_results"])

    polls = process_polls(raw["polls"])
    hunt_transfer = process_hunt_transfer(raw["hunt_transfer"])
    primary = raw["primary_results"].copy()
    markets = raw["markets"].copy()
    market_timeseries = process_market_timeseries(raw["market_timeseries"])
    finance_ads = raw["finance_ads"].copy()
    endorsements_events = raw["endorsements_events"].copy()
    subgroup_signals = raw["subgroup_signals"].copy()
    candidate_strength = raw["candidate_strength"].copy()
    turnout_signals = raw["turnout_signals"].copy()
    general_election_polls = raw["general_election_polls"].copy()
    wager_settings = raw["wager_settings"].copy()
    shock_scenarios = raw["shock_scenarios"].copy()
    county_features = raw["county_features"].copy()
    county_primary_results = raw["county_primary_results"].copy()
    early_vote_turnout = process_early_vote_turnout(raw["early_vote_turnout"])

    scenario_df, diagnostics = build_scenarios(polls, primary, finance_ads, hunt_transfer, candidate_strength)
    sensitivity = build_sensitivity(polls, primary, finance_ads, hunt_transfer, candidate_strength)
    market_comparison = build_market_comparison(markets, scenario_df)
    poll_diagnostics = build_poll_diagnostics(polls)
    poll_miss_diagnostics = build_poll_miss_diagnostics(polls, primary)
    margin_distribution = build_margin_distribution(scenario_df)
    shock_model = build_shock_model(scenario_df, shock_scenarios)
    county_turnout_model, county_model_status = process_county_turnout_model(
        county_primary_results,
        county_features,
        hunt_transfer,
    )
    runoff_county_projection = build_runoff_county_projection(county_turnout_model, early_vote_turnout)
    poll_canonicalization_report = build_poll_canonicalization_report(polls)
    model_ablation = build_model_ablation(diagnostics, scenario_df)
    calibration_summary = build_calibration_summary(
        poll_diagnostics,
        poll_miss_diagnostics,
        sensitivity,
        scenario_df,
    )

    processed_for_quality = {
        "early_vote_turnout": early_vote_turnout,
        "market_timeseries": market_timeseries,
        "poll_miss_diagnostics": poll_miss_diagnostics,
    }
    data_quality_report = build_data_quality_report(raw, processed_for_quality)
    source_registry_status = build_source_registry_status(raw, processed_for_quality)
    release_status = build_release_status(
        scenario_df,
        calibration_summary,
        data_quality_report,
        county_model_status,
        poll_canonicalization_report,
        source_registry_status,
    )
    wager_value_table = build_wager_value_table(
        market_timeseries,
        scenario_df,
        wager_settings,
        release_status,
        sensitivity,
        shock_model,
    )

    diagnostics["poll_effective_count"] = float(
        poll_diagnostics.loc[poll_diagnostics["diagnostic"] == "effective_poll_count", "value"].iloc[0]
    )
    diagnostics["poll_weighted_standard_error"] = float(
        poll_diagnostics.loc[poll_diagnostics["diagnostic"] == "weighted_standard_error", "value"].iloc[0]
    )
    diagnostics["county_detail_available"] = bool(county_model_status["county_detail_available"].iloc[0])
    diagnostics["release_status"] = str(release_status["release_status"].iloc[0])
    diagnostics["reliability_score"] = float(release_status["reliability_score"].iloc[0])

    processed = {
        "polls": polls,
        "hunt_transfer": hunt_transfer,
        "primary_results": primary,
        "markets": markets,
        "market_timeseries": market_timeseries,
        "wager_settings": wager_settings,
        "finance_ads": finance_ads,
        "endorsements_events": endorsements_events,
        "subgroup_signals": subgroup_signals,
        "candidate_strength": candidate_strength,
        "turnout_signals": turnout_signals,
        "general_election_polls": general_election_polls,
        "shock_scenarios": shock_scenarios,
        "county_primary_results": county_primary_results,
        "county_features": county_features,
        "early_vote_turnout": early_vote_turnout,
        "sensitivity": sensitivity,
        "market_comparison": market_comparison,
        "poll_diagnostics": poll_diagnostics,
        "poll_miss_diagnostics": poll_miss_diagnostics,
        "margin_distribution": margin_distribution,
        "shock_model": shock_model,
        "county_turnout_model": county_turnout_model,
        "county_model_status": county_model_status,
        "runoff_county_projection": runoff_county_projection,
        "wager_value_table": wager_value_table,
        "data_quality_report": data_quality_report,
        "source_manifest": build_source_manifest(raw),
        "source_registry_status": source_registry_status,
        "poll_canonicalization_report": poll_canonicalization_report,
        "model_ablation": model_ablation,
        "calibration_summary": calibration_summary,
        "release_status": release_status,
    }

    run_id = str(release_status["run_id"].iloc[0])
    forecast_package = build_forecast_package(
        scenario_df,
        diagnostics,
        release_status,
        calibration_summary,
        data_quality_report,
        source_registry_status,
        wager_value_table,
        model_ablation,
    )
    audit_report = build_audit_report(
        forecast_package,
        release_status,
        data_quality_report,
        wager_value_table,
    )
    write_outputs(raw, processed, scenario_df, forecast_package, audit_report, run_id)

    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    print(
        "Full model: Paxton "
        f"{full['paxton_win_probability']:.1%}, Cornyn {full['cornyn_win_probability']:.1%}, "
        f"mean Paxton margin {full['mean_paxton_margin_points']:.1f} pts, "
        f"release {release_status['release_status'].iloc[0]}"
    )
    print(f"Wrote processed outputs to {PROCESSED_DIR}")
    return {
        "scenarios": scenario_df,
        "diagnostics": diagnostics,
        "wager_value_table": wager_value_table,
        "poll_miss_diagnostics": poll_miss_diagnostics,
        "data_quality_report": data_quality_report,
        "release_status": release_status,
        "forecast_package": forecast_package,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Texas GOP Senate runoff analysis dataset and model.")
    parser.parse_args()
    main()
