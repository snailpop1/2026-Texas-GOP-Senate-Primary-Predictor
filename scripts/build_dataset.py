from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
AS_OF = pd.Timestamp("2026-05-03")
RNG_SEED = 20260503


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
    }


def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


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
    usable = hunt_transfer[hunt_transfer["usable_in_model"].astype(str).str.lower().eq("true")]
    missing = usable[required].isna().any(axis=1)
    if missing.any():
        bad = usable.loc[missing, ["pollster", "release_date"]].to_dict("records")
        raise AssertionError(f"Usable Hunt-transfer rows have missing required fields: {bad}")
    if len(usable) < 4:
        raise AssertionError("Expected at least four usable Hunt-transfer data points.")


def process_polls(polls: pd.DataFrame, as_of: pd.Timestamp = AS_OF) -> pd.DataFrame:
    out = polls.copy()
    for column in ["start_date", "end_date", "release_date"]:
        out[column] = pd.to_datetime(out[column])

    out["poll_midpoint"] = out["start_date"] + (out["end_date"] - out["start_date"]) / 2
    out["days_old"] = (as_of - out["poll_midpoint"]).dt.days.clip(lower=0)
    out["two_candidate_paxton_share"] = out["paxton_pct"] / (out["paxton_pct"] + out["cornyn_pct"])
    out["two_candidate_margin"] = 200 * out["two_candidate_paxton_share"] - 100

    recency_half_life_days = 21.0
    out["recency_weight"] = np.exp(-np.log(2) * out["days_old"] / recency_half_life_days)
    out["sample_weight"] = np.sqrt(out["sample_size"] / 600.0)
    out["population_weight"] = np.where(out["population"].str.upper().eq("LV"), 1.0, 0.75)
    out["partisan_weight"] = np.where(out["partisan_or_internal"].astype(str).str.lower().eq("true"), 0.70, 1.0)
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
    out["partisan_weight"] = np.where(out["partisan_or_internal"].astype(str).str.lower().eq("true"), 0.70, 1.0)
    out["model_weight"] = (
        out["recency_weight"] * out["sample_weight"] * out["population_weight"] * out["partisan_weight"]
    )
    out.loc[~out["usable_in_model"].astype(str).str.lower().eq("true"), "model_weight"] = 0.0
    out["explicit_two_candidate_paxton_share"] = out["paxton_pct"] / (out["paxton_pct"] + out["cornyn_pct"])
    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.astype(float), weights=weights.astype(float)))


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    average = weighted_mean(values, weights)
    variance = np.average((values.astype(float) - average) ** 2, weights=weights.astype(float))
    return float(math.sqrt(max(variance, 0.0)))


def aggregate_hunt_transfer(hunt_transfer: pd.DataFrame) -> Dict[str, float]:
    usable = hunt_transfer[hunt_transfer["model_weight"] > 0].copy()
    paxton_pct = weighted_mean(usable["paxton_pct"], usable["model_weight"])
    cornyn_pct = weighted_mean(usable["cornyn_pct"], usable["model_weight"])
    undecided_pct = weighted_mean(usable["undecided_other_pct"], usable["model_weight"])
    explicit_two_way = weighted_mean(usable["explicit_two_candidate_paxton_share"], usable["model_weight"])

    # Unresolved Hunt voters are modeled as slightly Paxton-leaning because every
    # direct data point and the qualitative alignment evidence points that way.
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

    # Split minor-candidate voters evenly; there is no reliable transfer polling for
    # these much smaller candidate groups.
    minor = statewide.loc[
        ~statewide["candidate"].isin(["Ken Paxton", "John Cornyn", "Wesley Hunt"]), "votes"
    ].sum()
    paxton_runoff = paxton + transfer["hunt_final_paxton_transfer_share"] * hunt + 0.50 * minor
    cornyn_runoff = cornyn + transfer["hunt_final_cornyn_transfer_share"] * hunt + 0.50 * minor
    paxton_share = paxton_runoff / (paxton_runoff + cornyn_runoff)
    return {
        "paxton_share": float(paxton_share),
        "two_candidate_margin": float(200 * paxton_share - 100),
        "top_two_primary_margin": float(
            200 * (paxton / (paxton + cornyn)) - 100
        ),
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

    # Money matters in a statewide runoff, but the primary showed money cannot fully
    # erase ideological fit. Cap the Cornyn adjustment at 1.5 margin points.
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

    base_sigma = max(6.5, poll_dispersion + 3.0)
    polling_plus_primary_margin = 0.78 * poll_margin + 0.22 * prior["two_candidate_margin"]

    full_margin = (
        0.72 * poll_margin
        + 0.18 * prior["two_candidate_margin"]
        + 0.10 * prior["top_two_primary_margin"]
        + strength["candidate_strength_margin_adjustment"]
        + money["money_media_margin_adjustment"]
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
        **{key: value for key, value in prior.items() if key.startswith("hunt_")},
        **strength,
        **money,
    }
    return scenario_df, diagnostics


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

    non_partisan = polls[~((polls["model_group"] == "core") & polls["partisan_or_internal"].astype(str).str.lower().eq("true"))]
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


def write_outputs(frames: Dict[str, pd.DataFrame], scenario_df: pd.DataFrame, diagnostics: Dict[str, float]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    scenario_df.to_csv(PROCESSED_DIR / "model_scenarios.csv", index=False)

    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0].to_dict()
    output = {
        "headline": {
            "as_of": diagnostics["as_of"],
            "paxton_fair_probability": full["paxton_win_probability"],
            "cornyn_fair_probability": full["cornyn_win_probability"],
            "mean_paxton_margin_points": full["mean_paxton_margin_points"],
            "margin_80pct_interval": [full["margin_80pct_low"], full["margin_80pct_high"]],
        },
        "diagnostics": diagnostics,
    }
    (PROCESSED_DIR / "model_output.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


def main() -> Dict[str, object]:
    raw = read_raw()
    validate_primary_results(raw["primary_results"])
    validate_polls(raw["polls"])
    validate_hunt_transfer(raw["hunt_transfer"])

    polls = process_polls(raw["polls"])
    hunt_transfer = process_hunt_transfer(raw["hunt_transfer"])
    primary = raw["primary_results"].copy()
    markets = raw["markets"].copy()
    finance_ads = raw["finance_ads"].copy()
    endorsements_events = raw["endorsements_events"].copy()
    subgroup_signals = raw["subgroup_signals"].copy()
    candidate_strength = raw["candidate_strength"].copy()
    turnout_signals = raw["turnout_signals"].copy()
    general_election_polls = raw["general_election_polls"].copy()

    scenario_df, diagnostics = build_scenarios(polls, primary, finance_ads, hunt_transfer, candidate_strength)
    sensitivity = build_sensitivity(polls, primary, finance_ads, hunt_transfer, candidate_strength)
    market_comparison = build_market_comparison(markets, scenario_df)

    processed = {
        "polls": polls,
        "hunt_transfer": hunt_transfer,
        "primary_results": primary,
        "markets": markets,
        "finance_ads": finance_ads,
        "endorsements_events": endorsements_events,
        "subgroup_signals": subgroup_signals,
        "candidate_strength": candidate_strength,
        "turnout_signals": turnout_signals,
        "general_election_polls": general_election_polls,
        "sensitivity": sensitivity,
        "market_comparison": market_comparison,
    }
    write_outputs(processed, scenario_df, diagnostics)

    full = scenario_df[scenario_df["scenario"] == "full_model_mid_turnout"].iloc[0]
    print(
        "Full model: Paxton "
        f"{full['paxton_win_probability']:.1%}, Cornyn {full['cornyn_win_probability']:.1%}, "
        f"mean Paxton margin {full['mean_paxton_margin_points']:.1f} pts"
    )
    print(f"Wrote processed outputs to {PROCESSED_DIR}")
    return {"scenarios": scenario_df, "diagnostics": diagnostics}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Texas GOP Senate runoff analysis dataset and model.")
    parser.parse_args()
    main()
