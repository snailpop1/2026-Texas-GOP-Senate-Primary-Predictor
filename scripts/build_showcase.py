from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SHOWCASE_DIR = ROOT / "showcase"


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def pts(value: float, digits: int = 1) -> str:
    return f"{float(value):+.{digits}f}"


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DIR / name)


def status_label(value: object) -> str:
    return "Pass" if bool(value) else "Watch"


def scenario_cards(scenarios: pd.DataFrame) -> str:
    rows = []
    for row in scenarios.itertuples():
        prob = float(row.paxton_win_probability)
        rows.append(
            f"""
            <article class="scenario">
              <div>
                <h3>{esc(str(row.scenario).replace('_', ' ').title())}</h3>
                <p>{esc(row.note)}</p>
              </div>
              <div class="scenario-metric">
                <strong>{pct(prob)}</strong>
                <span>Paxton win probability</span>
              </div>
              <div class="mini-bar" aria-hidden="true"><span style="width:{prob * 100:.2f}%"></span></div>
              <footer>{pts(row.mean_paxton_margin_points)} pt mean margin</footer>
            </article>
            """
        )
    return "\n".join(rows)


def sensitivity_rows(sensitivity: pd.DataFrame, full_prob: float) -> str:
    rows = []
    ordered = sensitivity.copy()
    ordered["shift"] = (ordered["paxton_win_probability"] - full_prob).abs()
    ordered = ordered.sort_values("shift", ascending=False).head(8)
    for row in ordered.itertuples():
        prob = float(row.paxton_win_probability)
        rows.append(
            f"""
            <tr>
              <td>{esc(row.removed_pollster)}</td>
              <td>{pct(prob)}</td>
              <td>{pts(row.mean_paxton_margin_points)} pts</td>
              <td><div class="table-bar"><span style="width:{prob * 100:.2f}%"></span></div></td>
            </tr>
            """
        )
    return "\n".join(rows)


def ablation_rows(ablation: pd.DataFrame) -> str:
    rows = []
    for row in ablation.itertuples():
        prob = float(row.paxton_win_probability)
        rows.append(
            f"""
            <tr>
              <td>{esc(str(row.ablation_stage).replace('_', ' ').title())}</td>
              <td>{pts(row.mean_paxton_margin_points)} pts</td>
              <td>{pct(prob)}</td>
              <td>{esc(row.notes)}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def source_rows(source_status: pd.DataFrame) -> str:
    rows = []
    priority = source_status.copy()
    priority["freshness_rank"] = priority["freshness_pass"].map(lambda value: 0 if bool(value) else 1)
    priority = priority.sort_values(["freshness_rank", "trust_tier", "dataset_name"]).head(12)
    for row in priority.itertuples():
        rows.append(
            f"""
            <tr>
              <td>{esc(row.dataset_name)}</td>
              <td><span class="pill {'ok' if bool(row.freshness_pass) else 'warn'}">{status_label(row.freshness_pass)}</span></td>
              <td>{esc(getattr(row, 'freshness_basis', ''))}</td>
              <td>{esc(getattr(row, 'latest_source_date', ''))}</td>
              <td>{esc(getattr(row, 'latest_source_check_date', ''))}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def market_rows(wager: pd.DataFrame) -> str:
    rows = []
    for row in wager.itertuples():
        rows.append(
            f"""
            <tr>
              <td>{esc(row.platform)}</td>
              <td>{esc(row.value_flag)}</td>
              <td>{pct(row.paxton_buy_price)}</td>
              <td>{pct(row.paxton_model_probability)}</td>
              <td>{esc(row.block_reasons or 'none')}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_html() -> str:
    package = json.loads((PROCESSED_DIR / "model_output.json").read_text(encoding="utf-8"))
    scenarios = read_csv("model_scenarios.csv")
    release = read_csv("release_status.csv").iloc[0]
    quality = read_csv("data_quality_report.csv")
    source_status = read_csv("source_registry_status.csv")
    sensitivity = read_csv("sensitivity.csv")
    ablation = read_csv("model_ablation.csv")
    wager = read_csv("wager_value_table.csv")
    poll_diag = read_csv("poll_diagnostics.csv")
    turnout = read_csv("turnout_prior.csv")
    hunt = read_csv("hunt_transfer_scenarios.csv")

    headline = package["headline_forecast"]
    uncertainty = package["uncertainty"]
    full_prob = float(headline["paxton_fair_probability"])
    cornyn_prob = float(headline["cornyn_fair_probability"])
    interval_low, interval_high = uncertainty["margin_80pct_interval"]
    publishable = bool(package["release_status"]["forecast_publishable"])
    required_quality = quality[quality["required_for_publish"]]
    source_pass_count = int(source_status["freshness_pass"].astype(bool).sum())
    source_count = int(len(source_status))
    effective_poll_count = poll_diag.loc[
        poll_diag["diagnostic"].eq("effective_poll_count"), "value"
    ].iloc[0]
    mid_turnout = turnout.loc[turnout["scenario"].eq("mid_turnout"), "projected_runoff_votes"].iloc[0]
    baseline_hunt = hunt.loc[hunt["scenario"].eq("baseline")].iloc[0]
    status_class = "ok" if publishable else "warn"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Texas GOP Senate Runoff Forecast</title>
  <style>
    :root {{
      --ink: #13212f;
      --muted: #647080;
      --line: #d9e0e7;
      --paper: #f7f9fb;
      --panel: #ffffff;
      --navy: #153756;
      --blue: #2e6f95;
      --red: #b83a36;
      --gold: #b8892f;
      --green: #287a5b;
      --warn: #9f5d12;
      --shadow: 0 18px 50px rgba(22, 33, 47, 0.09);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font: 15px/1.55 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow-x: hidden;
    }}
    header {{
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    .wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 24px; }}
    .topbar {{ height: 64px; display: flex; align-items: center; justify-content: space-between; gap: 18px; }}
    .brand {{ display: flex; align-items: center; gap: 12px; font-weight: 800; letter-spacing: 0.01em; }}
    .mark {{
      width: 34px; height: 34px; border-radius: 8px; color: white; background: var(--navy);
      display: grid; place-items: center; font-weight: 900;
    }}
    nav {{ display: flex; gap: 18px; color: var(--muted); font-size: 13px; }}
    main {{ padding: 34px 0 60px; }}
    .hero {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.8fr);
      gap: 22px;
      align-items: stretch;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }}
    .hero-main {{ padding: 30px; }}
    .kicker {{ color: var(--muted); font-size: 13px; margin: 0 0 10px; }}
    h1 {{ margin: 0; font-size: clamp(34px, 5vw, 58px); line-height: 0.98; letter-spacing: 0; max-width: 780px; }}
    h2 {{ margin: 0 0 16px; font-size: 22px; letter-spacing: 0; }}
    h3 {{ margin: 0; font-size: 14px; letter-spacing: 0; }}
    .headline-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-top: 28px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 112px;
      background: #fbfcfd;
    }}
    .metric span, .scenario span, .small-label {{ color: var(--muted); font-size: 12px; display: block; }}
    .metric strong {{ display: block; font-size: 30px; line-height: 1.1; margin-top: 8px; }}
    .red {{ color: var(--red); }}
    .blue {{ color: var(--blue); }}
    .gold {{ color: var(--gold); }}
    .probability {{ margin-top: 24px; }}
    .probability-labels {{ display: flex; justify-content: space-between; font-weight: 750; font-size: 13px; }}
    .probability-track {{
      height: 18px;
      border-radius: 999px;
      background: linear-gradient(90deg, var(--red) 0 {full_prob * 100:.2f}%, var(--blue) {full_prob * 100:.2f}% 100%);
      margin: 8px 0;
      border: 1px solid rgba(19, 33, 47, 0.16);
    }}
    .health {{ padding: 22px; display: flex; flex-direction: column; gap: 13px; }}
    .health-row {{ display: flex; justify-content: space-between; align-items: center; gap: 14px; border-bottom: 1px solid var(--line); padding-bottom: 12px; }}
    .health-row:last-child {{ border-bottom: 0; padding-bottom: 0; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }}
    .pill.ok {{ color: var(--green); background: #e8f5ee; }}
    .pill.warn {{ color: var(--warn); background: #fff3de; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; margin-top: 22px; }}
    .section {{ margin-top: 22px; padding: 24px; }}
    .section-title {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 16px; }}
    .section-title p {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .scenarios {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .scenario {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 15px;
      background: #fbfcfd;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .scenario p {{ margin: 6px 0 0; color: var(--muted); font-size: 12px; }}
    .scenario-metric strong {{ font-size: 22px; }}
    .mini-bar, .table-bar {{ height: 8px; border-radius: 999px; background: #e5ebf0; overflow: hidden; }}
    .mini-bar span, .table-bar span {{ display: block; height: 100%; background: var(--red); }}
    .scenario footer {{ color: var(--muted); font-size: 12px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .method {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }}
    .method div {{ border-left: 3px solid var(--gold); padding: 4px 0 4px 12px; }}
    .method p {{ margin: 5px 0 0; color: var(--muted); font-size: 13px; }}
    .footer-note {{ margin: 22px 0 0; color: var(--muted); font-size: 12px; }}
    @media (max-width: 920px) {{
      .wrap {{ max-width: 390px; }}
      .hero, .grid-2 {{ grid-template-columns: 1fr; }}
      .headline-grid, .scenarios, .method {{ grid-template-columns: 1fr; }}
      nav {{ display: none; }}
      .hero-main {{ padding: 22px; }}
    }}
    @media (max-width: 520px) {{
      .wrap {{ padding: 0 16px; max-width: 100%; }}
      main {{ padding-top: 22px; }}
      .topbar {{ gap: 10px; }}
      .brand {{ min-width: 0; font-size: 14px; }}
      .brand span {{ overflow-wrap: anywhere; line-height: 1.2; }}
      .hero-main, .health {{ max-width: calc(100vw - 32px); overflow: hidden; }}
      h1 {{ max-width: 310px; font-size: 25px; line-height: 1.08; overflow-wrap: anywhere; word-break: break-word; }}
      .headline-grid, .probability, .metric {{ max-width: 310px; }}
      .metric {{ min-height: 0; }}
      .probability-labels {{ display: grid; grid-template-columns: 1fr; gap: 4px; font-size: 12px; }}
      .probability-labels span:last-child {{ text-align: left; }}
      .probability-track {{ width: 100%; max-width: 310px; }}
      .small-label {{ max-width: 310px; white-space: normal; overflow-wrap: anywhere; }}
      .health-row {{ align-items: flex-start; flex-direction: column; gap: 4px; }}
      .section {{ padding: 18px; }}
      .section-title {{ display: block; }}
      table {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="wrap topbar">
      <div class="brand"><div class="mark">TX</div><span>Texas GOP Senate Runoff Forecast</span></div>
      <nav aria-label="Dashboard sections">
        <span>Forecast</span><span>Scenarios</span><span>Audit</span><span>Method</span>
      </nav>
    </div>
  </header>
  <main>
    <div class="wrap">
      <section class="hero">
        <div class="panel hero-main">
          <p class="kicker">As of {esc(package['as_of'])} | May 26, 2026 Republican runoff</p>
          <h1>Paxton leads, but the forecast remains source-gated.</h1>
          <div class="headline-grid">
            <div class="metric"><span>Ken Paxton fair probability</span><strong class="red">{pct(full_prob)}</strong></div>
            <div class="metric"><span>John Cornyn fair probability</span><strong class="blue">{pct(cornyn_prob)}</strong></div>
            <div class="metric"><span>Mean margin</span><strong class="gold">{pts(headline['mean_paxton_margin_points'])} pts</strong></div>
          </div>
          <div class="probability">
            <div class="probability-labels"><span>Paxton {pct(full_prob)}</span><span>Cornyn {pct(cornyn_prob)}</span></div>
            <div class="probability-track" aria-label="Forecast probability bar"></div>
            <p class="small-label">80% margin interval: {pts(interval_low)} to {pts(interval_high)} points. Forecast status: {esc(package['release_status']['state'])}.</p>
          </div>
        </div>
        <aside class="panel health">
          <div class="health-row"><span>Release status</span><span class="pill {status_class}">{esc(package['release_status']['state']).title()}</span></div>
          <div class="health-row"><span>Reliability</span><strong>{float(release.reliability_score):.0f}/100 | {esc(release.reliability_class).title()}</strong></div>
          <div class="health-row"><span>Source freshness</span><strong>{source_pass_count}/{source_count} pass</strong></div>
          <div class="health-row"><span>County coverage</span><strong>{float(release.candidate_vote_coverage_pct):.0%}</strong></div>
          <div class="health-row"><span>Effective poll count</span><strong>{float(effective_poll_count):.1f}</strong></div>
          <div class="health-row"><span>Mid-turnout prior</span><strong>{float(mid_turnout):,.0f} votes</strong></div>
        </aside>
      </section>

      <section class="panel section">
        <div class="section-title">
          <h2>Scenario Range</h2>
          <p>Professional forecast, not a single-point promise.</p>
        </div>
        <div class="scenarios">{scenario_cards(scenarios)}</div>
      </section>

      <section class="grid-2">
        <div class="panel section">
          <div class="section-title"><h2>Sensitivity</h2><p>Largest shifts when a pollster is removed.</p></div>
          <table>
            <thead><tr><th>Removed</th><th>Paxton</th><th>Margin</th><th>Bar</th></tr></thead>
            <tbody>{sensitivity_rows(sensitivity, full_prob)}</tbody>
          </table>
        </div>
        <div class="panel section">
          <div class="section-title"><h2>Ablation</h2><p>How each model layer changes the forecast.</p></div>
          <table>
            <thead><tr><th>Layer</th><th>Margin</th><th>Paxton</th><th>Note</th></tr></thead>
            <tbody>{ablation_rows(ablation)}</tbody>
          </table>
        </div>
      </section>

      <section class="grid-2">
        <div class="panel section">
          <div class="section-title"><h2>Source Audit</h2><p>Freshness can pass via source check when no new data exists.</p></div>
          <table>
            <thead><tr><th>Dataset</th><th>Status</th><th>Basis</th><th>Latest data</th><th>Checked</th></tr></thead>
            <tbody>{source_rows(source_status)}</tbody>
          </table>
        </div>
        <div class="panel section">
          <div class="section-title"><h2>Market Comparison</h2><p>Research-only; excluded from the core model.</p></div>
          <table>
            <thead><tr><th>Platform</th><th>Flag</th><th>Buy price</th><th>Model</th><th>Blocks</th></tr></thead>
            <tbody>{market_rows(wager)}</tbody>
          </table>
        </div>
      </section>

      <section class="panel section">
        <div class="section-title">
          <h2>Methodology</h2>
          <p>Every adjustment is bounded and auditable.</p>
        </div>
        <div class="method">
          <div><h3>Polling</h3><p>Recency, sample, LV, partisan, and internal-cluster weighting.</p></div>
          <div><h3>Primary Prior</h3><p>First-round vote baseline plus Hunt-transfer evidence; baseline Hunt-to-Paxton transfer is {pct(float(baseline_hunt.paxton_transfer_share))}.</p></div>
          <div><h3>Turnout</h3><p>Low/mid/high retention scenarios stay explicit until early voting starts on May 18.</p></div>
          <div><h3>Release Gates</h3><p>Historical calibration limits reduce reliability, while current source, duplicate, and county gates control publication.</p></div>
        </div>
        <p class="footer-note">Generated from data/processed/model_output.json and companion CSVs. This is election-forecasting research, not financial advice.</p>
      </section>
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    SHOWCASE_DIR.mkdir(parents=True, exist_ok=True)
    html_text = build_html()
    output = SHOWCASE_DIR / "index.html"
    output.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
