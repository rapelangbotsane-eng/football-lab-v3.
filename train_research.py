"""Cloud training runner; saves auditable diagnostics, not auto-promoted models.

Usage: python train_research.py --league PL --advanced
"""
import argparse
import json
import sys
import traceback
from datetime import datetime,timezone
from pathlib import Path
from training_lab import train_league

OUT=Path("data/training_report.json")

def _compact(report):
    # Avoid committing thousands of fixture-level records into the app repo.
    return {k:v for k,v in report.items() if k!="forecast_rows"}

def execute(leagues=("PL",),include_advanced=False):
    report=dict(
        schema_version=1,
        generated_at=datetime.now(timezone.utc).isoformat(),
        method="Rolling-origin historical men's club results",
        test_seasons=[2023,2024,2025],
        status="completed",
        leagues={},
        errors={},
        advanced={
            "status":"not_run",
            "note":"Advanced xG comparison is research-only. Live model not changed."},
        live_model_promoted=False,
        caution=("2023–2025 seasons have previously been used during Football Lab "
                 "development. These are DEVELOPMENT diagnostics, not an independent "
                 "prospective validation. The original live Elo/form engine is unchanged."),
        )
    for code in leagues:
        try:
            print(f"Training {code}: downloading 2020/21–2025/26 match results...",flush=True)
            training=train_league(code)
            report["leagues"][code]=_compact(training)
            fold=training["folds"][-1]
            print(f"{code}: {training['historical_matches']} matches; "
                  f"latest retrospective selected={fold['selected_by_validation']}; "
                  f"2025/26 log loss={fold['selected_test']['log_loss']:.3f} "
                  f"vs fixed Elo {fold['baseline_test']['log_loss']:.3f}",flush=True)
        except Exception as error:
            report["errors"][code]=f"{type(error).__name__}: {str(error)[:250]}"
            print(f"{code}: FAILED ({type(error).__name__}: {error})",flush=True)
    if include_advanced and "PL" in report["leagues"]:
        try:
            from understat_data import retrieve_historical
            from advanced_club_experiment import run_xg_experiment
            from club_data import historical_league
            print("Attempting Understat 2020–2025 advanced data for EPL, research-only...",flush=True)
            advanced=retrieve_historical()
            result=run_xg_experiment(historical_league("PL"),advanced)
            report["advanced"]={
                "status":"completed",
                "matched_matches":result["source_matched"],
                "unmatched_matches":result["source_unmatched"],
                "season_coverage":result["matching_by_season"],
                "train_matches":result["train_matches"],
                "validation_matches":result["validation_matches"],
                "retrospective_matches":result["retrospective_matches"],
                "selected_on_validation":result["selected"],
                "comparison":json.loads(result["table"].to_json(orient="records")),
                "note":result["notes"]+"\nLive fixtures continue with the old results-based model.",
            }
            print(f"Advanced EPL xG experiment complete: {result['source_matched']} matched games",flush=True)
        except Exception as error:
            report["advanced"]={
                "status":"unavailable",
                "reason":f"{type(error).__name__}: {str(error)[:260]}",
                "note":"Do not impute missing xG/pressing data or silently substitute results.",
            }
            print("Advanced EPL data unavailable. Baseline results remain valid:",report["advanced"]["reason"],flush=True)
    if not report["leagues"]:
        report["status"]="failed"
    elif report["errors"] or report["advanced"]["status"]=="unavailable":
        report["status"]="partial"
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Training report written to {OUT} (status={report['status']})",flush=True)
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--league",action="append",default=None,choices=["PL","PD","BL1","SA","FL1","DED","PPL"])
    parser.add_argument("--advanced",action="store_true")
    args=parser.parse_args()
    result=execute(leagues=tuple(args.league or ["PL"]),include_advanced=args.advanced)
    sys.exit(0 if result["leagues"] else 1)
