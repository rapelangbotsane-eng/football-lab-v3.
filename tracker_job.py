"""Scheduled unattended forecast archive, run by GitHub Actions with repo secret.

No secrets are printed, committed or embedded in forecast records.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import os
import sys
from club_data import historical_league
from club_engine import build_history,train_model
from live_data import fetch_season_matches,LEAGUES
from weekly_scout import forecast_week
from prediction_tracker import snapshot_prediction,settle_with_feed,merge_archives

PATH=Path("data/tracker_archive.json")
DEFAULT_CODES=("PL","PD","BL1","SA","FL1")
LABELS={v:k for k,v in LEAGUES.items()}

def job_run(token,now=None):
    now=now or datetime.now(timezone.utc)
    if not token:
        print("FOOTBALL_DATA_API_TOKEN GitHub Actions secret not set. Scheduled archival skipped.")
        return 0
    if PATH.exists():
        try:archive=json.loads(PATH.read_text(encoding="utf-8"))
        except (ValueError,UnicodeError) as error:
            raise RuntimeError("Tracker archive unreadable: refusing overwrite") from error
        if not isinstance(archive,list):raise RuntimeError("Tracker archive must be a JSON list")
    else:archive=[]
    new_count=0;settled_count=0;failures={}
    for code in DEFAULT_CODES:
        try:
            snapshot=fetch_season_matches(token,code)
            archive,graded=settle_with_feed(archive,code,snapshot["matches"],as_of=now)
            settled_count+=graded
        except Exception as exc:
            failures[code]=f"API/final-score retrieval failed: {type(exc).__name__}"
            print(code,failures[code]);continue
        try:
            records=historical_league(code)
            frame,_=build_history(records)
            trained=train_model(frame)
            cards,_=forecast_week(records,trained,snapshot,LABELS[code],now=now,horizon_days=10)
            recent=[]
            for card in cards:
                row=snapshot_prediction(card,code,now=now)
                recent.append(row)
            old_count=len(archive)
            archive=merge_archives(archive,recent)
            new_count+=len(archive)-old_count
        except Exception as exc:
            failures[code]=f"Forecast generation failed: {type(exc).__name__}"
            print(code,failures[code]);continue
    PATH.parent.mkdir(parents=True,exist_ok=True)
    PATH.write_text(json.dumps(archive,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Tracker archive: {len(archive)} fixture(s), {new_count} new snapshots, {settled_count} newly settled.")
    if failures:print("Partial league failures (existing predictions preserved):",failures)
    if len(failures)==len(DEFAULT_CODES):
        return 1
    return 0

if __name__=="__main__":
    sys.exit(job_run(os.environ.get("FOOTBALL_DATA_API_TOKEN")))
