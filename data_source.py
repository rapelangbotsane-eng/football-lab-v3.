"""StatsBomb Open Data ingestion: selected historical matches only, NOT a live feed."""
from pathlib import Path
import json
import requests
from advanced_engine import match_record

BASE="https://raw.githubusercontent.com/statsbomb/open-data/master/data"
TIMEOUT=25

def get_json(path):
    uri=f"{BASE}/{path}"
    resp=requests.get(uri,timeout=TIMEOUT,headers={"User-Agent":"FootballLabResearch/3.0"})
    resp.raise_for_status()
    return resp.json()


def list_competitions():
    entries=get_json("competitions.json")
    # Most recent first, and readable competition name
    return sorted(entries,key=lambda x:(str(x.get("competition_name","")),str(x.get("season_name",""))))


def list_matches(cid,sid):
    return sorted(get_json(f"matches/{cid}/{sid}.json"),key=lambda m:m.get("match_date", ""))


def collect(records,limit=125,cache_dir="data_cache",on_progress=None):
    """Cache open-data events locally; skip incomplete or unavailable records."""
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    completed=[];fail=[]
    to_fetch=[m for m in records if isinstance(m.get("home_score"),int) and isinstance(m.get("away_score"),int)]
    for i,m in enumerate(to_fetch[:limit]):
        mid=int(m["match_id"]);path=cache/f"{mid}.json"
        try:
            if path.exists():events=json.loads(path.read_text())
            else:
                events=get_json(f"events/{mid}.json")
                path.write_text(json.dumps(events),encoding="utf-8")
            if not isinstance(events,list) or len(events)<100:raise ValueError("Too few events")
            completed.append(match_record(m,events))
        except (requests.RequestException,ValueError,KeyError,TypeError,TimeoutError) as exc:
            fail.append(f"{mid}: {exc}")
        if on_progress:on_progress((i+1)/max(1,min(limit,len(to_fetch))),len(completed),len(fail))
    return completed,fail
