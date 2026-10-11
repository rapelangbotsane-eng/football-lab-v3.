"""Opt-in Understat EPL xG research importer; selected historical seasons only.

Community-documented read-only AJAX endpoint. Unofficial, no stability guarantee.
Only use if access is permitted. No CAPTCHA/proxy/paywall or access-control bypass.
"""
from datetime import datetime
from collections import defaultdict
import time
import requests
from club_engine import canonical

BASE="https://understat.com"
START=2020
END=2025
METRIC_KEYS=("xgf","xga","npxgf","ppda","deep","xpts")
class UnderstatUnavailable(Exception):pass

def _decimal(value):
    try:
        if value is None or value=="":return None
        return float(value)
    except (ValueError,TypeError):return None

def _metrics(entry):
    ppda=entry.get("ppda") or {}
    p_att,p_def=_decimal(ppda.get("att")),_decimal(ppda.get("def"))
    p=p_att/p_def if p_att is not None and p_def and p_def>0 else None
    return dict(xgf=_decimal(entry.get("xG")),xga=_decimal(entry.get("xGA")),
        npxgf=_decimal(entry.get("npxG")),ppda=p,
        deep=_decimal(entry.get("deep")),xpts=_decimal(entry.get("xpts")))

def parse_understat_league(data,season):
    """Return completed match metrics without ever importing shots from future fixtures."""
    teams=data.get("teams") or {}
    matches=data.get("dates") or []
    if not isinstance(teams,dict) or not isinstance(matches,list):
        raise UnderstatUnavailable("Unexpected Understat response structure")
    team_index={}
    team_history=defaultdict(list)
    for ident,team in teams.items():
        tid=str(team.get("id") or ident)
        team_index[tid]=canonical(team.get("title"))
        for game in team.get("history",[]):
            day=str(game.get("date",""))[:10]
            side=str(game.get("h_a",""))
            if day and side in ("h","a"):
                team_history[(tid,day,side)].append(game)
    output=[]
    for match in matches:
        if not match.get("isResult"):continue
        h=match.get("h") or {};a=match.get("a") or {}
        hid,aid=str(h.get("id","")),str(a.get("id",""))
        day=str(match.get("datetime") or match.get("date") or "")[:10]
        home=canonical(h.get("title") or team_index.get(hid,""))
        away=canonical(a.get("title") or team_index.get(aid,""))
        if not day or not home or not away:continue
        hs=team_history.get((hid,day,"h"),[])
        as_=team_history.get((aid,day,"a"),[])
        if len(hs)!=1 or len(as_)!=1:continue
        hstats=_metrics(hs[0]);astats=_metrics(as_[0])
        if hstats["xgf"] is None or astats["xgf"] is None:continue
        # The same fixture must have matching scores in the historical CSV join.
        goals=match.get("goals") or {}
        try:
            hg,ag=int(goals["h"]),int(goals["a"])
        except (ValueError,TypeError,KeyError):continue
        output.append(dict(date=day,season=int(season),
            home=home,away=away,hg=hg,ag=ag,home_metrics=hstats,away_metrics=astats,
            provider="understat",fetched_at=None))
    return output

def retrieve_league_season(season,session=None):
    if not isinstance(season,int) or not START<=season<=END:
        raise UnderstatUnavailable("Season outside the historical research range")
    s=session or requests.Session()
    try:
        response=s.get(f"{BASE}/getLeagueData/EPL/{season}",
            headers={"Accept":"application/json, text/javascript, */*; q=0.01",
                     "X-Requested-With":"XMLHttpRequest",
                     "User-Agent":"FootballLabResearch/3.2 (+https://github.com/rapelangbotsane-eng/football-lab-v3.)"},
            timeout=25)
        if response.status_code in (401,403,429):
            raise UnderstatUnavailable(f"Provider refused access: HTTP {response.status_code}; stopped.")
        response.raise_for_status()
        data=response.json()
    except (requests.RequestException,ValueError) as e:
        raise UnderstatUnavailable(f"Could not read Understat {season}: {type(e).__name__}.") from e
    rows=parse_understat_league(data,season)
    if len(rows)<80:
        raise UnderstatUnavailable(f"Only {len(rows)} completed event-metric matches in {season}; expected a fuller season.")
    stamp=datetime.now().astimezone().isoformat()
    for row in rows:row["fetched_at"]=stamp
    return rows

def retrieve_historical(progress=None):
    all_rows=[]
    with requests.Session() as session:
        for i,season in enumerate(range(START,END+1)):
            season_rows=retrieve_league_season(season,session=session)
            all_rows.extend(season_rows)
            if progress:progress(i+1,END-START+1,len(season_rows))
            if i<END-START:time.sleep(1.1)
    unique={(r["date"],r["home"],r["away"]):r for r in all_rows}
    return sorted(unique.values(),key=lambda r:(r["date"],r["home"],r["away"]))

def aligned_metrics(results,understat_rows):
    """Exact team/date/score match only; never fuzzy-join xG from another fixture."""
    lookup={(x["date"],x["home"],x["away"],x["hg"],x["ag"]):x for x in understat_rows}
    matched={};unknown=[]
    for r in results:
        k=(r["date"],r["home"],r["away"],r["hg"],r["ag"])
        item=lookup.get(k)
        if item:matched[(r["date"],r["home"],r["away"])]=item
        else:unknown.append(dict(season=r["season"],date=r["date"],home=r["home"],away=r["away"]))
    return matched,unknown
