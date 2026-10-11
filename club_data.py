"""Free public historical men's league results; no API key used.

Source: https://www.football-data.co.uk/
CSV fields used: Date, HomeTeam, AwayTeam, FTHG, FTAG.
We deliberately DO NOT train on odds or historical event metrics absent from the live API.
"""
from io import StringIO
from datetime import datetime
import pandas as pd
import requests
from club_engine import canonical, LEAGUES

HISTORICAL_START=2020
HISTORICAL_LAST=2025
BASE="https://www.football-data.co.uk/mmz4281"

class HistoricalDataError(Exception):
    pass

def season_results(league_code,year):
    if league_code not in LEAGUES:
        raise HistoricalDataError("Unsupported league code")
    if not isinstance(year,int) or year<HISTORICAL_START or year>HISTORICAL_LAST:
        raise HistoricalDataError("Historical season must be between 2020 and 2025")
    code=f"{year%100:02d}{(year+1)%100:02d}"
    url=f"{BASE}/{code}/{LEAGUES[league_code]}.csv"
    try:
        response=requests.get(url,headers={"User-Agent":"Mozilla/5.0 (compatible; FootballLabResearch/3.1)"},timeout=25)
        response.raise_for_status()
        if "<html" in response.text[:300].lower() or len(response.content)<1000:
            raise HistoricalDataError("Received a web page or empty file instead of match CSV")
        df=pd.read_csv(StringIO(response.content.decode("utf-8-sig",errors="replace")),on_bad_lines="skip")
    except (requests.RequestException,UnicodeError,ValueError,pd.errors.ParserError) as exc:
        raise HistoricalDataError(f"Could not retrieve {year}/{year+1} data: {type(exc).__name__}: {exc}") from exc
    need={"Date","HomeTeam","AwayTeam","FTHG","FTAG"}
    if not need.issubset(df.columns):
        raise HistoricalDataError(f"CSV for {year} lacks required columns: {sorted(need-df.columns)}")
    dates=pd.to_datetime(df["Date"],errors="coerce",dayfirst=True,format="mixed")
    output=[]
    for i,(_,r) in enumerate(df.iterrows()):
        if pd.isna(dates.iloc[i]):continue
        try:
            hg,ag=int(r.FTHG),int(r.FTAG)
            if hg<0 or ag<0 or hg>30 or ag>30:continue
        except (TypeError,ValueError,OverflowError):continue
        h,a=canonical(r.HomeTeam),canonical(r.AwayTeam)
        if not h or not a or h==a:continue
        output.append(dict(date=dates.iloc[i].date().isoformat(),season=year,
            home=h,away=a,hg=hg,ag=ag))
    if len(output)<80:
        raise HistoricalDataError(f"Only {len(output)} usable fixtures in {year}/{year+1} CSV")
    return output

def historical_league(league_code,progress=None):
    all_games=[]
    for idx,season in enumerate(range(HISTORICAL_START,HISTORICAL_LAST+1)):
        batch=season_results(league_code,season)
        all_games.extend(batch)
        if progress:progress(idx+1,6,len(batch),len(all_games))
    # Dedupe historical date/team pairs and do not silently inflate training size.
    dedup={(r["date"],r["home"],r["away"]):r for r in all_games}
    return sorted(dedup.values(),key=lambda m:(m["date"],m["home"],m["away"]))

def current_season_results(api_matches,fixture_date,known_names):
    """Add only matches finished before fixture date and outside historical CSV span.

    If a team can't be matched exactly or via explicit alias, skip it;
    this prevents the false-history bug from fuzzy joining team names.
    """
    results=[];skipped=set()
    cutoff="2026-07-01"
    canonical_clubs={n for r in known_names for n in (r["home"],r["away"])}
    for m in api_matches:
        if m.get("status")!="FINISHED":continue
        dt=str(m.get("utcDate",""))[:10]
        if not (cutoff<=dt<fixture_date):continue
        score=(m.get("score") or {}).get("fullTime") or {}
        hg,ag=score.get("home"),score.get("away")
        if not isinstance(hg,int) or not isinstance(ag,int):continue
        h=match_api_name(m.get("homeTeam") or {},canonical_clubs)
        a=match_api_name(m.get("awayTeam") or {},canonical_clubs)
        if not h or not a or h==a:
            skipped.add(f"{(m.get('homeTeam') or {}).get('shortName')} vs {(m.get('awayTeam') or {}).get('shortName')}")
            continue
        results.append(dict(date=dt,season=2026,home=h,away=a,hg=hg,ag=ag))
    unique={(r["date"],r["home"],r["away"]):r for r in results}
    return list(unique.values()),skipped

def match_api_name(team,known_clubs):
    names=[team.get("shortName"),team.get("name"),team.get("tla")]
    for value in names:
        if not value:continue
        canon=canonical(value)
        if canon in known_clubs:return canon
    return None

def fixture_teams(fixture,historical):
    known={n for r in historical for n in (r["home"],r["away"])}
    h=match_api_name(fixture.get("homeTeam") or {},known)
    a=match_api_name(fixture.get("awayTeam") or {},known)
    return h,a
