"""Weekly multi-league scouting: one chronological team-state pass per league.

Only pre-kickoff finished results are used. Suggestions are rankings of model
probabilities, NOT calibrated guarantees or evidence of betting value.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from club_engine import Form, OUTCOMES, probabilities, soften
from club_data import current_season_results, fixture_teams
from live_data import next_fixtures, kickoff

SA_TIME=ZoneInfo("Africa/Johannesburg")
CHOICES={"1":"Home win","X":"Draw","2":"Away win"}

def forecast_week(history,trained,snapshot,league_name,now=None,horizon_days=7):
    """Predict every mapped future match with one historical state construction.

    This is equivalent to a fixture-by-fixture pre-match forecast, but much faster
    for mobile: historical results are applied once, not once per fixture.
    """
    now=now or datetime.now(timezone.utc)
    fixtures=next_fixtures(snapshot["matches"],now=now,horizon_days=horizon_days)
    output=[]
    stats={"upcoming":len(fixtures),"unmapped":0,"insufficient":0,"failed":0,"skipped_results":0}
    if not fixtures:return output,stats
    last_day=max(kickoff(m).date().isoformat() for m in fixtures if kickoff(m))
    extra,skipped=current_season_results(snapshot["matches"],"9999-12-31",history)
    stats["skipped_results"]=len(skipped)
    # The finished-match filter is controlled by dates below, never by future
    # information. A later completed match cannot influence an earlier forecast.
    records=sorted(history+extra,key=lambda r:(r["date"],r["home"],r["away"]))
    state=Form()
    j=0
    for fixture in fixtures:
        kick=kickoff(fixture)
        if kick is None:continue
        day=kick.date().isoformat()
        while j<len(records) and records[j]["date"]<day:
            state.update(records[j]);j+=1
        h,a=fixture_teams(fixture,history)
        if not h or not a or h==a:
            stats["unmapped"]+=1
            continue
        nh,na=len(state.games[h]),len(state.games[a])
        if min(nh,na)<5:
            stats["insufficient"]+=1
            continue
        try:
            feat=state.feature(h,a,day)
            base=state.baseline(h,a)
            challenger=soften(probabilities(trained["model"],pd.DataFrame([feat]),trained["columns"]),trained["temperature"])[0]
            p=challenger if trained["selected"]=="trained" else base
            if not np.all(np.isfinite(p)) or abs(float(p.sum())-1)>1e-4:
                stats["failed"]+=1;continue
        except (KeyError,TypeError,ValueError,IndexError):
            stats["failed"]+=1;continue
        fav=int(np.argmax(p))
        under=0 if base[0]<base[2] else 2
        uplift=float(challenger[under]-base[under])
        upset=bool(uplift>=.07 and challenger[under]>=.27)
        home=(fixture.get("homeTeam") or {}).get("shortName") or (fixture.get("homeTeam") or {}).get("name") or h.title()
        away=(fixture.get("awayTeam") or {}).get("shortName") or (fixture.get("awayTeam") or {}).get("name") or a.title()
        output.append(dict(
            id=str(fixture.get("id") or f"{league_name}-{day}-{h}-{a}"),
            league=league_name,home=home,away=away,
            home_code=h,away_code=a,datetime=kick.isoformat(),
            local_time=kick.astimezone(SA_TIME).strftime("%a %d %b · %H:%M SAST"),
            probabilities={k:float(p[i]) for i,k in enumerate(OUTCOMES)},
            pick=OUTCOMES[fav],pick_label=CHOICES[OUTCOMES[fav]],
            probability=float(p[fav]),engine=trained["selected"],
            baseline={k:float(base[i]) for i,k in enumerate(OUTCOMES)},
            challenger={k:float(challenger[i]) for i,k in enumerate(OUTCOMES)},
            upset_alert=upset,underdog=OUTCOMES[under],
            underdog_team=home if under==0 else away,
            upset_chance=float(challenger[under]),upset_uplift=uplift,
            historic_home=nh,historic_away=na,
            snapshot=snapshot.get("fetched"),
            source_note="Results, Elo and venue form only; live xG, PPDA, injuries and lineups are not included.",
        ))
    return output,stats

def strong_predictions(cards,threshold=.60):
    """Include only the highest probability single 1/X/2 choice per fixture."""
    return sorted((r for r in cards if r["probability"]+1e-12>=threshold),
                  key=lambda x:(-x["probability"],x["datetime"],x["league"]))

def upset_predictions(cards):
    """ML challenger signals, NOT an assertion that underdogs are likely winners."""
    return sorted((r for r in cards if r["upset_alert"]),
                  key=lambda x:(-x["upset_uplift"],-x["upset_chance"],x["datetime"]))
