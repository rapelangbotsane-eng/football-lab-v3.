"""Prediction ledger, immutable pre-kickoff probabilities and post-match grading.

This module is independent of Streamlit. File archive on free Streamlit servers
is a convenience, NOT permanent or private storage.
"""
from datetime import datetime,timezone
from pathlib import Path
from collections import defaultdict
from statistics import mean
import csv
import io
import json
import math
import os
import sqlite3

OUTCOMES=("1","X","2")
SCHEMA_VERSION=1
DB_PATH=Path(os.environ.get("FOOTBALL_LAB_TRACKER_DB","tracker_workspace/predictions.sqlite3"))

def utc(value):
    obj=value if isinstance(value,datetime) else datetime.fromisoformat(str(value).replace("Z","+00:00"))
    if obj.tzinfo is None:
        raise ValueError("Tracking timestamps must include a time zone")
    return obj.astimezone(timezone.utc)

def fixture_key(league_code,match_id):
    return f"{league_code}:{match_id}"

def snapshot_prediction(card,league_code,now=None):
    """Produce a verifiable snapshot, only if created strictly before kickoff."""
    now=utc(now or datetime.now(timezone.utc))
    kickoff=utc(card["datetime"])
    if now>=kickoff:
        raise ValueError("Cannot archive a prediction after kickoff")
    probs={key:float(card["probabilities"][key]) for key in OUTCOMES}
    if any(not math.isfinite(x) or x<0 or x>1 for x in probs.values()) or abs(sum(probs.values())-1)>1e-5:
        raise ValueError("Invalid prediction probability vector")
    code=str(card["pick"])
    if code not in OUTCOMES or abs(probs[code]-max(probs.values()))>1e-5:
        raise ValueError("Pick does not match the primary probability vector")
    snap=card.get("snapshot")
    if snap is not None and utc(snap)>now:
        raise ValueError("Source snapshot is newer than archive time")
    return dict(
        key=fixture_key(league_code,card["id"]),
        fixture_id=str(card["id"]),league_code=str(league_code),league=str(card["league"]),
        home=str(card["home"]),away=str(card["away"]),kickoff=kickoff.isoformat(),
        archived_at=now.isoformat(),source_fetched=snap,
        model_version=str(card.get("model_version") or "v3.3-results"),
        engine=str(card["engine"]),probabilities=probs,
        pick=code,pick_probability=probs[code],strong_60=probs[code]>=.60,
        upset_alert=bool(card.get("upset_alert",False)),
        upset_side=str(card.get("underdog") or ""),
        upset_team=str(card.get("underdog_team") or ""),
        upset_chance=float(card.get("upset_chance") or 0),
        upset_uplift=float(card.get("upset_uplift") or 0),
        result=None
    )

def _connect(path=DB_PATH):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    conn=sqlite3.connect(str(path),timeout=15)
    conn.execute("PRAGMA busy_timeout=15000")
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
      key TEXT PRIMARY KEY, archived_at TEXT NOT NULL,
      kickoff TEXT NOT NULL, json TEXT NOT NULL)""")
    return conn

def load_local(path=DB_PATH):
    with _connect(path) as con:
        payloads=con.execute("SELECT json FROM predictions ORDER BY kickoff DESC").fetchall()
    return [json.loads(r[0]) for r in payloads]

def archive_local(cards,league_code,now=None,path=DB_PATH):
    created=0; skipped=0
    with _connect(path) as con:
        for card in cards:
            try:row=snapshot_prediction(card,league_code,now)
            except (ValueError,KeyError,TypeError):
                skipped+=1;continue
            result=con.execute("""INSERT OR IGNORE INTO predictions(key,archived_at,kickoff,json)
                VALUES (?,?,?,?)""",(row["key"],row["archived_at"],row["kickoff"],json.dumps(row,separators=(",",":"))))
            created+=int(result.rowcount==1)
        con.commit()
    return dict(new=created,skipped=skipped)

def settle_one(entry,fixture,as_of=None):
    """Mark finished ONLY using identical fixture ID and explicit FINISHED score.

    Unfinished or unrelated fixture information is never taken as a result.
    """
    if entry.get("result") is not None:return entry
    if str(fixture.get("id"))!=str(entry.get("fixture_id")):
        return entry
    if fixture.get("status")!="FINISHED":return entry
    score=(fixture.get("score") or {}).get("fullTime") or {}
    hg,ag=score.get("home"),score.get("away")
    if isinstance(hg,bool) or isinstance(ag,bool) or not isinstance(hg,int) or not isinstance(ag,int):
        return entry
    if hg<0 or ag<0:return entry
    result="1" if hg>ag else "X" if hg==ag else "2"
    # The finish must be reported AFTER kickoff, never before.
    end=utc(as_of or datetime.now(timezone.utc))
    if end<=utc(entry["kickoff"]) or utc(entry["archived_at"])>=utc(entry["kickoff"]):
        return entry
    entry=dict(entry)
    entry["result"]=dict(outcome=result,home_goals=hg,away_goals=ag,checked_at=end.isoformat(),
        correct=entry["pick"]==result,
        upset_success=bool(entry.get("upset_alert") and entry.get("upset_side")==result))
    return entry

def settle_with_feed(rows,league_code,matches,as_of=None):
    """Returns new rows and number newly resolved, without mutating input."""
    by_id={str(m.get("id")):m for m in matches if m.get("id") is not None}
    count=0;out=[]
    for r in rows:
        if r["league_code"]==league_code and r.get("result") is None and r["fixture_id"] in by_id:
            updated=settle_one(r,by_id[r["fixture_id"]],as_of=as_of)
            count+=int(updated.get("result") is not None)
            out.append(updated)
        else:out.append(r)
    return out,count

def reconcile_local(league_code,matches,as_of=None,path=DB_PATH):
    updated=0
    with _connect(path) as con:
        for (payload,) in con.execute("SELECT json FROM predictions").fetchall():
            entry=json.loads(payload)
            if entry.get("result") or entry["league_code"]!=league_code:continue
            mapped={str(m.get("id")):m for m in matches}
            match=mapped.get(entry["fixture_id"])
            if not match:continue
            new=settle_one(entry,match,as_of=as_of)
            if new.get("result") is not None:
                con.execute("UPDATE predictions SET json=? WHERE key=? AND json=?",
                    (json.dumps(new,separators=(",",":")),entry["key"],payload))
                updated+=1
        con.commit()
    return updated

def merge_archives(primary,others):
    """Select earliest valid prediction snapshot; merge its later settlement only.

    Never replace older probabilities with newer (possibly post-kickoff) ones.
    """
    chosen={}
    for row in list(primary)+list(others):
        try:
            if utc(row["archived_at"])>=utc(row["kickoff"]):continue
            key=row["key"]
        except (KeyError,ValueError,TypeError):continue
        old=chosen.get(key)
        if old is None or utc(row["archived_at"])<utc(old["archived_at"]):
            chosen[key]=row
        elif utc(row["archived_at"])==utc(old["archived_at"]) and old.get("result") is None and row.get("result"):
            chosen[key]=row
        elif old.get("result") is None and row.get("result") and utc(row["archived_at"])==utc(old["archived_at"]):
            chosen[key]=row
        elif old.get("result") is None and row.get("result") and row.get("probabilities")==old.get("probabilities"):
            # A graded copy with the same probability vector can update an earlier snapshot.
            chosen[key]=dict(old,result=row["result"])
    return sorted(chosen.values(),key=lambda r:r["kickoff"],reverse=True)

def save_archive_local(rows,path=DB_PATH):
    """Merge trusted cloud/synced rows without rewriting original probability vectors."""
    existing=load_local(path)
    merged=merge_archives(existing,rows)
    original={r["key"]:r for r in existing}
    count=0
    with _connect(path) as con:
        for r in merged:
            old=original.get(r["key"])
            if old==r:continue
            # If the record already exists, only its final-score settlement may change.
            if old and old.get("result") is not None:
                continue
            if old and r.get("probabilities")!=old.get("probabilities"):
                continue
            payload=json.dumps(r,separators=(",",":"))
            con.execute("""INSERT INTO predictions (key,archived_at,kickoff,json)
                VALUES (?,?,?,?) ON CONFLICT(key) DO UPDATE SET json=excluded.json
                """,(r["key"],r["archived_at"],r["kickoff"],payload))
            count+=1
        con.commit()
    return count

def probability_quality(rows):
    """Proper-scoring rules for saved 1X2 forecasts, not retrospective models."""
    finished=[r for r in rows if r.get("result") and r.get("probabilities")]
    losses=[];briers=[]
    for row in finished:
        actual=row["result"]["outcome"]
        if actual not in OUTCOMES:continue
        try:
            p=[float(row["probabilities"][k]) for k in OUTCOMES]
        except (ValueError,TypeError,KeyError):continue
        if any(not math.isfinite(x) or x<0 or x>1 for x in p):continue
        index=OUTCOMES.index(actual)
        losses.append(-math.log(max(1e-12,p[index])))
        briers.append(sum((p[i]-(1 if i==index else 0))**2 for i in range(3)))
    return dict(samples=len(losses),
        log_loss=mean(losses) if losses else None,
        brier=mean(briers) if briers else None)

def confidence_buckets(rows):
    """Observed hit rate versus claimed confidence; meaningful after sufficient data."""
    groups=defaultdict(list)
    for row in rows:
        if not row.get("result"):continue
        p=float(row["pick_probability"])
        lower=min(90,int(p*10)*10)
        groups[lower].append(row)
    output=[]
    for lower,group in sorted(groups.items()):
        output.append(dict(range=f"{lower}–{lower+9}%",samples=len(group),
            average_model_probability=mean(float(r["pick_probability"]) for r in group),
            observed_win_rate=mean(int(r["result"]["correct"]) for r in group)))
    return output

def tracking_summary(rows):
    graded=[r for r in rows if r.get("result") is not None]
    all_count=len(rows)
    def group_statistics(items,pick="primary"):
        resolved=[r for r in items if r.get("result") is not None]
        n=len(resolved)
        if pick=="upset":
            won=sum(bool(r["result"]["upset_success"]) for r in resolved)
        else:won=sum(bool(r["result"]["correct"]) for r in resolved)
        return dict(resolved=n,correct=won,accuracy=(won/n if n else None))
    return dict(total=all_count,pending=all_count-len(graded),
        all=group_statistics(rows),
        strong=group_statistics([r for r in rows if r.get("strong_60")]),
        upset=group_statistics([r for r in rows if r.get("upset_alert")],pick="upset"))

def monthly_stats(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[row["kickoff"][:7]].append(row)
    result=[]
    for month,group in sorted(groups.items(),reverse=True):
        info=tracking_summary(group)
        result.append(dict(month=month,tracked=info["total"],resolved=info["all"]["resolved"],
            accuracy=info["all"]["accuracy"],strong_accuracy=info["strong"]["accuracy"],
            upset_precision=info["upset"]["accuracy"]))
    return result

def weekly_stats(rows):
    groups=defaultdict(list)
    for row in rows:
        dt=utc(row["kickoff"]).date()
        monday=dt.fromordinal(dt.toordinal()-dt.weekday()).isoformat()
        groups[monday].append(row)
    result=[]
    for week,group in sorted(groups.items(),reverse=True):
        info=tracking_summary(group)
        result.append(dict(week_start=week,tracked=info["total"],resolved=info["all"]["resolved"],
            accuracy=info["all"]["accuracy"],strong_accuracy=info["strong"]["accuracy"],
            upset_precision=info["upset"]["accuracy"]))
    return result

def to_csv(rows):
    fields=["archived_at","league","kickoff","home","away","engine","model_version",
        "pick","pick_probability","strong_60","upset_alert","upset_team","upset_chance",
        "result","correct","upset_success","home_goals","away_goals"]
    fp=io.StringIO();writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
    for row in rows:
        result=row.get("result") or {}
        writer.writerow(dict(archived_at=row["archived_at"],league=row["league"],kickoff=row["kickoff"],
            home=row["home"],away=row["away"],engine=row["engine"],model_version=row.get("model_version"),
            pick=row["pick"],pick_probability=row["pick_probability"],strong_60=row["strong_60"],
            upset_alert=row["upset_alert"],upset_team=row.get("upset_team"),
            upset_chance=row.get("upset_chance"),result=result.get("outcome"),
            correct=result.get("correct"),upset_success=result.get("upset_success"),
            home_goals=result.get("home_goals"),away_goals=result.get("away_goals")))
    return fp.getvalue()
