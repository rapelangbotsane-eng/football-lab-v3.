"""Leakage-safe retrospective EPL feature ablation for free advanced metrics.

Same matches and chronological splits for all models; never tune on 2025/26.
The 2025/26 season was already inspected in V3.1, so NOT a pristine holdout.
"""
from collections import defaultdict
from itertools import groupby
from datetime import date
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from club_engine import build_history, OUTCOMES, evaluate, soften, probabilities

ADVANCE=("xgf","xga","npxgf","ppda","deep","xpts")
PRIOR={"xgf":1.35,"xga":1.35,"npxgf":1.15,"ppda":12.0,"deep":7.0,"xpts":1.35}
ADV_XG=("xgf","xga","npxgf","xpts")
ADV_PRESS=("ppda","deep")

def make_rolling_advanced_features(results,matched):
    """Features for a given date are computed BEFORE adding that day's events."""
    histories=defaultdict(list)
    feats={}
    for day,group in groupby(sorted(results,key=lambda m:(m["date"],m["home"],m["away"])),key=lambda m:m["date"]):
        games=list(group)
        for game in games:
            h,a=game["home"],game["away"]
            row={}
            for tag,club in (("hx",h),("ax",a)):
                prev=histories[club][-8:]
                row[tag+"_samples"]=len(prev)
                for col in ADVANCE:
                    nums=[float(rec[col]) for rec in prev if rec.get(col) is not None and np.isfinite(float(rec[col]))]
                    # Prior smoothing: no misleading zero-fill when a metric is missing.
                    row[tag+"_"+col]=(sum(nums)+3*PRIOR[col])/(len(nums)+3)
                    row[tag+"_coverage_"+col]=len(nums)
                # Independently test a smaller, recency-weighted chance-quality
                # signal rather than many correlated raw metrics.
                for col in ("xgf","xga","npxgf"):
                    weighted=[]
                    for rec in prev:
                        v=rec.get(col)
                        if v is None:continue
                        try:
                            age=(date.fromisoformat(day)-date.fromisoformat(rec["date"])).days
                            if age<=0:continue
                            value=float(v)
                            if not np.isfinite(value):continue
                            weighted.append((2.**(-age/180.0),value))
                        except (ValueError,TypeError,KeyError):continue
                    effective=sum(weight for weight,value in weighted)
                    estimate=(sum(weight*value for weight,value in weighted)+
                        4*PRIOR[col])/(effective+4)
                    row[tag+"_decay_"+col]=estimate
            for col in ADVANCE:
                row["xgd_"+col]=row["hx_"+col]-row["ax_"+col]
            row["xgd_decay_net_xg"]=(row["hx_decay_xgf"]-row["hx_decay_xga"]-
                row["ax_decay_xgf"]+row["ax_decay_xga"])
            feats[(day,h,a)]=row
        for game in games:
            key=(day,game["home"],game["away"])
            if key not in matched:continue
            m=matched[key]
            for side,club in (("home_metrics",game["home"]),("away_metrics",game["away"])):
                histories[club].append(dict(m[side],date=day))
    return feats

def feature_frame(results,advanced_matches):
    basic,_=build_history(results)
    feats=make_rolling_advanced_features(results,advanced_matches)
    rows=[]
    for _,r in basic.iterrows():
        k=(r["date"],r["home"],r["away"])
        f=feats.get(k)
        if f is not None:
            rows.append(dict(**r.to_dict(),**f))
    df=pd.DataFrame(rows)
    if df.empty:raise ValueError("No fixtures with advanced pre-match features")
    return df

def _fit_predict(train,val,test,columns):
    model=make_pipeline(StandardScaler(),LogisticRegression(C=.12,max_iter=2000))
    model.fit(train[columns],train.outcome)
    if set(model[-1].classes_)!=set(OUTCOMES):
        raise ValueError("Training data missing an outcome class")
    vp=probabilities(model,val,columns)
    # Choose temperature using 2024/25 only.
    candidates=(.8,1.0,1.2,1.5,1.8,2.2,2.8)
    temp=min(candidates,key=lambda t:evaluate(val.outcome.to_numpy(),soften(vp,t))["log_loss"])
    val_probs=soften(vp,temp)
    test_probs=soften(probabilities(model,test,columns),temp)
    return dict(temperature=temp,
        validation=evaluate(val.outcome.to_numpy(),val_probs),
        retrospective_test=evaluate(test.outcome.to_numpy(),test_probs),
        val_probs=val_probs,test_probs=test_probs)

def run_xg_experiment(results,understat_matches,advanced_matched=None):
    from understat_data import aligned_metrics
    if advanced_matched is None:
        matched,unmatched=aligned_metrics(results,understat_matches)
    else:
        matched=advanced_matched
        unmatched=[r for r in results if (r["date"],r["home"],r["away"]) not in matched]
    f=feature_frame(results,matched)
    # Compare all models on the exact SAME rows with sufficient historical xG.
    f=f[(f.hx_samples>=4)&(f.ax_samples>=4)].copy()
    train=f[f.season<=2023]
    val=f[f.season==2024]
    test=f[f.season==2025]
    if min(len(train),len(val),len(test))<80:
        raise ValueError(f"Insufficient coverage to test fairly: {len(train)} training, {len(val)} validation, {len(test)} retrospective test matches (minimum 80 each).")
    exclude={"h_elo","a_elo"}
    # Match the existing V3.1 basic model's feature specification as closely as possible.
    base_cols=[c for c in f.columns if c=="elo_diff" or c in exclude or
               c.startswith(("h_","a_","d_"))]
    # No advanced columns share the basic h_/a_/d_ prefixes (uses hx_/ax_/xgd_).
    xg_cols=[c for c in f.columns if c in
        {tag+"_"+m for tag in ("hx","ax") for m in ADV_XG}|
        {"xgd_"+m for m in ADV_XG}]
    press_cols=[c for c in f.columns if c in
        {tag+"_"+m for tag in ("hx","ax") for m in ADV_PRESS}|
        {"xgd_"+m for m in ADV_PRESS}]
    models={
        "Elo only":None,
        "Results + Elo":base_cols,
        "Results + Elo + xG":base_cols+xg_cols,
        "Compact decayed xG + Elo":["elo_diff",
            "hx_decay_xgf","hx_decay_xga",
            "ax_decay_xgf","ax_decay_xga",
            "xgd_decay_net_xg"],
        "Compact decayed non-penalty xG + Elo":["elo_diff",
            "hx_decay_npxgf","hx_decay_xga",
            "ax_decay_npxgf","ax_decay_xga",
            "xgd_decay_net_xg"],
        "Results + Elo + xG + pressing":base_cols+xg_cols+press_cols,
    }
    results_by_model={}
    test_elo=test[["elo_1","elo_X","elo_2"]].to_numpy(dtype=float)
    val_elo=val[["elo_1","elo_X","elo_2"]].to_numpy(dtype=float)
    results_by_model["Elo only"]=dict(temperature=None,
        validation=evaluate(val.outcome.to_numpy(),val_elo),
        retrospective_test=evaluate(test.outcome.to_numpy(),test_elo))
    for name,cols in models.items():
        if cols is None:continue
        result=_fit_predict(train,val,test,cols)
        results_by_model[name]={k:v for k,v in result.items() if k not in ("val_probs","test_probs")}
    # Validation selects a winner BEFORE comparing the 2025 retrospective scores.
    chosen=min(results_by_model,key=lambda name:results_by_model[name]["validation"]["log_loss"])
    diagnostics=pd.DataFrame([dict(model=name,temp=rec["temperature"],
         val_log_loss=rec["validation"]["log_loss"],
         val_brier=rec["validation"]["brier"],
         val_accuracy=rec["validation"]["accuracy"],
         test_log_loss=rec["retrospective_test"]["log_loss"],
         test_brier=rec["retrospective_test"]["brier"],
         test_accuracy=rec["retrospective_test"]["accuracy"],
         selected=(name==chosen))
         for name,rec in results_by_model.items()])
    matched_by_season={year:sum(r["season"]==year for r in results if
         (r["date"],r["home"],r["away"]) in matched) for year in range(2020,2026)}
    return dict(table=diagnostics,selected=chosen,
        train_matches=len(train),validation_matches=len(val),
        retrospective_matches=len(test),source_matched=len(matched),
        source_unmatched=len(unmatched),matching_by_season=matched_by_season,
        used_features=dict(xg=xg_cols,pressing=press_cols),
        notes="2025/26 already examined in V3.1: retrospective diagnostic only, NOT an untouched final holdout. A future prospective season is needed to verify uplift.")
