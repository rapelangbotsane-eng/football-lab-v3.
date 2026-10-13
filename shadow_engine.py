"""Frozen prospective challengers: paired, pre-kickoff 1/X/2 forecasts.

No live-model promotion. Calibrated Elo trains on the fixed historical
2020/21--2025/26 sample, independent of 2026/27 results. Goal model uses a
documented fixed configuration; it is NOT retrospectively tuned on outcomes.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from club_engine import Form, OUTCOMES, build_history
from club_data import current_season_results
from dynamic_goal_engine import DynamicGoals, outcome_from_goals

MODEL_VERSION="shadow-v1-2026-10-13"
GOAL_CONFIG=dict(half_life=240,shrink=4,rho=-.10,blend=.50)

def _probabilities(values):
    p=np.asarray(values,dtype=float)
    if p.shape!=(3,) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or abs(p.sum()-1)>1e-5:
        raise ValueError("Invalid shadow model probabilities")
    return {k:float(v) for k,v in zip(OUTCOMES,p)}

def fit_historical_elo_calibration(history):
    """Frozen sample: only completed seasons through 2025/26."""
    historical=[r for r in history if int(r["season"])<=2025]
    frame,_=build_history(historical)
    if len(frame)<500 or len(set(frame.outcome))!=3:
        raise ValueError("Not enough completed historical results for Elo calibration")
    calibrated=make_pipeline(StandardScaler(),
        LogisticRegression(C=.08,max_iter=1500,random_state=17))
    calibrated.fit(frame[["elo_diff"]],frame.outcome)
    return calibrated

def shadow_forecast_cards(history,cards,api_matches,calibrator=None):
    """Attach three benchmark forecasts using only games before fixture DAY.

    Per-fixture primary predictions from the caller are never modified.
    Only append a separate, research-only shadow_models dictionary.
    """
    if not cards:return []
    calibration=calibrator if calibrator is not None else fit_historical_elo_calibration(history)
    additional,_=current_season_results(api_matches,"9999-12-31",history)
    records=sorted(history+additional,key=lambda r:(r["date"],r["home"],r["away"]))
    ordered=sorted(cards,key=lambda c:(c["datetime"],str(c["id"])))
    elo_state=Form()
    goals_state=DynamicGoals(GOAL_CONFIG["half_life"],GOAL_CONFIG["shrink"])
    j=0;output=[]
    for card in ordered:
        day=str(card["datetime"])[:10]
        # Both models use strictly earlier dates. Same-day matches cannot leak.
        while j<len(records) and records[j]["date"]<day:
            elo_state.update(records[j]);goals_state.update(records[j]);j+=1
        home,away=card["home_code"],card["away_code"]
        primary_baseline=np.array([card["baseline"][x] for x in OUTCOMES],dtype=float)
        recomputed=elo_state.baseline(home,away)
        # A failed join/cutoff is NOT a reason to invent probabilities.
        if not np.allclose(primary_baseline,recomputed,rtol=0,atol=1e-7):
            raise ValueError("Elo fixture-state mismatch; refusing non-comparable shadow forecasts")
        features=elo_state.feature(home,away,day)
        raw=calibration.predict_proba(pd.DataFrame([{"elo_diff":features["elo_diff"]}]))[0]
        classes=list(calibration[-1].classes_)
        calibrated=np.array([raw[classes.index(outcome)] for outcome in OUTCOMES])
        lh,la=goals_state.rates(home,away,day)
        goal_probs=outcome_from_goals(lh,la,GOAL_CONFIG["rho"])
        mixed=GOAL_CONFIG["blend"]*goal_probs+(1-GOAL_CONFIG["blend"])*recomputed
        new=dict(card)
        new["model_version"]=MODEL_VERSION
        new["shadow_models"]={
            "Fixed Elo":_probabilities(recomputed),
            "Calibrated Elo":_probabilities(calibrated),
            "Dynamic goals + Elo":_probabilities(mixed),
        }
        output.append(new)
    return output
