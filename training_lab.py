"""Football Lab research training: rolling-origin, chronological, results-only benchmarks.

This is NOT the model serving live predictions. No model is promoted based on
retrospective tests. The test years (including 2025/26) are research diagnostics,
not pristine independent holdouts after earlier project experiments.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from club_engine import build_history, OUTCOMES, evaluate, soften, upset_diagnostics

TEMPERATURES=(.85,1.0,1.15,1.35,1.6,2.0,2.5)
TEST_SEASONS=(2023,2024,2025)
MIN_TRAIN,MIN_VALIDATION,MIN_TEST=150,80,80

def feature_groups(frame):
    """All features are derived from pre-match form and Elo ratings."""
    all_cols=[c for c in frame.columns
        if c=="elo_diff" or c in ("h_elo","a_elo") or
            c.startswith(("h_","a_","d_"))]
    if "elo_diff" not in all_cols:raise ValueError("Elo differential is required")
    return {
        "Calibrated Elo":["elo_diff"],
        "Results + Elo logistic":all_cols,
        "Results + Elo boosted":all_cols,
    }

def _model(name):
    if name=="Calibrated Elo":
        return make_pipeline(StandardScaler(),LogisticRegression(
            C=.08,max_iter=1500,random_state=17))
    if name=="Results + Elo logistic":
        return make_pipeline(StandardScaler(),LogisticRegression(
            C=.16,max_iter=1500,random_state=17))
    if name=="Results + Elo boosted":
        return HistGradientBoostingClassifier(
            learning_rate=.045,max_iter=110,max_leaf_nodes=7,
            min_samples_leaf=30,l2_regularization=15.0,
            early_stopping=False,random_state=17)
    raise ValueError("Unknown research model")

def _proba(model,features):
    values=np.asarray(model.predict_proba(features),dtype=float)
    classes=list(model.classes_ if hasattr(model,"classes_") else model[-1].classes_)
    return np.column_stack([values[:,classes.index(o)] for o in OUTCOMES])

def _safe_metrics(y,p):
    out=evaluate(np.asarray(y),np.asarray(p,dtype=float))
    if not all(np.isfinite(v) for v in out.values()):
        raise ValueError("Non-finite metrics")
    return {k:round(v,6) for k,v in out.items()}

def _outcome_accuracy_at_threshold(actual,p,threshold=.60):
    actual=np.asarray(actual)
    p=np.asarray(p,dtype=float)
    idx=np.argmax(p,axis=1)
    top=np.max(p,axis=1)
    eligible=top>=threshold
    correct=np.array([OUTCOMES.index(y) for y in actual])==idx
    count=int(eligible.sum())
    return dict(matches=count,wins=int((correct&eligible).sum()),
        hit_rate=round(float(np.mean(correct[eligible])),5) if count else None,
        mean_confidence=round(float(np.mean(top[eligible])),5) if count else None)

def _forecast_candidate(name,train,val,test,columns):
    xtrain=train[columns].astype(float);xval=val[columns].astype(float)
    xtest=test[columns].astype(float)
    clf=_model(name)
    clf.fit(xtrain,train["outcome"])
    if set(clf.classes_)!=set(OUTCOMES):
        raise ValueError("Missing outcome class in fitted model")
    vp=_proba(clf,xval)
    # All calibration/selection happens on the previous season ONLY.
    temp=min(TEMPERATURES,key=lambda t:evaluate(
        val["outcome"].to_numpy(),soften(vp,t))["log_loss"])
    return dict(
        name=name,temperature=temp,
        validation=_safe_metrics(val["outcome"],soften(vp,temp)),
        test=_safe_metrics(test["outcome"],soften(_proba(clf,xtest),temp)),
        test_proba=soften(_proba(clf,xtest),temp))

def rolling_benchmark(records,test_seasons=TEST_SEASONS):
    """Each test year uses earlier training, immediately prior validation season.

    All match-level features were computed in forward calendar order.
    Each season is evaluated on precisely the same fixtures across candidates.
    """
    frame,_=build_history(records)
    if frame.empty:raise ValueError("Historical data is empty")
    groups=feature_groups(frame)
    folds=[];all_rows=[]
    for year in test_seasons:
        train=frame[frame.season<=year-2].copy()
        val=frame[frame.season==year-1].copy()
        test=frame[frame.season==year].copy()
        if len(train)<MIN_TRAIN or len(val)<MIN_VALIDATION or len(test)<MIN_TEST:
            raise ValueError(
                f"Season {year}: insufficient train/validation/test: "
                f"{len(train)}/{len(val)}/{len(test)}")
        test_labels=test["outcome"].to_numpy()
        elo_val=val[["elo_1","elo_X","elo_2"]].to_numpy(dtype=float)
        elo_test=test[["elo_1","elo_X","elo_2"]].to_numpy(dtype=float)
        metrics={
            "Fixed Elo":dict(name="Fixed Elo",temperature=None,
                validation=_safe_metrics(val["outcome"],elo_val),
                test=_safe_metrics(test_labels,elo_test),
                test_proba=elo_test)
        }
        for name,cols in groups.items():
            metrics[name]=_forecast_candidate(name,train,val,test,cols)
        # Choose using validation log loss; fixed model names are tie-breakers.
        selected=min(metrics,key=lambda n:(metrics[n]["validation"]["log_loss"],n))
        candidate=metrics[selected]["test_proba"]
        diagnostic=upset_diagnostics(candidate,elo_test,test_labels)
        strong=_outcome_accuracy_at_threshold(test_labels,candidate)
        report=dict(
            test_season=year,training_through=year-2,validation_season=year-1,
            train_matches=len(train),validation_matches=len(val),
            test_matches=len(test),selected_by_validation=selected,
            selected_test=metrics[selected]["test"],baseline_test=metrics["Fixed Elo"]["test"],
            selected_60_plus=strong,
            selected_upset_vs_elo={
                "alerts":diagnostic["alerts"],"captured":diagnostic["captured"],
                "false_alarms":diagnostic["false_alarms"],
                "precision":diagnostic["hit_rate"],"recall":diagnostic["recall"],
            },
            competitors=[
                dict(model=name,temperature=metrics[name]["temperature"],
                    validation=metrics[name]["validation"],test=metrics[name]["test"])
                for name in metrics
            ])
        folds.append(report)
        for i,(_,r) in enumerate(test.iterrows()):
            all_rows.append(dict(
                date=r["date"],test_season=year,
                home=r["home"],away=r["away"],actual=r["outcome"],
                selected=selected,
                p_1=float(candidate[i,0]),p_X=float(candidate[i,1]),p_2=float(candidate[i,2]),
                elo_1=float(elo_test[i,0]),elo_X=float(elo_test[i,1]),elo_2=float(elo_test[i,2])))
    # Don't merge folds into a deceptively "untouched" score.
    return dict(folds=folds,number_of_folds=len(folds),
        matching_predicted_fixtures=len(all_rows),forecast_rows=all_rows,
        notes=("Retrospective development diagnostic. The seasons have been examined "
            "during Football Lab development; this is NOT a fresh prospective test. "
            "The live engine is unchanged. No automatic promotion."))

def train_league(code):
    from club_data import historical_league
    records=historical_league(code)
    result=rolling_benchmark(records)
    result["league_code"]=code
    result["historical_matches"]=len(records)
    result["historical_first"]=min(r["date"] for r in records)
    result["historical_last"]=max(r["date"] for r in records)
    return result
