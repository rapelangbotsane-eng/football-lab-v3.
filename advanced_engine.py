"""Event-level historical football model; research only, not validated for live betting."""
from collections import defaultdict
from datetime import date
from itertools import groupby
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score

LABELS=["1","X","2"]
METRICS=["xg","xga","shots","big_chances","progressive_passes","progressive_carries","box_entries","final_third_entries","pressures","press_to_regain","high_regains","ppda_proxy","goals","goals_against"]
PRIORS=dict(xg=1.3,xga=1.3,shots=11,big_chances=1,progressive_passes=18,progressive_carries=7,box_entries=6,final_third_entries=22,pressures=20,press_to_regain=.15,high_regains=5,ppda_proxy=11,goals=1.3,goals_against=1.3)

def name(x):
    return x.get("name","") if isinstance(x,dict) else str(x or "")
def loc(e,field=None):
    v=(e.get("location") if field is None else (e.get(field) or {}).get("end_location"))
    return v if isinstance(v,list) and len(v)>=2 else None

def team_match_metrics(events,home,away,hg,ag):
    s={t:defaultdict(float) for t in (home,away)}
    c={t:defaultdict(float) for t in (home,away)}
    ev=sorted(events,key=lambda e:(e.get("period",1),e.get("index",0)))
    for i,e in enumerate(ev):
        t=name(e.get("team"))
        if t not in s:continue
        opp=away if t==home else home
        typ=name(e.get("type")); p=loc(e)
        if typ=="Shot":
            xg=float((e.get("shot") or {}).get("statsbomb_xg") or 0)
            s[t]["xg"]+=xg;s[t]["shots"]+=1
            if xg>=.2:s[t]["big_chances"]+=1
        if typ in ("Pass","Carry") and p:
            k=typ.lower(); end=loc(e,k)
            if end and (typ=="Carry" or not (e.get("pass") or {}).get("outcome")):
                dx=end[0]-p[0]
                if dx>=(12 if typ=="Pass" else 10) and end[0]>=60:
                    s[t]["progressive_passes" if typ=="Pass" else "progressive_carries"]+=1
                if p[0]<80<=end[0]:s[t]["final_third_entries"]+=1
                if p[0]<102<=end[0] and 18<=end[1]<=62:s[t]["box_entries"]+=1
        if typ=="Pass" and p and p[0]<=72:c[opp]["opponent_passes"]+=1
        if typ in ("Duel","Interception","Pressure","Block","Ball Recovery") and p and p[0]>=48:
            c[t]["defensive_actions"]+=1
        if typ=="Ball Recovery" and p and p[0]>=80:s[t]["high_regains"]+=1
        if typ=="Pressure":
            s[t]["pressures"]+=1
            start=60*e.get("minute",0)+e.get("second",0)
            for other in ev[i+1:i+25]:
                if other.get("period")!=e.get("period"):break
                seconds=60*other.get("minute",0)+other.get("second",0)-start
                if seconds>5:break
                if seconds>=0 and name(other.get("team"))==t and name(other.get("type")) in ("Pass","Carry","Shot","Ball Recovery"):
                    c[t]["regains"]+=1;break
    result={}
    for t,gf,ga in ((home,hg,ag),(away,ag,hg)):
        d=s[t];d["xga"]=s[away if t==home else home]["xg"]
        d["goals"]=gf;d["goals_against"]=ga
        d["press_to_regain"]=c[t]["regains"]/d["pressures"] if d["pressures"] else 0
        d["ppda_proxy"]=min(35,c[t]["opponent_passes"]/max(1,c[t]["defensive_actions"]))
        result[t]={k:float(d[k]) for k in METRICS}
    return result

def match_record(m,events):
    h=m["home_team"].get("home_team_name") or m["home_team"].get("name")
    a=m["away_team"].get("away_team_name") or m["away_team"].get("name")
    date.fromisoformat(m["match_date"])
    return dict(match_id=int(m["match_id"]),date=m["match_date"],home=h,away=a,hg=m["home_score"],ag=m["away_score"],metrics=team_match_metrics(events,h,a,m["home_score"],m["away_score"]))

def pre_match_features(h,a,day,hist,elo,last):
    f={"elo_difference":elo[h]-elo[a]+55}
    for team,prefix in ((h,"h"),(a,"a")):
        records=hist[team][-10:]
        f[prefix+"_samples"]=len(records)
        f[prefix+"_rest_days"]=min(28,(date.fromisoformat(day)-date.fromisoformat(last[team])).days) if team in last else 14
        weights=np.exp(np.linspace(-1.6,0,len(records))) if records else []
        for metric in METRICS:
            f[prefix+"_"+metric]=(sum(w*r[1][metric] for w,r in zip(weights,records))+3*PRIORS[metric])/(sum(weights)+3)
    for k in METRICS:f["diff_"+k]=f["h_"+k]-f["a_"+k]
    return f

def fixture_rows(records):
    hist=defaultdict(list);elo=defaultdict(lambda:1500.);last={};rows=[]
    for day,grp in groupby(sorted(records,key=lambda m:(m["date"],m["match_id"])),key=lambda m:m["date"]):
        matches=list(grp)
        for m in matches:
            outcome="1" if m["hg"]>m["ag"] else "2" if m["hg"]<m["ag"] else "X"
            rows.append(dict(date=day,match_id=m["match_id"],home=m["home"],away=m["away"],hg=m["hg"],ag=m["ag"],outcome=outcome,**pre_match_features(m["home"],m["away"],day,hist,elo,last)))
        for m in matches:
            h,a=m["home"],m["away"]
            expected=1/(1+10**(-(elo[h]+55-elo[a])/400))
            actual=1 if m["hg"]>m["ag"] else .5 if m["hg"]==m["ag"] else 0
            change=21*(actual-expected);elo[h]+=change;elo[a]-=change
            hist[h].append((day,m["metrics"][h]));hist[a].append((day,m["metrics"][a]))
            last[h]=last[a]=day
    return pd.DataFrame(rows),hist,elo,last

def train_and_evaluate(df,min_train=60,min_test=20):
    f=df.sort_values(["date","match_id"]).reset_index(drop=True)
    if len(f)<min_train+min_test:raise ValueError("Need at least 80 event-complete matches")
    days=sorted(f.date.unique());threshold=days[min(len(days)-1,max(1,int(len(days)*.8)))]
    tr=f.date<threshold;te=~tr
    if tr.sum()<min_train or te.sum()<min_test:raise ValueError(f"Insufficient distinct dates: {tr.sum()} train, {te.sum()} test")
    cols=[c for c in f.columns if c.startswith(("h_","a_","diff_","elo_"))]
    X=f[cols].fillna(0).astype(float);y=f.outcome
    clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,C=.2))
    clf.fit(X[tr],y[tr])
    if len(clf[-1].classes_)!=3:raise ValueError("Training data must contain home wins, draws and away wins")
    p=clf.predict_proba(X[te])
    p=np.column_stack([p[:,list(clf[-1].classes_).index(k)] for k in LABELS])
    labels=f.loc[te,"outcome"].to_numpy()
    predicted=np.array(LABELS)[p.argmax(axis=1)]
    actual=np.eye(3)[[LABELS.index(k) for k in labels]]
    regressors={}
    for key,col in (("home","hg"),("away","ag")):
        r=HistGradientBoostingRegressor(loss="poisson",max_iter=90,max_leaf_nodes=7,min_samples_leaf=15,l2_regularization=25,random_state=42)
        r.fit(X[tr],f.loc[tr,col]);regressors[key]=r
    report=dict(matches=len(f),train_matches=int(tr.sum()),test_matches=int(te.sum()),test_start=threshold,
        # Cross-entropy computed explicitly to preserve 1/X/2 probability-column order.
        # sklearn.metrics.log_loss otherwise sorts class labels as 1/2/X.
        accuracy=float(accuracy_score(labels,predicted)),
        log_loss=float(-np.mean(np.log(np.clip(p[np.arange(len(labels)),[LABELS.index(label) for label in labels]],1e-15,1.0)))),
        brier_multiclass=float(np.mean(np.sum((p-actual)**2,axis=1))),
        baseline_accuracy=float(np.mean(labels==f.loc[tr,"outcome"].value_counts().idxmax())))
    holdout=f.loc[te,["date","home","away","hg","ag","outcome"]].copy()
    holdout["forecast"]=predicted
    for i,k in enumerate(LABELS):holdout["p_"+k]=p[:,i]
    clf.fit(X,y)
    for key,col in (("home","hg"),("away","ag")):regressors[key].fit(X,f[col])
    return dict(classifier=clf,regressors=regressors,columns=cols,report=report,holdout=holdout)

def make_prediction(model,h,a,day,hist,elo,last):
    if h==a:raise ValueError("Choose different teams")
    f=pre_match_features(h,a,day,hist,elo,last)
    X=pd.DataFrame([{k:f.get(k,0) for k in model["columns"]}])
    clf=model["classifier"];raw=clf.predict_proba(X)[0]
    ps=np.array([raw[list(clf[-1].classes_).index(k)] for k in LABELS])
    lh=float(model["regressors"]["home"].predict(X)[0]);la=float(model["regressors"]["away"].predict(X)[0])
    rng=np.random.default_rng(42);H=rng.poisson(max(.05,lh),10000);A=rng.poisson(max(.05,la),10000)
    category=np.where(H>A,0,np.where(H==A,1,2));base=np.bincount(category,minlength=3)/len(H)
    if min(base)>.002:
        weights=np.array([ps[k]/base[k] for k in category]);weights/=weights.sum()
        idx=rng.choice(len(H),size=len(H),replace=True,p=weights);H=H[idx];A=A[idx]
    scores=pd.Series([f"{x}–{y}" for x,y in zip(H,A)]).value_counts(normalize=True).head(8)
    elo_home=1/(1+10**(-(elo[h]+55-elo[a])/400))
    underdog="2" if elo_home>=.5 else "1"
    ep={"1":.75*elo_home,"X":.25,"2":.75*(1-elo_home)}
    uplift=ps[LABELS.index(underdog)]-ep[underdog]
    return dict(home=h,away=a,date=day,probs=dict(zip(LABELS,map(float,ps))),pick=LABELS[ps.argmax()],
                underdog=underdog,underdog_uplift=float(uplift),upset_watch=bool(min(f["h_samples"],f["a_samples"])>=5 and uplift>=.07 and ps[LABELS.index(underdog)]>=.25),
                low_data=min(f["h_samples"],f["a_samples"])<5,
                goal_expectations=[lh,la],top_scores=[dict(score=k,probability=float(v)) for k,v in scores.items()],
                metrics={k:(round(f["h_"+k],2),round(f["a_"+k],2)) for k in METRICS})
