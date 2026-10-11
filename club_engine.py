"""Football Lab V3.1: men's league match-result model.

This deliberately uses ONLY fields obtainable before today's matches from the
free results API: completed results, home/away form, Elo and rest. StatsBomb
event-trained advanced model remains a separate, historical research engine.
"""
from collections import defaultdict
from datetime import date
from itertools import groupby
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTCOMES=("1","X","2")
HOME_ADVANTAGE=58.0
LEAGUES={"PL":"E0","PD":"SP1","BL1":"D1","SA":"I1","FL1":"F1","DED":"N1","PPL":"P1"}

# Canonical aliases for publicly available results files and the current API.
# Unknown clubs are flagged and never silently matched to a different club.
ALIASES={
    "manchester city":"man city","manchester united":"man united","man utd":"man united",
    "man city":"man city","man united":"man united",
    "nottingham forest":"nottm forest","nott m forest":"nottm forest",
    "newcastle united":"newcastle","west ham united":"west ham",
    "tottenham hotspur":"tottenham","brighton hove albion":"brighton",
    "wolverhampton wanderers":"wolves","wolves":"wolves","afc bournemouth":"bournemouth",
    "leicester city":"leicester","leeds united":"leeds","ipswich town":"ipswich",
    "sheffield united":"sheffield united","norwich city":"norwich","hull city":"hull",
    "inter milan":"inter","internazionale":"inter","fc internazionale milano":"inter",
    "ac milan":"milan","ssc napoli":"napoli","as roma":"roma","ss lazio":"lazio",
    "juventus fc":"juventus","atalanta bc":"atalanta",
    "paris saint germain":"paris sg","paris saint germain fc":"paris sg",
    "psg":"paris sg","as monaco":"monaco","olympique marseille":"marseille",
    "olympique lyonnais":"lyon","ogc nice":"nice","losc lille":"lille",
    "real madrid cf":"real madrid","atletico madrid":"ath madrid",
    "atletico de madrid":"ath madrid","athletico madrid":"ath madrid",
    "ath bilbao":"ath bilbao","athletic club":"ath bilbao","athletic bilbao":"ath bilbao",
    "real betis":"betis","real betis balompie":"betis","real sociedad":"sociedad",
    "rc celta":"celta","celta vigo":"celta","rayo vallecano":"vallecano",
    "fc barcelona":"barcelona","sevilla fc":"sevilla","valencia cf":"valencia",
    "bayern munchen":"bayern munich","bayern munich":"bayern munich",
    "fc bayern munchen":"bayern munich","borussia dortmund":"dortmund",
    "borussia monchengladbach":"mgladbach","monchengladbach":"mgladbach",
    "bayer 04 leverkusen":"leverkusen","bayer leverkusen":"leverkusen",
    "vfb stuttgart":"stuttgart","vfl wolfsburg":"wolfsburg",
    "eintracht frankfurt":"frankfurt","rb leipzig":"rb leipzig",
    "tsg 1899 hoffenheim":"hoffenheim","tsg hoffenheim":"hoffenheim",
    "fc koln":"cologne","fc schalke 04":"schalke 04",
    "ajax amsterdam":"ajax","psv eindhoven":"psv","feyenoord rotterdam":"feyenoord",
    "sporting cp":"sporting","sporting clube de portugal":"sporting",
    "sl benfica":"benfica","fc porto":"porto",
}

def canonical(value):
    import re, unicodedata
    txt=unicodedata.normalize("NFKD",str(value or "")).encode("ascii","ignore").decode().lower()
    txt=txt.replace("&"," and ")
    txt=re.sub(r"[^a-z0-9]+"," ",txt).strip()
    if txt in ALIASES:return ALIASES[txt]
    txt=re.sub(r"^(fc|afc) ","",txt)
    txt=re.sub(r" (fc|afc|sc)$","",txt).strip()
    return ALIASES.get(txt,txt)

def score(record):
    return "1" if record["hg"]>record["ag"] else "X" if record["hg"]==record["ag"] else "2"

class Form:
    def __init__(self):
        self.elo=defaultdict(lambda:1500.0)
        self.games=defaultdict(list)
        self.last={}
    def feature(self,h,a,day):
        # Never read the results of this fixture or any other fixture on its date.
        e=self.elo[h]-self.elo[a]+HOME_ADVANTAGE
        def summary(team,venue):
            games=self.games[team][-10:]
            venue_games=[g for g in self.games[team] if g["venue"]==venue][-6:]
            def avg(seq,key,prior,weight=3):
                return (sum(x[key] for x in seq)+prior*weight)/(len(seq)+weight)
            recent_points=avg(games,"points",1.25)
            scored=avg(games,"gf",1.35)
            conceded=avg(games,"ga",1.35)
            venue_for=avg(venue_games,"gf",1.35,2)
            venue_against=avg(venue_games,"ga",1.35,2)
            days=21 if team not in self.last else min(35,max(1,(date.fromisoformat(day)-date.fromisoformat(self.last[team])).days))
            return [len(games),recent_points,scored,conceded,venue_for,venue_against,days]
        h_features=summary(h,"H");a_features=summary(a,"A")
        return dict(elo_diff=e, h_elo=self.elo[h],a_elo=self.elo[a],
            **{f"h_{i}":v for i,v in enumerate(h_features)},
            **{f"a_{i}":v for i,v in enumerate(a_features)},
            **{f"d_{i}":h_features[i]-a_features[i] for i in range(1,6)})
    def baseline(self,h,a):
        win=1/(1+10**(-(self.elo[h]-self.elo[a]+HOME_ADVANTAGE)/400))
        return np.array([.745*win,.255,.745*(1-win)])
    def update(self,r):
        h,a=r["home"],r["away"];hg,ag=r["hg"],r["ag"]
        p=1/(1+10**(-(self.elo[h]-self.elo[a]+HOME_ADVANTAGE)/400))
        actual=1 if hg>ag else .5 if hg==ag else 0
        delta=23*(actual-p)
        self.elo[h]+=delta;self.elo[a]-=delta
        self.games[h].append(dict(venue="H",gf=hg,ga=ag,points=3 if hg>ag else 1 if hg==ag else 0))
        self.games[a].append(dict(venue="A",gf=ag,ga=hg,points=3 if ag>hg else 1 if ag==hg else 0))
        self.last[h]=self.last[a]=r["date"]

def build_history(records):
    state=Form();rows=[]
    ordered=sorted(records,key=lambda r:(r["date"],r["home"],r["away"]))
    for day,grp in groupby(ordered,key=lambda r:r["date"]):
        matches=list(grp)
        for r in matches:
            x=state.feature(r["home"],r["away"],day)
            rows.append(dict(date=day,season=int(r["season"]),home=r["home"],away=r["away"],
                outcome=score(r),elo_1=state.baseline(r["home"],r["away"])[0],
                elo_X=.255,elo_2=state.baseline(r["home"],r["away"])[2],**x))
        for r in matches:state.update(r)
    return pd.DataFrame(rows),state

def features_for_fixture(records,home,away,fixture_day):
    # Remove ALL games from the fixture date and later to avoid lookahead.
    previous=[r for r in records if r["date"]<fixture_day]
    _,state=build_history(previous)
    return state.feature(home,away,fixture_day),state.baseline(home,away),len(state.games[home]),len(state.games[away])

def probabilities(clf,frame,columns):
    raw=clf.predict_proba(frame[columns].astype(float))
    return np.column_stack([raw[:,list(clf[-1].classes_).index(k)] for k in OUTCOMES])

def soften(p,t):
    q=np.power(np.maximum(p,1e-12),1.0/t)
    return q/q.sum(axis=1,keepdims=True)

def evaluate(y,p):
    idx=np.array([OUTCOMES.index(z) for z in y])
    actual=np.eye(3)[idx]
    return dict(accuracy=float(np.mean(p.argmax(axis=1)==idx)),
        log_loss=float(-np.mean(np.log(np.maximum(p[np.arange(len(y)),idx],1e-12)))),
        brier=float(np.mean(np.sum((p-actual)**2,axis=1))))

def train_model(frame):
    # A calendar-season split: train 2020-23; tune 2024; test 2025 only ONCE.
    train=frame[frame.season<=2023].copy()
    val=frame[frame.season==2024].copy()
    test=frame[frame.season==2025].copy()
    if min(len(train),len(val),len(test))<80:
        raise ValueError(f"Need >=80 matches in each split. Have train={len(train)}, validation={len(val)}, test={len(test)}.")
    cols=[c for c in frame if c=="elo_diff" or c in ("h_elo","a_elo") or c.startswith(("h_","a_","d_"))]
    classifier=make_pipeline(StandardScaler(),LogisticRegression(C=.16,max_iter=1600))
    classifier.fit(train[cols],train.outcome)
    if set(classifier[-1].classes_)!=set(OUTCOMES):
        raise ValueError("All 1 / X / 2 outcomes must appear in training.")
    raw_val=probabilities(classifier,val,cols)
    candidates=[.8,1,1.2,1.4,1.7,2,2.4,3]
    temp=min(candidates,key=lambda t:evaluate(val.outcome.to_numpy(),soften(raw_val,t))["log_loss"])
    val_machine=evaluate(val.outcome.to_numpy(),soften(raw_val,temp))
    val_elo=np.asarray(val[["elo_1","elo_X","elo_2"]])
    # Select without consulting the untouched final test season.
    selected="trained" if val_machine["log_loss"]<evaluate(val.outcome.to_numpy(),val_elo)["log_loss"] else "elo"
    test_ml=soften(probabilities(classifier,test,cols),temp)
    test_elo=test[["elo_1","elo_X","elo_2"]].to_numpy()
    best=test_ml if selected=="trained" else test_elo
    actual=test.outcome.to_numpy()
    comparison=dict(trained=evaluate(actual,test_ml),elo=evaluate(actual,test_elo),selected=evaluate(actual,best))
    # A historical proxy: Elo-underestimated winners. Not bookmaker upsets.
    underdog=np.where(test_elo[:,0]<test_elo[:,2],0,2)
    baseline_underdog=test_elo[np.arange(len(test)),underdog]
    predicted_underdog=best[np.arange(len(test)),underdog]
    alerts=(predicted_underdog-baseline_underdog>=.07)&(predicted_underdog>=.27)
    did_win=np.array([OUTCOMES.index(x) for x in actual])==underdog
    alert_count=int(alerts.sum())
    upset=dict(alerts=alert_count,underdog_wins=int(did_win.sum()),
        captured=int((alerts&did_win).sum()),
        false_alarms=int((alerts&~did_win).sum()),
        hit_rate=(float((alerts&did_win).sum()/alert_count) if alert_count else None))
    holdout=test[["date","home","away","outcome"]].copy()
    holdout["forecast"]=np.asarray(OUTCOMES)[best.argmax(axis=1)]
    holdout["p_1"]=best[:,0];holdout["p_X"]=best[:,1];holdout["p_2"]=best[:,2]
    holdout["elo_1"]=test_elo[:,0];holdout["elo_X"]=test_elo[:,1];holdout["elo_2"]=test_elo[:,2]
    holdout["upset_alert"]=alerts
    # Final production model may use all HISTORICAL seasons, never the live future.
    classifier.fit(frame[cols],frame.outcome)
    return dict(model=classifier,columns=cols,temperature=temp,selected=selected,
        comparison=comparison,upset=upset,holdout=holdout,train_size=len(train),
        val_size=len(val),test_size=len(test))

def forecast(trained,records,home,away,fixture_day):
    if home==away:raise ValueError("Cannot forecast a team against itself")
    f,elo,nh,na=features_for_fixture(records,home,away,fixture_day)
    if nh<5 or na<5:
        return dict(available=False,reason=f"Insufficient historical data for these teams: {nh} / {na} matches")
    x=pd.DataFrame([f])
    p=soften(probabilities(trained["model"],x,trained["columns"]),trained["temperature"])[0]
    if trained["selected"]=="elo":p=elo
    under=0 if elo[0]<elo[2] else 2
    uplift=float(p[under]-elo[under])
    upset_flag=bool(uplift>=.07 and p[under]>=.27 and trained["selected"]=="trained")
    return dict(available=True,probabilities=dict(zip(OUTCOMES,map(float,p))),
        pick=OUTCOMES[int(np.argmax(p))],baseline=dict(zip(OUTCOMES,map(float,elo))),
        upset_flag=upset_flag,underdog=OUTCOMES[under],underdog_uplift=uplift,
        history_count_home=nh,history_count_away=na,engine=trained["selected"],
        coverage="Completed match results, Elo, home/away form, rest (NO current xG / pressing data).")
