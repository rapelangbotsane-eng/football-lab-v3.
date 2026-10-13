"""Dynamic, time-weighted attack/defence forecast from historical goals.

Research challenger only. No match-day leakage. This is an online/shrunk
Poisson attack-vs-defence model with optional Dixon-Coles-style low-score
correction (not a full jointly fitted Dixon-Coles MLE).
"""
from collections import defaultdict
from datetime import date
from itertools import groupby
import math
import numpy as np
from club_engine import OUTCOMES, soften, evaluate

GOAL_MAX=11
HALF_LIVES=(120,240,400)
SHRINKS=(4.,10.)
RHOS=(-.10,0.,.10)
TEMPS=(.85,1.,1.15,1.35)
WEIGHTS=(0.,.25,.50,.75,1.)
GOAL_PRIORS=(1.48,1.18)

def _poisson(lam):
    lam=max(.12,min(4.8,float(lam)))
    probs=[math.exp(-lam)]
    for k in range(1,GOAL_MAX+1):
        probs.append(probs[-1]*lam/k)
    return np.asarray(probs,dtype=float)

def outcome_from_goals(lam_h,lam_a,rho=0.):
    """Normalise all cells after the low-score correlation adjustment."""
    h,a=_poisson(lam_h),_poisson(lam_a)
    matrix=np.outer(h,a)
    if rho:
        # Standard Dixon-Coles tau factors for score pairs 0-0,0-1,1-0,1-1.
        matrix[0,0]*=max(.05,1-lam_h*lam_a*rho)
        matrix[0,1]*=max(.05,1+lam_h*rho)
        matrix[1,0]*=max(.05,1+lam_a*rho)
        matrix[1,1]*=max(.05,1-rho)
    matrix/=matrix.sum()
    p=np.array([np.tril(matrix,-1).sum(),np.trace(matrix),np.triu(matrix,1).sum()])
    # rows: home goals, columns: away goals; home win is below diagonal.
    return p/p.sum()

class DynamicGoals:
    def __init__(self,half_life=240,shrink=4):
        if half_life<=0 or shrink<=0:raise ValueError("half-life and shrinkage must be positive")
        self.half_life=half_life
        self.shrink=shrink
        self.history=defaultdict(list)
        self.league=[]
    def _weighted(self,history,day,field,venue=None):
        seq=(r for r in history if venue is None or r["venue"]==venue)
        weighted=[]
        for r in seq:
            age=(date.fromisoformat(day)-date.fromisoformat(r["date"])).days
            if age<=0:continue
            weight=2.**(-age/self.half_life)
            weighted.append((weight,r[field]))
        return sum(w for w,v in weighted),sum(w*v for w,v in weighted)
    def rates(self,home,away,day):
        # Global home/away means are derived only from the past. Home advantage
        # is dynamic, not hardcoded to a constant draw prior.
        n,s_h=self._weighted(self.league,day,"hg")
        _,s_a=self._weighted(self.league,day,"ag")
        mean_h=(s_h+80*GOAL_PRIORS[0])/(n+80)
        mean_a=(s_a+80*GOAL_PRIORS[1])/(n+80)
        def rate(team,venue,field,global_prior):
            n_all,s_all=self._weighted(self.history[team],day,field)
            n_venue,s_venue=self._weighted(self.history[team],day,field,venue)
            # Blend venue-specific and general form; shrink unstable/new clubs.
            base_all=(s_all+global_prior*self.shrink)/(n_all+self.shrink)
            base_venue=(s_venue+global_prior*self.shrink)/(n_venue+self.shrink)
            return .65*base_venue+.35*base_all
        h_for=rate(home,"H","gf",mean_h)
        h_against=rate(home,"H","ga",mean_a)
        a_for=rate(away,"A","gf",mean_a)
        a_against=rate(away,"A","ga",mean_h)
        home_rate=mean_h*(h_for/max(.4,mean_h))*(a_against/max(.4,mean_h))
        away_rate=mean_a*(a_for/max(.4,mean_a))*(h_against/max(.4,mean_a))
        # Shrink extreme rates; do not imply impossibly decisive matches.
        return float(np.clip(home_rate,.20,3.6)),float(np.clip(away_rate,.20,3.6))
    def update(self,r):
        entry_h=dict(date=r["date"],venue="H",gf=r["hg"],ga=r["ag"])
        entry_a=dict(date=r["date"],venue="A",gf=r["ag"],ga=r["hg"])
        self.history[r["home"]].append(entry_h)
        self.history[r["away"]].append(entry_a)
        self.league.append(dict(date=r["date"],hg=r["hg"],ag=r["ag"]))

def chronological_goal_forecasts(records,half_life=240,shrink=4,rho=0.):
    state=DynamicGoals(half_life,shrink)
    output={}
    for day,group in groupby(sorted(records,key=lambda r:(r["date"],r["home"],r["away"])),
                            key=lambda r:r["date"]):
        games=list(group)
        for r in games:
            lamh,lama=state.rates(r["home"],r["away"],day)
            output[(day,r["home"],r["away"])]=outcome_from_goals(lamh,lama,rho)
        for r in games:state.update(r)
    return output

def poisson_candidates(records):
    """Precompute six chronological time-decay/form settings.

    For each half-life and shrink pair, recompute score-grid outcomes for each
    rho candidate. Runs once per league research job, not per phone tap.
    """
    ordered=sorted(records,key=lambda r:(r["date"],r["home"],r["away"]))
    keys=[(r["date"],r["home"],r["away"]) for r in ordered]
    seasons=np.asarray([int(r["season"]) for r in ordered])
    y=np.asarray(["1" if r["hg"]>r["ag"] else "X" if r["hg"]==r["ag"] else "2" for r in ordered])
    result={}
    for half in HALF_LIVES:
        for shrink in SHRINKS:
            # Reuse the same online state for all three low-score adjustments.
            state=DynamicGoals(half,shrink)
            predictions={rho:[] for rho in RHOS}
            for day,group in groupby(ordered,key=lambda r:r["date"]):
                games=list(group)
                for r in games:
                    lamh,lama=state.rates(r["home"],r["away"],day)
                    for rho in RHOS:
                        predictions[rho].append(outcome_from_goals(lamh,lama,rho))
                for r in games:state.update(r)
            for rho,arrays in predictions.items():
                result[(half,shrink,rho)]=np.asarray(arrays,dtype=float)
    return dict(keys=keys,season=seasons,y=y,candidates=result)

def optimise_goals_on_validation(data,validation_mask,elo_val):
    """Choose decay, shrink, rho, temperature and optional Elo blend on val ONLY."""
    yval=data["y"][validation_mask]
    best=None
    for cfg,all_p in data["candidates"].items():
        p=all_p[validation_mask]
        for temperature in TEMPS:
            calibrated=soften(p,temperature)
            for blend in WEIGHTS:
                mixed=blend*calibrated+(1-blend)*elo_val
                ll=evaluate(yval,mixed)["log_loss"]
                proposal=(ll,cfg,temperature,blend)
                if best is None or proposal<best:best=proposal
    return dict(log_loss=float(best[0]),half_life=best[1][0],
        shrink=best[1][1],rho=best[1][2],temperature=best[2],blend=best[3])

def validated_goal_fold(data,validation_year,test_year,elo_val,elo_test):
    val=data["season"]==validation_year
    test=data["season"]==test_year
    if int(val.sum())!=len(elo_val) or int(test.sum())!=len(elo_test):
        raise ValueError("Goal forecasts and Elo baseline must cover identical fixtures in date order")
    config=optimise_goals_on_validation(data,val,elo_val)
    base=data["candidates"][(config["half_life"],config["shrink"],config["rho"])]
    def combine(mask,elo):
        return config["blend"]*soften(base[mask],config["temperature"])+(1-config["blend"])*elo
    vp,tp=combine(val,elo_val),combine(test,elo_test)
    return dict(config=config,validation=evaluate(data["y"][val],vp),
        test=evaluate(data["y"][test],tp),test_probs=tp)
