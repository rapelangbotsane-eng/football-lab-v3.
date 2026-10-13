"""Dynamic-goal score model tests: coherence, no same-day leakage, validation-only."""
import unittest
from datetime import date,timedelta
import numpy as np
from dynamic_goal_engine import (
    outcome_from_goals,chronological_goal_forecasts,
    poisson_candidates,validated_goal_fold
)
from club_engine import build_history

def sample_games():
    teams=[f"club{k}" for k in range(12)]
    rows=[]
    for season in range(2020,2026):
        for i in range(120):
            day=(date(season,8,1)+timedelta(days=i)).isoformat()
            home=teams[i%12];away=teams[(i*7+3)%12]
            if away==home:away=teams[(i*7+4)%12]
            rows.append(dict(date=day,home=home,away=away,
                season=season,hg=(i+season)%4,ag=(2*i+season)%3))
    return rows

class DynamicGoalsTests(unittest.TestCase):
    def test_poisson_normalisation_and_home_advantage(self):
        for rho in (-.1,0,.1):
            p=outcome_from_goals(1.8,1.1,rho)
            self.assertAlmostEqual(float(p.sum()),1.,places=11)
            self.assertTrue(np.all(p>=0))
            self.assertGreater(p[0],p[2])
            self.assertGreater(p[1],.01)
    def test_future_and_same_day_results_dont_leak(self):
        rows=sample_games()
        base=chronological_goal_forecasts(rows)
        a=next(x for x in rows if x["season"]==2025)
        changed=[dict(x,hg=15,ag=0) if x==a else x for x in rows]
        after=chronological_goal_forecasts(changed)
        day=a["date"]
        for k in base:
            if k[0]<=day:
                self.assertTrue(np.allclose(base[k],after[k]))
    def test_goal_model_matches_elo_fixture_order_and_tunes_on_val(self):
        rows=sample_games()
        frame,_=build_history(rows)
        data=poisson_candidates(rows)
        self.assertEqual(data["keys"],list(zip(frame.date,frame.home,frame.away)))
        val=frame[frame.season==2024]
        test=frame[frame.season==2025]
        val_base=val[["elo_1","elo_X","elo_2"]].to_numpy()
        test_base=test[["elo_1","elo_X","elo_2"]].to_numpy()
        checked=validated_goal_fold(data,2024,2025,val_base,test_base)
        self.assertAlmostEqual(float(checked["test_probs"][0].sum()),1.)
        self.assertGreater(checked["test"]["log_loss"],0)
        self.assertIn(checked["config"]["blend"],(0,.25,.50,.75,1.))
        self.assertEqual(len(checked["test_probs"]),len(test))

if __name__=="__main__":unittest.main()
