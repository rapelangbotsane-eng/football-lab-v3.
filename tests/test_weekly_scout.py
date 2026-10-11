"""Offline regression tests for the mobile weekly suggestions."""
import unittest
from datetime import datetime,timezone,date,timedelta
import numpy as np
from weekly_scout import forecast_week,strong_predictions,upset_predictions
from club_engine import forecast

class FakeClassifier:
    classes_=np.array(["1","2","X"])
    def __getitem__(self,index):
        if index==-1:return self
        raise IndexError(index)
    def predict_proba(self,x):
        return np.tile(np.array([.62,.17,.21]),(len(x),1))

def sample_history():
    names=["man city","arsenal","everton","chelsea"]
    results=[]
    for i in range(35):
        h=names[i%4];a=names[(i%4+1)%4]
        results.append(dict(date=(date(2025,1,1)+timedelta(days=i*2)).isoformat(),
            season=2024,home=h,away=a,hg=2 if i%3 else 0,ag=i%3))
    return results

def fixture(when,home="Manchester City",away="Arsenal",status="SCHEDULED",id=1):
    return dict(id=id,utcDate=when,status=status,homeTeam={"id":11,"shortName":home},
        awayTeam={"id":12,"shortName":away},score={"fullTime":{}})

class ScoutTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,11,12,tzinfo=timezone.utc)
        self.history=sample_history()
        self.trained=dict(model=FakeClassifier(),columns=["elo_diff"],temperature=1.,selected="trained")
        self.game=fixture("2026-10-13T15:00:00Z")
        self.snapshot={"matches":[self.game],"fetched":"2026-10-11T12:00:00+00:00"}
    def test_weekly_estimates_match_full_forecast(self):
        items,stats=forecast_week(self.history,self.trained,self.snapshot,"Premier League",
            now=self.now,horizon_days=7)
        self.assertEqual(len(items),1)
        self.assertEqual(stats["upcoming"],1)
        independent=forecast(self.trained,self.history,"man city","arsenal","2026-10-13")
        for label in ("1","X","2"):
            self.assertAlmostEqual(items[0]["probabilities"][label],independent["probabilities"][label])
        self.assertAlmostEqual(sum(items[0]["probabilities"].values()),1.0)
    def test_same_day_completed_result_never_leaks_into_features(self):
        normal,_=forecast_week(self.history,self.trained,self.snapshot,"PL",now=self.now)
        completed=fixture("2026-10-13T08:00:00Z",home="Arsenal",away="Manchester City",status="FINISHED",id=2)
        completed["score"]={"fullTime":{"home":9,"away":0}}
        added=dict(self.snapshot,matches=[self.game,completed])
        test,_=forecast_week(self.history,self.trained,added,"PL",now=self.now)
        self.assertEqual(len(test),1)
        self.assertEqual(normal[0]["probabilities"],test[0]["probabilities"])
    def test_incomplete_mapping_is_excluded(self):
        unknown=fixture("2026-10-14T18:00:00Z",home="Fake Neverland",away="Arsenal",id=3)
        snapshot=dict(self.snapshot,matches=[unknown])
        items,stats=forecast_week(self.history,self.trained,snapshot,"PL",now=self.now)
        self.assertEqual(items,[])
        self.assertEqual(stats["unmapped"],1)
    def test_week_and_threshold(self):
        cards=[
            dict(id="a",datetime="2026-10-13",league="PL",probability=.64,upset_alert=False,upset_uplift=0,upset_chance=0),
            dict(id="b",datetime="2026-10-12",league="PL",probability=.59,upset_alert=True,upset_uplift=.1,upset_chance=.3),
            dict(id="c",datetime="2026-10-14",league="SA",probability=.72,upset_alert=True,upset_uplift=.08,upset_chance=.31),
        ]
        self.assertEqual([x["id"] for x in strong_predictions(cards,.60)],["c","a"])
        self.assertEqual([x["id"] for x in strong_predictions(cards,.70)],["c"])
        self.assertEqual([x["id"] for x in upset_predictions(cards)],["b","c"])
    def test_fixture_outside_week_excluded(self):
        later=fixture("2026-11-11T16:00:00Z",id=4)
        snap=dict(self.snapshot,matches=[later])
        items,stats=forecast_week(self.history,self.trained,snap,"PL",now=self.now,horizon_days=7)
        self.assertEqual(stats["upcoming"],0)
        self.assertFalse(items)

if __name__=="__main__":unittest.main()
