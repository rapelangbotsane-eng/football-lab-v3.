"""Offline checks for rolling-origin training: fixed seasons, no lookahead."""
from datetime import date,timedelta
import unittest
import numpy as np
from club_engine import build_history
from training_lab import rolling_benchmark,feature_groups,_outcome_accuracy_at_threshold

def fake_results():
    clubs=[f"club{i}" for i in range(12)]
    out=[]
    for season in range(2020,2026):
        for i in range(150):
            day=(date(season,8,1)+timedelta(days=i)).isoformat()
            home=clubs[i%12]
            away=clubs[(i*7+3)%12]
            if away==home:away=clubs[(i*7+4)%12]
            hg=(i*5+season)%4
            ag=(i*7+season+1)%3
            out.append(dict(date=day,season=season,
                home=home,away=away,hg=hg,ag=ag))
    return out

class TrainingExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=fake_results()
        cls.study=rolling_benchmark(cls.fixture)
    def test_rolling_origin_split(self):
        self.assertEqual(self.study["number_of_folds"],3)
        for fold in self.study["folds"]:
            year=fold["test_season"]
            self.assertEqual(fold["training_through"],year-2)
            self.assertEqual(fold["validation_season"],year-1)
            self.assertEqual(fold["test_matches"],150)
            self.assertIn(fold["selected_by_validation"],
                {row["model"] for row in fold["competitors"]})
            self.assertEqual({row["model"] for row in fold["competitors"]},
                {"Fixed Elo","Calibrated Elo","Results + Elo logistic","Results + Elo boosted"})
            self.assertTrue(all(row["test"]["log_loss"]>0 for row in fold["competitors"]))
    def test_model_winner_uses_validation_only(self):
        for fold in self.study["folds"]:
            best=min(fold["competitors"],key=lambda r:(r["validation"]["log_loss"],r["model"]))
            self.assertEqual(best["model"],fold["selected_by_validation"])
            self.assertEqual(best["test"],fold["selected_test"])
    def test_probabilities_from_same_day_results_never_leak(self):
        frame,_=build_history(self.fixture)
        other=list(self.fixture)
        sample=next(x for x in self.fixture if x["season"]==2025)
        forged=dict(sample,hg=13,ag=0)
        other=[forged if x is sample else x for x in self.fixture]
        changed,_=build_history(other)
        key=(frame["season"]==2025)&(frame["date"]==sample["date"])
        cols=feature_groups(frame)["Results + Elo logistic"]
        self.assertTrue(np.allclose(frame.loc[key,cols],changed.loc[key,cols]))
        # Past performance may influence LATER dates; feature pipeline is chronological.
    def test_strong_picks_no_forced_recommendation(self):
        pred=np.array([[.45,.30,.25],[.59,.20,.21],[.62,.19,.19]])
        scored=_outcome_accuracy_at_threshold(np.array(["1","2","1"]),pred)
        self.assertEqual(scored["matches"],1)
        self.assertEqual(scored["wins"],1)
        empty=_outcome_accuracy_at_threshold(np.array(["1","2"]),pred[:2])
        self.assertEqual(empty["matches"],0)
        self.assertIsNone(empty["hit_rate"])

if __name__=="__main__":unittest.main()
