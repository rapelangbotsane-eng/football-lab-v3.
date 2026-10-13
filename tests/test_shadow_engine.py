"""Tests for shadow forecasts frozen before kickoff; no rewriting or hindsight."""
from datetime import datetime,timezone,date,timedelta
import unittest
import numpy as np
from club_engine import Form
from shadow_engine import shadow_forecast_cards,fit_historical_elo_calibration,MODEL_VERSION
from prediction_tracker import (
    snapshot_prediction,settle_one,prospective_comparison,merge_archives
)

def historical():
    teams=[f"club{i}" for i in range(12)]
    matches=[]
    for year in range(2020,2026):
        for j in range(130):
            h=teams[j%12];a=teams[(j*7+3)%12]
            if h==a:a=teams[(j*7+4)%12]
            d=(date(year,8,1)+timedelta(days=j)).isoformat()
            matches.append(dict(date=d,season=year,home=h,away=a,
                hg=(j+year)%4,ag=(j*3+year)%3))
    return matches

def fixture_card(rows,fixture_id="777",dt="2026-10-19T17:00:00+00:00"):
    state=Form()
    for game in sorted(rows,key=lambda r:(r["date"],r["home"],r["away"])):
        if game["date"] < dt[:10]:state.update(game)
    h,a="club0","club1"
    p=state.baseline(h,a)
    return dict(id=fixture_id,league="Premier League · England",
        home="club0",away="club1",home_code=h,away_code=a,
        datetime=dt,snapshot="2026-10-13T09:00:00+00:00",
        pick=("1","X","2")[int(p.argmax())],
        engine="elo",probabilities=dict(zip(("1","X","2"),map(float,p))),
        baseline=dict(zip(("1","X","2"),map(float,p))),
        upset_alert=False)

class ProspectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history=historical()
        cls.calibrator=fit_historical_elo_calibration(cls.history)
        cls.now=datetime(2026,10,13,10,tzinfo=timezone.utc)
    def test_three_challengers_sum_to_one_and_do_not_change_main(self):
        card=fixture_card(self.history)
        enhanced=shadow_forecast_cards(self.history,[card],[],self.calibrator)[0]
        self.assertEqual(enhanced["model_version"],MODEL_VERSION)
        self.assertEqual(enhanced["probabilities"],card["probabilities"])
        self.assertEqual(set(enhanced["shadow_models"]),
            {"Fixed Elo","Calibrated Elo","Dynamic goals + Elo"})
        for p in enhanced["shadow_models"].values():
            self.assertAlmostEqual(sum(p.values()),1.)
        row=snapshot_prediction(enhanced,"PL",self.now)
        self.assertEqual(set(row["shadow_models"]),set(enhanced["shadow_models"]))
    def test_finished_fixture_on_same_day_does_not_leak(self):
        card=fixture_card(self.history)
        baseline=shadow_forecast_cards(self.history,[card],[],self.calibrator)[0]
        completed=dict(id="something_else",status="FINISHED",utcDate="2026-10-19T09:00:00Z",
            homeTeam={"shortName":"club0"},awayTeam={"shortName":"club1"},
            score={"fullTime":{"home":19,"away":0}})
        changed=shadow_forecast_cards(self.history,[card],[completed],self.calibrator)[0]
        self.assertEqual(baseline["shadow_models"],changed["shadow_models"])
    def test_no_retrospective_shadow_filling_or_overwriting(self):
        base=fixture_card(self.history)
        first=snapshot_prediction(base,"PL",self.now)
        later=shadow_forecast_cards(self.history,[base],[],self.calibrator)[0]
        later_snap=snapshot_prediction(later,"PL",self.now+timedelta(minutes=5))
        rows=merge_archives([first],[later_snap])
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0].get("shadow_models"))
    def test_paired_scoring_only_after_finished_result(self):
        cards=[fixture_card(self.history,str(j),
            f"2026-10-{19+j:02d}T17:00:00+00:00") for j in range(3)]
        annotated=shadow_forecast_cards(self.history,cards,[],self.calibrator)
        saved=[snapshot_prediction(c,"PL",self.now) for c in annotated]
        self.assertEqual(prospective_comparison(saved)["matches"],0)
        final=dict(id="0",status="FINISHED",
            score={"fullTime":{"home":2,"away":1}})
        saved[0]=settle_one(saved[0],final,as_of=self.now+timedelta(days=10))
        report=prospective_comparison(saved)
        self.assertEqual(report["matches"],1)
        self.assertEqual(len(report["models"]),4)
        self.assertAlmostEqual(report["models"][1]["log_loss_delta_vs_fixed_elo"],0.)
        self.assertFalse(report["trustworthy_sample"])
    def test_corrupt_shadow_probabilities_rejected(self):
        card=fixture_card(self.history)
        card["shadow_models"]={"Fixed Elo":{"1":.8,"X":.8,"2":-.6}}
        with self.assertRaises(ValueError):snapshot_prediction(card,"PL",self.now)

if __name__=="__main__":unittest.main()
