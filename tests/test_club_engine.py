import unittest
import numpy as np
from datetime import date, timedelta
from club_engine import canonical, build_history, train_model, forecast, features_for_fixture, upset_diagnostics
from club_data import match_api_name, fixture_teams, current_season_results

def fake_results():
    results=[]
    for y in range(2020,2026):
        for i in range(140):
            h=f"team{i%20}"
            a=f"team{(i*7+3)%20}"
            if h==a:a=f"team{(i*7+4)%20}"
            results.append(dict(date=(date(y,8,1)+timedelta(days=i)).isoformat(),season=y,
                home=h,away=a,hg=(i*7+y)%4,ag=(i*5+y*2)%4))
    return results

class ClubEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=fake_results()
        cls.frame,_=build_history(cls.rows)
        cls.model=train_model(cls.frame)

    def test_names_known_aliases(self):
        self.assertEqual(canonical("Manchester City FC"),"man city")
        self.assertEqual(canonical("Nottingham Forest"),"nottm forest")
        self.assertEqual(canonical("Paris Saint-Germain"),"paris sg")
        self.assertIsNone(match_api_name({"shortName":"Imaginary United"},{"man city","everton"}))

    def test_split_and_probability_finite(self):
        self.assertEqual(self.model["test_size"],140)
        self.assertGreater(self.model["train_size"],400)
        for method in ("trained","elo","selected"):
            for metric in self.model["comparison"][method].values():
                self.assertTrue(isinstance(metric,float))
                self.assertTrue(metric==metric)
        self.assertIn(self.model["selected"],("trained","elo"))

    def test_final_predictions_sum(self):
        p=forecast(self.model,self.rows,"team1","team2","2026-10-12")
        self.assertTrue(p["available"])
        self.assertAlmostEqual(sum(p["probabilities"].values()),1.,places=9)

    def test_future_match_no_leakage(self):
        a=features_for_fixture(self.rows,"team1","team2","2026-10-12")[0]
        extra=dict(date="2026-10-12",season=2026,home="team1",away="team2",hg=9,ag=0)
        b=features_for_fixture(self.rows+[extra],"team1","team2","2026-10-12")[0]
        self.assertEqual(a,b)

    def test_ml_challenger_alerts_even_when_selected_elo(self):
        # The prior bug compared selected Elo against itself and always yielded zero.
        # These fixed test inputs must identify a meaningful ML-versus-Elo uplift.
        elo=np.array([[.65,.25,.10],[.22,.25,.53],[.40,.25,.35]])
        challenger=np.array([[.39,.23,.38],[.41,.20,.39],[.33,.27,.40]])
        actual=np.array(["2","1","X"])
        result=upset_diagnostics(challenger,elo,actual)
        self.assertEqual(result["alerts"],2)
        self.assertEqual(result["captured"],2)
        self.assertEqual(result["false_alarms"],0)
        self.assertEqual(result["underdog_wins"],2)
        self.assertAlmostEqual(result["hit_rate"],1.)
        self.assertAlmostEqual(result["recall"],1.)

    def test_upset_report_is_generated_for_holdout(self):
        self.assertEqual(self.model["upset"]["underdog_wins"],int(self.model["holdout"]["elo_underdog_won"].sum()))
        self.assertEqual(self.model["upset"]["alerts"],int(self.model["holdout"]["ml_challenger_alert"].sum()))
        self.assertEqual(self.model["upset"]["captured"]+self.model["upset"]["false_alarms"],self.model["upset"]["alerts"])
        self.assertIn("ml_upset_uplift",self.model["holdout"].columns)
        self.assertIn("ml_underdog_probability",self.model["holdout"].columns)

    def test_api_mapping_does_not_guess(self):
        hist=[dict(home="man city",away="everton",date="2025-10-10",season=2025,hg=1,ag=1)]
        fixture=dict(homeTeam={"shortName":"Manchester City"},awayTeam={"shortName":"Everton"})
        self.assertEqual(fixture_teams(fixture,hist),("man city","everton"))
        m=dict(fixture,status="FINISHED",utcDate="2026-10-01T17:00:00Z",score={"fullTime":{"home":2,"away":1}})
        past,unknown=current_season_results([m],"2026-10-10",hist)
        self.assertEqual(len(past),1)
        self.assertFalse(unknown)
        future,_=current_season_results([m],"2026-09-20",hist)
        self.assertEqual(len(future),0)

if __name__=="__main__":unittest.main()
