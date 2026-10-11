import unittest
from datetime import date,timedelta
from understat_data import parse_understat_league,aligned_metrics,UnderstatUnavailable
from advanced_club_experiment import make_rolling_advanced_features, run_xg_experiment

class UnderstatAdapterTest(unittest.TestCase):
    def test_parse_and_strict_score_match(self):
        d="2025-08-16 15:00:00"
        stats=lambda side:dict(date=d,h_a=side,xG="1.4",xGA="0.9",npxG="1.1",
            deep=8,xpts="1.6",ppda={"att":84,"def":7})
        data={
          "teams":{"10":{"id":"10","title":"Manchester City","history":[stats("h")]},
                   "20":{"id":"20","title":"Everton","history":[stats("a")]}},
          "dates":[{"datetime":d,"isResult":True,
                    "h":{"id":"10","title":"Manchester City"},
                    "a":{"id":"20","title":"Everton"},
                    "goals":{"h":"2","a":"1"}}]
        }
        parsed=parse_understat_league(data,2025)
        self.assertEqual(len(parsed),1)
        self.assertEqual(parsed[0]["home"],"man city")
        self.assertAlmostEqual(parsed[0]["home_metrics"]["ppda"],12)
        results=[dict(date="2025-08-16",season=2025,home="man city",away="everton",hg=2,ag=1)]
        matched,unmatched=aligned_metrics(results,parsed)
        self.assertEqual(len(matched),1)
        self.assertEqual(len(unmatched),0)
        wrong=[dict(results[0],hg=3)]
        matched,unmatched=aligned_metrics(wrong,parsed)
        self.assertEqual(len(matched),0)
        self.assertEqual(len(unmatched),1)

    def test_no_same_day_advanced_feature_leakage(self):
        m=lambda x:dict(xgf=x,xga=x,npxgf=x,ppda=10,deep=5,xpts=1)
        games=[
            dict(date="2025-01-01",season=2024,home="arsenal",away="chelsea",hg=1,ag=0),
            dict(date="2025-01-01",season=2024,home="man city",away="everton",hg=2,ag=2),
            dict(date="2025-01-06",season=2024,home="arsenal",away="everton",hg=1,ag=1)]
        matched={(g["date"],g["home"],g["away"]):dict(home_metrics=m(200),away_metrics=m(10)) for g in games}
        f=make_rolling_advanced_features(games,matched)
        self.assertEqual(f[("2025-01-01","arsenal","chelsea")]["hx_samples"],0)
        self.assertEqual(f[("2025-01-01","man city","everton")]["hx_samples"],0)
        self.assertEqual(f[("2025-01-06","arsenal","everton")]["hx_samples"],1)
        self.assertLess(f[("2025-01-06","arsenal","everton")]["hx_xgf"],100)

class XgAblationTest(unittest.TestCase):
    def test_same_matches_and_four_models(self):
        games=[];xm=[]
        for season in range(2020,2026):
            for i in range(160):
                h=f"team{i%12}"
                a=f"team{(i*5+1)%12}"
                if a==h:a=f"team{(i*5+2)%12}"
                day=(date(season,8,1)+timedelta(days=i)).isoformat()
                hg=(i+season)%4;ag=(i*3+season)%3
                g=dict(date=day,season=season,home=h,away=a,hg=hg,ag=ag)
                games.append(g)
                m=lambda ishome:dict(xgf=1.2+(hg if ishome else ag)*.3,
                    xga=1.0,npxgf=1.0,ppda=12.0,deep=7,xpts=1.3)
                xm.append(dict(**g,home_metrics=m(True),away_metrics=m(False)))
        report=run_xg_experiment(games,xm)
        self.assertEqual(len(report["table"]),4)
        self.assertIn(report["selected"],set(report["table"]["model"]))
        self.assertEqual(report["retrospective_matches"],160)
        self.assertEqual(report["source_matched"],len(games))
        self.assertTrue((report["table"]["test_log_loss"]>0).all())

if __name__=="__main__":
    unittest.main()
