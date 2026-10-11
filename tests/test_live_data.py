import unittest
from datetime import datetime, timezone
from live_data import forecast_fixture, next_fixtures

def past_match(match_id, home_id, away_id, home_goals, away_goals, month, day):
    return dict(id=match_id, status="FINISHED", utcDate=f"2026-{month:02d}-{day:02d}T18:00:00Z",
        homeTeam={"id": home_id}, awayTeam={"id": away_id},
        score={"fullTime":{"home":home_goals,"away":away_goals}})

class LiveForecastTest(unittest.TestCase):
    def setUp(self):
        self.fixture=dict(id=50,status="SCHEDULED", utcDate="2026-10-20T19:00:00Z",
            homeTeam={"id":1},awayTeam={"id":2})
        self.matches=[
            *[past_match(i,1,100+i,2,1,9,i+1) for i in range(1,8)],
            *[past_match(i+10,200+i,2,1,1,9,i+1) for i in range(1,8)],
            self.fixture
        ]

    def test_probabilities_sum_to_one(self):
        result=forecast_fixture(self.fixture,self.matches)
        self.assertFalse(result["insufficient"])
        self.assertAlmostEqual(result["p_home"]+result["p_draw"]+result["p_away"],1.0,places=8)
        self.assertIn(result["pick"],("1","X","2"))

    def test_minimum_history_requirement(self):
        result=forecast_fixture(self.fixture,self.matches[:3])
        self.assertTrue(result["insufficient"])

    def test_future_results_cannot_leak(self):
        initial=forecast_fixture(self.fixture,self.matches)
        future=past_match(200,1,2,0,7,11,1)
        changed=forecast_fixture(self.fixture,self.matches+[future])
        self.assertAlmostEqual(initial["p_home"],changed["p_home"],places=10)

    def test_only_future_scheduled_matches(self):
        upcoming=next_fixtures(self.matches,now=datetime(2026,10,11,tzinfo=timezone.utc),horizon_days=21)
        self.assertEqual([m["id"] for m in upcoming],[50])

if __name__ == "__main__":
    unittest.main()
