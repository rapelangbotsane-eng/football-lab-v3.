"""Offline immutable tracking, accuracy and result reconciliation tests."""
import json
import os
import tempfile
import unittest
from datetime import datetime,timezone,timedelta
from pathlib import Path

from prediction_tracker import (
    snapshot_prediction,archive_local,load_local,reconcile_local,settle_one,
    settle_with_feed,merge_archives,tracking_summary,weekly_stats,monthly_stats,
    save_archive_local,to_csv
)

BASE=datetime(2026,10,12,9,tzinfo=timezone.utc)
KICK=(BASE+timedelta(days=3)).isoformat()
def card(fixture_id="301",kickoff=KICK,prob=.64,upset=True):
    p={"1":prob,"X":.20,"2":.80-prob}
    return dict(id=fixture_id,league="Premier League · England",
        home="Arsenal",away="Everton",datetime=kickoff,
        snapshot=(BASE-timedelta(minutes=20)).isoformat(),
        probabilities=p,pick="1",engine="elo",upset_alert=upset,
        underdog="2",underdog_team="Everton",upset_chance=.31,
        upset_uplift=.08)
def finished(fixture_id="301",hg=2,ag=1):
    return dict(id=fixture_id,status="FINISHED",
        score={"fullTime":{"home":hg,"away":ag}})

class PredictionTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"tracker.sqlite3"
    def tearDown(self):
        self.tmp.cleanup()
    def test_before_kickoff_only_and_proba(self):
        snap=snapshot_prediction(card(),"PL",BASE)
        self.assertTrue(snap["strong_60"])
        self.assertEqual(snap["result"],None)
        with self.assertRaises(ValueError):
            snapshot_prediction(card(),"PL",BASE+timedelta(days=4))
        with self.assertRaises(ValueError):
            snapshot_prediction(card(prob=.1),"PL",BASE)
    def test_first_snapshot_is_immutable_after_rescan(self):
        first=archive_local([card()],"PL",BASE,path=self.path)
        self.assertEqual(first["new"],1)
        second=archive_local([card(prob=.69)],"PL",BASE+timedelta(hours=1),path=self.path)
        self.assertEqual(second["new"],0)
        rows=load_local(self.path)
        self.assertEqual(len(rows),1)
        self.assertAlmostEqual(rows[0]["pick_probability"],.64)
        self.assertEqual(rows[0]["archived_at"],BASE.isoformat())
    def test_only_finished_matching_fixture_settles(self):
        row=snapshot_prediction(card(),"PL",BASE)
        after=BASE+timedelta(days=4)
        self.assertIsNone(settle_one(row,finished("999"),as_of=after)["result"])
        self.assertIsNone(settle_one(row,dict(finished(),status="SCHEDULED"),as_of=after)["result"])
        self.assertIsNone(settle_one(row,finished(),as_of=BASE)["result"])
        final=settle_one(row,finished(),as_of=after)
        self.assertTrue(final["result"]["correct"])
        self.assertFalse(final["result"]["upset_success"])
        self.assertIsNone(row["result"])
        self.assertEqual(final["result"]["outcome"],"1")
    def test_archive_and_score_accuracy(self):
        archive_local([card()],"PL",BASE,path=self.path)
        archive_local([card("302",prob=.69)],"PL",BASE,path=self.path)
        after=BASE+timedelta(days=4)
        count=reconcile_local("PL",[finished(),finished("302",hg=1,ag=2)],
            as_of=after,path=self.path)
        self.assertEqual(count,2)
        summary=tracking_summary(load_local(self.path))
        self.assertEqual(summary["total"],2)
        self.assertEqual(summary["all"]["resolved"],2)
        self.assertAlmostEqual(summary["all"]["accuracy"],.5)
        self.assertAlmostEqual(summary["strong"]["accuracy"],.5)
        self.assertAlmostEqual(summary["upset"]["accuracy"],.5)
        self.assertEqual(len(weekly_stats(load_local(self.path))),1)
        self.assertEqual(len(monthly_stats(load_local(self.path))),1)
        self.assertIn("Everton",to_csv(load_local(self.path)))
    def test_cloud_backup_does_not_replace_earlier_prediction(self):
        original=snapshot_prediction(card(),"PL",BASE)
        later=snapshot_prediction(card(prob=.69),"PL",BASE+timedelta(minutes=5))
        saved=merge_archives([original],[later])
        self.assertEqual(len(saved),1)
        self.assertAlmostEqual(saved[0]["pick_probability"],.64)
        save_archive_local([original],path=self.path)
        save_archive_local([later],path=self.path)
        self.assertAlmostEqual(load_local(self.path)[0]["pick_probability"],.64)
    def test_cloud_result_can_merge_without_overriding_forecast(self):
        original=snapshot_prediction(card(),"PL",BASE)
        graded=settle_one(original,finished(),as_of=BASE+timedelta(days=4))
        save_archive_local([original],path=self.path)
        save_archive_local([graded],path=self.path)
        self.assertIsNotNone(load_local(self.path)[0]["result"])
        self.assertEqual(load_local(self.path)[0]["pick_probability"],.64)
    def test_future_scoring_refused_on_wrong_league(self):
        one=snapshot_prediction(card(),"PL",BASE)
        rows,n=settle_with_feed([one],"PD",[finished()],as_of=BASE+timedelta(days=4))
        self.assertEqual(n,0)
        self.assertIsNone(rows[0]["result"])

if __name__=="__main__":unittest.main()
