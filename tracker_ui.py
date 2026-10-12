"""Football Lab Prediction Tracker — immutable pre-kickoff forecast records."""
from datetime import datetime,timezone
import json
import pandas as pd
import streamlit as st
from live_data import LEAGUES
from club_ui import live_matches
from prediction_tracker import (
    load_local,reconcile_local,merge_archives,tracking_summary,
    weekly_stats,monthly_stats,to_csv,settle_with_feed,save_archive_local,
)
from tracker_cloud import fetch_cloud_archive
from weekly_scout import SA_TIME

@st.cache_data(ttl=180,show_spinner=False)
def cloud_archive():
    return fetch_cloud_archive()

def _pct(v):
    return f"{v:.1%}" if v is not None else "—"

def _summary(rows):
    metrics=tracking_summary(rows)
    st.subheader("Prediction performance")
    a,b=st.columns(2)
    a.metric("Predictions recorded",metrics["total"])
    b.metric("Matches settled",metrics["all"]["resolved"])
    c,d=st.columns(2)
    c.metric("Main 1X2 accuracy",_pct(metrics["all"]["accuracy"]))
    d.metric("Still awaiting results",metrics["pending"])
    st.divider()
    st.subheader("Your shortlist categories")
    e,f=st.columns(2)
    e.metric("60%+ accuracy",_pct(metrics["strong"]["accuracy"]),
        help=f'{metrics["strong"]["correct"]}/{metrics["strong"]["resolved"]} finished 60%+ predictions correct')
    f.metric("Upset alert hit rate",_pct(metrics["upset"]["accuracy"]),
        help=f'{metrics["upset"]["correct"]}/{metrics["upset"]["resolved"]} experimental alerts ended in underdog wins')
    st.caption(f'60%+ results: {metrics["strong"]["correct"]}/{metrics["strong"]["resolved"]} settled. '
        f'Upset signal wins: {metrics["upset"]["correct"]}/{metrics["upset"]["resolved"]} settled. '
        "No samples means insufficient history, not 0% accuracy.")

def _table(rows):
    data=[]
    for r in rows:
        result=r.get("result") or {}
        kick=datetime.fromisoformat(r["kickoff"]).astimezone(SA_TIME)
        data.append(dict(
            kickoff=kick.strftime("%d %b %Y %H:%M"),
            match=f'{r["home"]} vs {r["away"]}',league=r["league"],
            pick=r["pick"],probability=r["pick_probability"],strong_60=bool(r.get("strong_60")),
            upset_alert=bool(r.get("upset_alert")),final_score=(
                f'{result["home_goals"]}–{result["away_goals"]}' if result else "Pending"),
            outcome=result.get("outcome","—"),
            correct=("Yes" if result.get("correct") else "No") if result else "Pending",
            archived_at=r["archived_at"]))
    return pd.DataFrame(data)

def render(token=None):
    st.subheader("Prediction Tracker")
    st.write("Proof over promises. Every saved estimate is time-stamped **before kickoff**, then compared with the real final score. Rebuilding the weekly brief never rewrites earlier probabilities.")
    local=load_local()
    remote,error=cloud_archive()
    rows=merge_archives(local,remote)
    if remote:
        # Protect the earliest original forecast while caching published cloud results.
        save_archive_local(remote)
    if error:st.caption("Cloud archive temporarily unavailable. Local forecast history, if any, is still accessible.")
    now=datetime.now(SA_TIME)
    st.caption(f"Last opened {now:%d %b %Y at %H:%M} SAST · Local recorded: {len(local)} · Cloud recorded: {len(remote)}")
    with st.expander("Tracking history and cloud backup",expanded=False):
        st.info("Local history is stored on Streamlit's server, which can restart or lose files. For durable daily tracking, a GitHub Actions workflow is included, but requires a one-time FOOTBALL_DATA_API_TOKEN secret in this GitHub repository's Actions settings. Once enabled, forecasts and final scores are automatically committed to the public repository's data/tracker_archive.json. Only predictions and results—not tokens—are stored.")
        st.caption("Cloud tracking is an optional public research archive; anyone with the public repository link can view the predictions. Saved files are not a private account.")
    if token and rows:
        if st.button("Check finished results",use_container_width=True):
            pending_codes={r["league_code"] for r in rows if r.get("result") is None}
            checked=0;new_results=0;errors={}
            for code in sorted(pending_codes):
                if code not in LEAGUES.values():continue
                try:
                    latest=live_matches(code,token)
                    count=reconcile_local(code,latest["matches"])
                    rows,new_cloud=settle_with_feed(rows,code,latest["matches"])
                    checked+=1;new_results+=count+new_cloud
                except Exception as exc:
                    errors[code]=type(exc).__name__
            save_archive_local(rows)
            cloud_archive.clear()
            st.success(f"Checked {checked} leagues against finished fixtures. Stored results were refreshed where a final score was available.")
            if errors:st.warning(f"Could not update {len(errors)} league(s); try again later.")
            st.rerun()
    if not rows:
        st.info("No archived predictions yet. Open **Brief**, tap **Build / refresh weekly brief**, and the model will record upcoming matches before kickoff. Return to this tab after fixtures finish.")
        return
    _summary(rows)
    st.subheader("Week-by-week performance")
    week_df=pd.DataFrame(weekly_stats(rows))
    if not week_df.empty:
        display=week_df.copy()
        for col in ("accuracy","strong_accuracy","upset_precision"):
            display[col]=display[col].map(_pct)
        st.dataframe(display,hide_index=True,use_container_width=True)
    st.subheader("Monthly performance")
    month_df=pd.DataFrame(monthly_stats(rows))
    if not month_df.empty:
        display=month_df.copy()
        for col in ("accuracy","strong_accuracy","upset_precision"):
            display[col]=display[col].map(_pct)
        st.dataframe(display,hide_index=True,use_container_width=True)
    st.subheader("Prediction history")
    choice=st.selectbox("Show",("All","Awaiting result","60%+ only","Upset alerts","Correct picks","Incorrect picks"))
    shown=rows
    if choice=="Awaiting result":shown=[r for r in rows if not r.get("result")]
    if choice=="60%+ only":shown=[r for r in rows if r.get("strong_60")]
    if choice=="Upset alerts":shown=[r for r in rows if r.get("upset_alert")]
    if choice=="Correct picks":shown=[r for r in rows if r.get("result") and r["result"]["correct"]]
    if choice=="Incorrect picks":shown=[r for r in rows if r.get("result") and not r["result"]["correct"]]
    if shown:
        st.dataframe(_table(shown),hide_index=True,use_container_width=True)
    else:st.caption("No matching archived predictions yet.")
    st.download_button("Download tracker results (CSV)",to_csv(rows),
        file_name=f"football_lab_tracker_{now:%Y%m%d}.csv",mime="text/csv",
        use_container_width=True)
    st.download_button("Download full backup (JSON)",json.dumps(rows,indent=2),
        file_name=f"football_lab_tracker_backup_{now:%Y%m%d}.json",mime="application/json",
        use_container_width=True)
    st.caption("Accuracy is evaluated on finished matches only. Brier and calibration charts can be added once enough independent scored forecasts accumulate.")
