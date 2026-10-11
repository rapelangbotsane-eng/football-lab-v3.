"""Football Lab V3 historical research dashboard (free StatsBomb events)."""
import json
from datetime import date,timedelta
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from data_source import list_competitions,list_matches,collect
from advanced_engine import fixture_rows,train_and_evaluate,make_prediction
from live_data import LEAGUES, FootballDataError, fetch_season_matches, next_fixtures, forecast_fixture, kickoff

st.set_page_config(page_title="Football Lab V3",page_icon="⚽",layout="wide")
st.title("⚽ Football Lab V3.1 — Club Predictions & Advanced Research")
st.warning("Research prototype: free StatsBomb event coverage is historical and incomplete. This is not a live fixture or injury feed. Match probabilities have not been proven reliable for current competitions.")
ROOT=Path("research_workspace");ROOT.mkdir(exist_ok=True)
MODEL=ROOT/"model.joblib";RECORDS=ROOT/"records.json"

@st.cache_data(ttl=43200)
def competitions():return list_competitions()
@st.cache_data(ttl=43200)
def fixtures(cid,sid):return list_matches(cid,sid)

@st.cache_data(ttl=1800,show_spinner=False)
def live_fixtures(league_code, _token):
    return fetch_season_matches(_token, league_code)

if "research" not in st.session_state and MODEL.exists() and RECORDS.exists():
    try:
        records=json.loads(RECORDS.read_text())
        frame,history,elo,last=fixture_rows(records)
        st.session_state.research=dict(records=records,frame=frame,history=history,elo=elo,last=last,model=joblib.load(MODEL))
    except Exception as exc:st.warning(f"Could not restore previous training: {exc}")
tab_data,tab_live,tab_match,tab_test,tab_about=st.tabs(["📥 Event-data research","⚽ Men\'s club predictions","🎯 Historical matchups","📊 Event backtest","ℹ️ About"])

with tab_data:
    st.subheader("Train using real historical match events")
    st.write("Select historical competition seasons, then tap **Sync and train**. Downloads can take a few minutes. This does not supply up-to-date metrics for today's teams.")
    try:
        seasons=competitions()
        choices={f"{c.get('competition_name')} · {c.get('season_name')} ({c['competition_id']}/{c['season_id']})":c for c in seasons}
    except Exception as exc:
        st.error(f"Could not access the historical data feed: {exc}")
        choices={}
    defaults=[s for s in choices if "Bundesliga" in s and "2023/2024" in s]
    chosen=st.multiselect("Historical seasons",list(choices),default=defaults[:1] if defaults else list(choices)[:1],max_selections=4)
    limit=st.slider("Maximum matches per season",25,450,125,25)
    if st.button("Sync event data + train model",type="primary",disabled=not chosen,width="stretch"):
        bar=st.progress(0);records=[];errors=[]
        for n,key in enumerate(chosen):
            spec=choices[key]
            try:
                matches=fixtures(spec["competition_id"],spec["season_id"])
                def cb(frac,ok,bad):
                    bar.progress(min(1.,(n+frac)/len(chosen)),text=f"Season {n+1}/{len(chosen)}: {ok} complete, {bad} skipped")
                got,failed=collect(matches,limit=limit,cache_dir=ROOT/"events",on_progress=cb)
                records.extend(got);errors.extend(failed)
            except Exception as exc:errors.append(f"{key}: {exc}")
        records=list({m["match_id"]:m for m in records}.values())
        if errors:st.warning(f"{len(errors)} dataset files skipped. First issue: {errors[0]}")
        try:
            frame,history,elo,last=fixture_rows(records)
            model=train_and_evaluate(frame)
            st.session_state.research=dict(records=records,frame=frame,history=history,elo=elo,last=last,model=model)
            RECORDS.write_text(json.dumps(records));joblib.dump(model,MODEL)
            st.success(f"Trained successfully on {len(frame)} real historical matches.")
        except Exception as exc:st.error(f"Training did not finish: {exc}. Select more suitable seasons or increase the match limit.")
    r=st.session_state.get("research")
    if r:
        metrics=r["model"]["report"]
        a,b,c=st.columns(3)
        a.metric("Matches",metrics["matches"])
        b.metric("Unseen-match accuracy",f"{metrics['accuracy']:.1%}")
        c.metric("Baseline accuracy",f"{metrics['baseline_accuracy']:.1%}")
        st.caption("These are measured on held-out historical event-covered matches only, not current fixtures.")

with tab_live:
    from club_ui import render as render_club_predictions
    render_club_predictions()

with tab_match:
    r=st.session_state.get("research")
    if not r:st.info("Train the model first using the Data & training tab.")
    else:
        teams=sorted(r["history"])
        if len(teams)<2:st.error("Need at least two teams.")
        else:
            left,right=st.columns(2)
            home=left.selectbox("Home team",teams)
            away=right.selectbox("Away team",teams,index=1)
            last=date.fromisoformat(max(x["date"] for x in r["records"]))
            chosen_date=st.date_input("Hypothetical match date",last+timedelta(days=7),min_value=last+timedelta(days=1))
            if st.button("Analyse matchup",type="primary",width="stretch"):
                try:
                    st.session_state.forecast=make_prediction(r["model"],home,away,chosen_date.isoformat(),r["history"],r["elo"],r["last"])
                except Exception as exc:st.error(str(exc))
            p=st.session_state.get("forecast")
            if p:
                st.subheader(f"{p['home']} vs {p['away']}")
                a,b,c=st.columns(3)
                a.metric("Home 1",f"{p['probs']['1']:.1%}")
                b.metric("Draw X",f"{p['probs']['X']:.1%}")
                c.metric("Away 2",f"{p['probs']['2']:.1%}")
                if p["low_data"]:st.error("Less than 5 event-covered past matches for at least one team. Treat the result as low confidence.")
                if p["upset_watch"]:st.warning(f"Upset watch: {p['underdog']} underdog; +{p['underdog_uplift']:.1%} model uplift versus Elo proxy.")
                st.subheader("10,000 score simulations")
                st.dataframe(pd.DataFrame(p["top_scores"]),hide_index=True,width="stretch")
                st.caption("Statistical goal draws reweighted toward the classifier's outcome probabilities; not a physical simulation of football.")
                st.subheader("Underlying team metrics")
                st.dataframe(pd.DataFrame([dict(metric=k,home=v[0],away=v[1]) for k,v in p["metrics"].items()]),hide_index=True,width="stretch")
                st.download_button("Save forecast as JSON",json.dumps(p,indent=2),file_name="forecast.json",mime="application/json")

with tab_test:
    r=st.session_state.get("research")
    if not r:st.info("Train first to view genuine historical backtest.")
    else:
        m=r["model"];report=m["report"]
        x,y,z=st.columns(3)
        x.metric("Holdout accuracy",f"{report['accuracy']:.1%}")
        y.metric("Log loss",f"{report['log_loss']:.3f}")
        z.metric("Multiclass Brier",f"{report['brier_multiclass']:.3f}")
        st.caption(f"Train: {report['train_matches']} matches. Holdout: {report['test_matches']} matches starting {report['test_start']}. Same-day matches do not leak into features.")
        h=m["holdout"].copy();h["correct"]=h["forecast"]==h["outcome"]
        st.dataframe(h,hide_index=True,width="stretch")
        misses=h[~h["correct"]]
        st.subheader("Missed predictions")
        st.dataframe(misses,hide_index=True,width="stretch")

with tab_about:
    st.markdown("""This prototype works with **selected historical matches** from [StatsBomb Open Data](https://github.com/hudl/open-data).
    
    Advanced signals: shot xG and conceded xG, high-quality shots, successful progressive passes and carries, final-third/box entries, pressures, quick post-pressure recoveries, high regains and a **PPDA proxy**. The pressing measures are approximations, not provider-certified measures. xT, VAEP, injuries, confirmed lineups and live fixtures are **not** included.
    
    Each match's feature row is computed using only earlier calendar dates. Evaluation uses a time-ordered holdout. The upset indicator is experimental: accuracy and precision must be measured on real results before use.
    
    Streaming and hosted computing are free-tier limited. Stored research data may disappear on app restarts. Research only, not betting advice.
    """)
