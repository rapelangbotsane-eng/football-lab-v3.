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
st.title("⚽ Football Lab V3 — Advanced Research & Current Fixtures")
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
tab_data,tab_live,tab_match,tab_test,tab_about=st.tabs(["📥 Data & training","📅 Upcoming fixtures","🎯 Analyse matchup","📊 Backtest & upsets","ℹ️ About"])

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
    st.subheader("Upcoming men's club matches — current-season data")
    st.info("This tab uses football-data.org's scores and fixtures. Forecasts here are a simple, EXPERIMENTAL goals/form baseline, **not** the advanced StatsBomb-trained model. No live xG, pressure, progression, lineup or injury data is included.")
    try:
        token=st.secrets.get("FOOTBALL_DATA_API_TOKEN","")
    except (FileNotFoundError,KeyError,AttributeError):
        token=""
    if not token:
        st.warning("API token not connected yet. In Streamlit Community Cloud, open Manage app → Settings → Secrets. Add FOOTBALL_DATA_API_TOKEN = \"your-token\" and save. Never share the token or put it in GitHub.")
    else:
        league_label=st.selectbox("Select league",list(LEAGUES))
        league_code=LEAGUES[league_label]
        if st.button("Load current fixtures",type="primary",width="stretch"):
            try:
                snapshot=live_fixtures(league_code,token)
                st.session_state.live_snapshot=snapshot
                st.session_state.live_code=league_code
                st.success("Connected to football-data.org.")
            except FootballDataError as exc:
                st.error(str(exc))
                st.session_state.pop("live_snapshot",None)
        snapshot=st.session_state.get("live_snapshot") if st.session_state.get("live_code")==league_code else None
        if snapshot:
            from zoneinfo import ZoneInfo
            matches=snapshot["matches"]
            upcoming=next_fixtures(matches,horizon_days=21)
            historical=sum(1 for m in matches if m.get("status")=="FINISHED")
            st.caption(f"API snapshot: {snapshot['fetched'][:19]} UTC · {historical} finished matches available this season · Results may be delayed on the free plan. Cached for 30 minutes.")
            if not upcoming:
                st.warning("No scheduled fixtures in the next 21 days for this league. Try another league.")
            for fixture in upcoming[:15]:
                h=fixture.get("homeTeam") or {}
                a=fixture.get("awayTeam") or {}
                start=kickoff(fixture)
                when=start.astimezone(ZoneInfo("Africa/Johannesburg")).strftime("%d %b %Y, %H:%M SAST") if start else "Time unavailable"
                title=f"{h.get('shortName') or h.get('name','Home')} vs {a.get('shortName') or a.get('name','Away')} · {when}"
                with st.expander(title):
                    estimate=forecast_fixture(fixture,matches)
                    if not estimate:
                        st.warning("Insufficient fixture information.")
                    elif estimate["insufficient"]:
                        st.warning(f"Insufficient recent results for a basic forecast: home {estimate['home_samples']}, away {estimate['away_samples']}. Minimum 5 matches per team. No prediction generated.")
                    else:
                        c1,c2,c3=st.columns(3)
                        c1.metric("Home win (1)",f"{estimate['p_home']:.1%}")
                        c2.metric("Draw (X)",f"{estimate['p_draw']:.1%}")
                        c3.metric("Away win (2)",f"{estimate['p_away']:.1%}")
                        st.write(f"**Experimental pick:** {estimate['pick']} · Most likely score: {estimate['top_score']}")
                        st.caption(f"Based on only {estimate['home_samples']} recent home-team and {estimate['away_samples']} away-team results from this league. Model-based goal rates: {estimate['home_xg']:.2f} / {estimate['away_xg']:.2f} (not provider xG). These forecasts have not passed a real historical calibration test.")
            st.caption("The upcoming-fixtures baseline does not validate or replace the advanced historical engine. A proper current-data model and prospective scorecard are next.")

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
