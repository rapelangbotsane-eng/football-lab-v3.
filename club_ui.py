"""Streamlit men's current fixtures with automatically trained historical club model."""
import pandas as pd
import streamlit as st
from zoneinfo import ZoneInfo
from club_data import historical_league, current_season_results, fixture_teams, HistoricalDataError
from club_engine import build_history, train_model, forecast
from live_data import LEAGUES, FootballDataError, fetch_season_matches, next_fixtures, kickoff, forecast_fixture

@st.cache_resource(ttl=24*3600,show_spinner=False)
def club_model(code):
    # Six seasons of men's club results, no token, odds, or private credentials.
    rows=historical_league(code)
    history,_=build_history(rows)
    result=train_model(history)
    return rows,result

@st.cache_data(ttl=1800,show_spinner=False)
def live_matches(code,_token):
    return fetch_season_matches(_token,code)

def metric_card(label,result):
    return f"{result['accuracy']:.1%} accuracy · {result['log_loss']:.3f} log loss · {result['brier']:.3f} Brier"

def render():
    st.subheader("Men's club predictions — V3.1")
    st.write("Choose a league and tap **Load & analyse matches**. Football Lab automatically trains on six previous seasons and loads current fixtures. No programming or manual statistics required.")
    st.info("Club model input: past scores, Elo, venue form and rest. No current xG, pressing, progressive passes, confirmed lineups or injury data is supplied by this free API. Advanced event research remains available in Data & training.")
    try:
        token=st.secrets.get("FOOTBALL_DATA_API_TOKEN","")
    except (FileNotFoundError,KeyError,AttributeError):
        token=""
    if not token:
        st.warning("Add FOOTBALL_DATA_API_TOKEN to Streamlit → Manage app → Settings → Secrets; never add it to public GitHub.")
        return
    league=st.selectbox("Men's league",list(LEAGUES))
    code=LEAGUES[league]
    supported=code in ("PL","PD","BL1","SA","FL1","DED","PPL")
    if st.button("Load & analyse matches",type="primary",width="stretch",disabled=not supported):
        try:
            with st.spinner("Downloading free historical seasons and training the club model..."):
                history,trained=club_model(code)
            with st.spinner("Fetching current fixtures..."):
                snapshot=live_matches(code,token)
            st.session_state["club_view"]=dict(code=code,history=history,trained=trained,snapshot=snapshot)
            st.success("Historical backtest and current fixtures are loaded.")
        except (HistoricalDataError,FootballDataError,ValueError,KeyError,TypeError) as exc:
            st.session_state.pop("club_view",None)
            st.error(f"Could not complete analysis for this league: {exc}")
            st.info("The free historical feed might be temporarily unavailable. You can try again or choose another supported league.")
    view=st.session_state.get("club_view")
    if not view or view["code"]!=code:
        st.caption("The first league load may take a minute while several seasons are downloaded.")
        return
    trained=view["trained"];history=view["history"]
    report=trained["comparison"]
    st.divider()
    st.subheader("Independent historical benchmark — 2025/26")
    a,b,c=st.columns(3)
    a.metric("Selected model accuracy",f"{report['selected']['accuracy']:.1%}")
    b.metric("Selected model log loss",f"{report['selected']['log_loss']:.3f}")
    c.metric("Selected model Brier",f"{report['selected']['brier']:.3f}")
    st.write(f"**Chosen on 2024/25 validation:** {'Machine-learning model' if trained['selected']=='trained' else 'Elo baseline'}")
    st.caption(f"Training: {trained['train_size']} pre-2024/25 matches · Validation: {trained['val_size']} in 2024/25 · Untouched test: {trained['test_size']} in 2025/26. The model was fitted again on all historical results only after this test.")
    st.dataframe(pd.DataFrame([
        dict(method="Trained form + Elo",**report["trained"]),
        dict(method="Elo-only baseline",**report["elo"]),
        dict(method="Validation-selected",**report["selected"]),
    ]),hide_index=True,width="stretch")
    st.subheader("Machine-learning upset challenger — historical 2025/26")
    u=trained["upset"]
    st.write(f"**Elo underdog victories:** {u['underdog_wins']} / {trained['test_size']} · **ML alerts:** {u['alerts']} · **Wins captured:** {u['captured']} · **False alarms:** {u['false_alarms']}.")
    c1,c2,c3=st.columns(3)
    c1.metric("Alert precision",f"{u['hit_rate']:.1%}" if u['hit_rate'] is not None else "N/A")
    c2.metric("Underdog recall",f"{u['recall']:.1%}" if u['recall'] is not None else "N/A")
    c3.metric("Positive ML uplift cases",str(u['positive_uplifts']))
    st.caption("Upset alert = ML challenger assigns the Elo underdog at least 7 percentage points more winning probability than Elo and at least 27% absolute win chance. Thresholds are predefined. Underdog is defined by Elo, NOT bookmaker odds.")
    if trained["selected"]=="elo":
        st.info("Elo won the 2024/25 validation and remains the PRIMARY prediction engine. The ML upset signals below are CHALLENGER-ONLY research and are not evidence of a superior live prediction model.")
    st.caption(f"Top {u['top_candidates']} positive-uplift candidates captured {u['top_captured']} underdog wins, regardless of the fixed alert threshold. Ranking is for diagnosis only, not a validated betting strategy.")
    holdout=trained["holdout"]
    with st.expander("See top historical upset challenger candidates"):
        top=holdout[holdout["ml_upset_uplift"]>0].sort_values("ml_upset_uplift",ascending=False).head(15)
        if len(top):
            st.dataframe(top[["date","home","away","outcome","elo_underdog_probability","ml_underdog_probability","ml_upset_uplift","ml_challenger_alert","elo_underdog_won"]],hide_index=True,width="stretch")
        else:st.info("ML never raised the underdog probability above Elo on this holdout.")
    with st.expander("See all held-out match predictions"):
        st.dataframe(holdout,hide_index=True,width="stretch")
        st.download_button("Download the 2025/26 backtest (CSV)",holdout.to_csv(index=False),file_name=f"football_lab_{code}_2025_holdout.csv",mime="text/csv",key=f"download_{code}")
    st.divider()
    st.subheader("Upcoming fixtures")
    snapshot=view["snapshot"]
    matches=snapshot["matches"]
    future=next_fixtures(matches,horizon_days=21)
    st.caption(f"Fixture snapshot {snapshot['fetched'][:19]} UTC · current API results updated on refresh · predictions before kickoff only.")
    if not future:
        st.warning("No scheduled fixtures in the next 21 days. Try another league.")
    shown=0
    for game in future[:25]:
        home_name=(game.get("homeTeam") or {}).get("shortName") or (game.get("homeTeam") or {}).get("name","Home")
        away_name=(game.get("awayTeam") or {}).get("shortName") or (game.get("awayTeam") or {}).get("name","Away")
        start=kickoff(game)
        if not start:continue
        day=start.date().isoformat()
        local=start.astimezone(ZoneInfo("Africa/Johannesburg")).strftime("%d %b %Y · %H:%M SAST")
        with st.expander(f"{home_name} vs {away_name} — {local}"):
            h,a=fixture_teams(game,history)
            if not h or not a:
                st.warning("Cannot confidently match one or both clubs to historical data. No machine-learning forecast made.")
                continue
            fresh,skipped=current_season_results(matches,day,history)
            combined=history+fresh
            if skipped:
                st.caption(f"Some current matches were not matched to historical club names ({len(skipped)}); forecasts use only reliably matched results.")
            try:
                prediction=forecast(trained,combined,h,a,day)
            except (ValueError,KeyError,TypeError) as exc:
                st.error(f"Prediction unavailable: {exc}")
                continue
            if not prediction["available"]:
                st.warning(prediction["reason"])
                continue
            p=prediction["probabilities"]
            x,y,z=st.columns(3)
            x.metric("Home win (1)",f"{p['1']:.1%}")
            y.metric("Draw (X)",f"{p['X']:.1%}")
            z.metric("Away win (2)",f"{p['2']:.1%}")
            st.write(f"**Prediction:** {prediction['pick']} · **Engine:** {'Trained club ML' if prediction['engine']=='trained' else 'Elo only (better in validation)'}")
            if prediction["ml_challenger_alert"]:
                st.warning(f"RESEARCH-ONLY ML upset signal: {prediction['underdog']} underdog · ML probability {prediction['ml_underdog_probability']:.1%}, {prediction['underdog_uplift']:+.1%} versus Elo. {'Selected trained model' if prediction['engine']=='trained' else 'The primary prediction remains Elo, which performed better in validation.'} Historical alert precision is shown above; this is not a verified edge.")
            st.caption(f"Based on {prediction['history_count_home']} / {prediction['history_count_away']} historical+current matches. Advanced event data is NOT in these live predictions.")
            simple=forecast_fixture(game,matches)
            if simple and not simple.get("insufficient"):
                st.caption(f"Separate basic Poisson score mode: {simple['top_score']} (not a score forecast from the trained club classifier).")
            shown+=1
    if shown==0:
        st.warning("No upcoming fixture could be matched to sufficient club history. Please report the teams involved; the app will not invent a matchup estimate.")
    st.caption("Research estimates only. A historical backtest is not proof of future accuracy. Run upcoming matches prospectively and retain predictions before kickoff.")
