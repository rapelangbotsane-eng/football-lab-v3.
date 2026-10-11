"""Football Lab mobile-first dashboard. Results only; does not invent probabilities."""
from datetime import datetime,timezone,timedelta
from html import escape
import io
import time
import pandas as pd
import streamlit as st
from club_ui import club_model,live_matches,CLUB_MODEL_VERSION
from live_data import LEAGUES, FootballDataError
from club_data import HistoricalDataError
from weekly_scout import forecast_week,strong_predictions,upset_predictions,SA_TIME

PAGES=("🏠 Brief","⭐ 60%+ picks","⚡ Upset watch","📅 Matches","🔬 Research")
DEFAULT_LEAGUES=list(LEAGUES)[:5]
CSS="""
<style>
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{background:#0a111f;color:#eff5fb}
[data-testid="stHeader"]{background:transparent}
[data-testid="stMainBlockContainer"]{max-width:1120px;padding-top:1.3rem;padding-bottom:5rem}
h1,h2,h3{letter-spacing:-.035em!important;font-family:system-ui,-apple-system,sans-serif!important}
h2{font-size:1.46rem!important}h3{font-size:1.2rem!important}
[data-testid="stSidebar"]{background:#111c2e}
[data-testid="stMetric"]{background:#121e32;border:1px solid #26364d;border-radius:16px;padding:13px!important}
[data-testid="stMetricLabel"]{color:#9baec4}
[data-testid="stMetricValue"]{font-size:1.85rem!important;color:#edf7fe}
div[data-testid="stRadio"]>div{display:flex;flex-wrap:nowrap!important;overflow-x:auto;gap:.30rem;
padding:5px 0 10px;scrollbar-width:none}
div[data-testid="stRadio"] label{white-space:nowrap;border:1px solid #253752;background:#111d30;
padding:9px 12px;border-radius:12px;min-height:42px}
div[data-testid="stRadio"] label:has(input:checked){border-color:#c8f57d;background:#23332a}
div[data-testid="stRadio"] label p{font-size:.83rem;font-weight:650}
div[data-testid="stRadio"] label>div:first-child{display:none}
div.stButton>button[kind="primary"]{border:0;border-radius:12px;background:#c8f57d;color:#0b1724;
font-weight:800;min-height:49px}
div.stButton>button[kind="primary"]:hover{background:#defca7;color:#0b1724}
div.stDownloadButton>button{border-radius:12px}
div[data-testid="stExpander"]{border-color:#25364d;background:#0f1929;border-radius:13px}
.fl-hero{border:1px solid #2c433e;background:linear-gradient(135deg,#142b31 0%,#101d30 62%,#111d32 100%);
border-radius:23px;padding:27px 29px;margin-bottom:18px;overflow:hidden}
.fl-eyebrow{font-size:11px;letter-spacing:.17em;font-weight:800;color:#c8f57d;text-transform:uppercase}
.fl-hero h1{color:#f8fbff;font-size:clamp(31px,6vw,48px);line-height:1.04;margin:10px 0 12px;font-weight:850}
.fl-hero p{color:#afc3d3;margin:0;max-width:580px;font-size:15px;line-height:1.6}
.fl-panel{border:1px solid #25364b;border-radius:19px;padding:19px 20px;background:#121d30;margin:13px 0}
.fl-empty{border:1px dashed #38506a;border-radius:16px;padding:26px;color:#afc3d3;
background:#121d30;margin:10px 0;line-height:1.6}
.fl-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:11px;margin:12px 0 20px}
.fl-stat{background:#121e32;border:1px solid #243b4d;border-radius:15px;padding:13px}
.fl-stat .number{color:#f2f8fb;font-size:27px;font-weight:850;line-height:1.3}
.fl-stat .label{color:#a3b8ca;font-size:11px;letter-spacing:.02em}
.fl-match{background:linear-gradient(110deg,#132035,#101b2e);border:1px solid #273b50;
border-radius:18px;margin:13px 0 8px;padding:16px 18px}
.fl-match .meta{color:#95abc3;font-size:11px;font-weight:750;letter-spacing:.045em}
.fl-match .teams{font-weight:800;font-size:clamp(17px,3.5vw,23px);letter-spacing:-.02em;
margin:12px 0 14px;color:#f4f8ff;overflow-wrap:anywhere}
.fl-match .vs{font-size:11px;color:#9eafbf;font-weight:600;padding:0 7px;vertical-align:middle}
.fl-match .pills{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}
.fl-match .pill{background:#0b1728;border:1px solid #263649;border-radius:10px;text-align:center;padding:9px 3px}
.fl-match .pill strong{font-size:18px;color:#e5edf7;display:block}
.fl-match .pill span{font-size:11px;color:#9eafbf}
.fl-match .pill.active{border-color:#b7e674;background:#263427}
.fl-match .pill.active strong{color:#d9ff9e}
.fl-match .footer{display:flex;flex-wrap:wrap;justify-content:space-between;gap:9px;
border-top:1px solid #28394d;margin-top:13px;padding-top:13px;color:#a7b8c9;font-size:12px}
.fl-upset{color:#ffdc96;font-weight:800}
.fl-ok{color:#c8f57d;font-weight:800}
.fl-section{font-size:19px;font-weight:800;letter-spacing:-.025em;margin:19px 0 3px;color:#f5f9ff}
.fl-kicker{color:#a6b8cb;font-size:13px;margin:0 0 14px}
@media(max-width:640px){
 [data-testid="stMainBlockContainer"]{padding-left:.77rem;padding-right:.77rem;padding-top:.55rem}
 .fl-hero{padding:21px 18px;border-radius:18px}
 .fl-hero h1{font-size:33px}
 .fl-match{padding:14px 12px}
 .fl-match .teams{font-size:18px}
 .fl-match .pill strong{font-size:16px}
 .fl-stat{padding:10px 8px}
 .fl-stat .number{font-size:23px}
 div[data-testid="stRadio"] label{padding:8px 9px}
}
</style>
"""

def _safe(value):
    return escape(str(value or ""),quote=True)

def apply_style():
    st.markdown(CSS,unsafe_allow_html=True)

def _secrets_token():
    try:return st.secrets.get("FOOTBALL_DATA_API_TOKEN","")
    except (FileNotFoundError,KeyError,AttributeError):return ""

def _header():
    st.markdown("""<section class="fl-hero">
      <div class="fl-eyebrow">⚽ FOOTBALL LAB <span style="color:#9cacb6">/ V3.3 SCOUT</span></div>
      <h1>Your football,<br>already analysed.</h1>
      <p>The week's strongest model estimates and potential upset signals, ranked for you. Research-backed probabilities, never guaranteed results.</p>
    </section>""",unsafe_allow_html=True)

def _empty(text):
    st.markdown(f'<div class="fl-empty">{_safe(text)}</div>',unsafe_allow_html=True)

def _card(r,show_upset=False,details=False):
    p=r["probabilities"];winner=r["pick"]
    pair=[]
    for code,label in (("1","HOME"),("X","DRAW"),("2","AWAY")):
        active=" active" if code==winner else ""
        pair.append(f'<div class="pill{active}"><strong>{p[code]*100:.1f}%</strong><span>{label}</span></div>')
    if show_upset:
        footer=f'<span class="fl-upset">⚡ Research-only: {_safe(r["underdog_team"])} · ML {r["upset_chance"]:.1%} (+{r["upset_uplift"]*100:.1f} pp vs Elo)</span>'
    else:
        footer=f'<span class="fl-ok">★ Top outcome: {_safe(r["pick_label"])} · {r["probability"]:.1%}</span>'
    st.markdown(f"""<article class="fl-match">
      <div class="meta">{_safe(r["league"])} &nbsp;·&nbsp; {_safe(r["local_time"])}</div>
      <div class="teams">{_safe(r["home"])} <span class="vs">VS</span> {_safe(r["away"])}</div>
      <div class="pills">{''.join(pair)}</div>
      <div class="footer">{footer}<span>{'Elo' if r['engine']=='elo' else 'Trained ML'} primary · {r['historic_home']}/{r['historic_away']} matches</span></div>
    </article>""",unsafe_allow_html=True)
    if details:
        with st.expander(f"Analysis · {r['home']} vs {r['away']}"):
            st.write(f"**Main model:** {'Elo team-strength rating' if r['engine']=='elo' else 'trained results/form model'} (selected using the 2024/25 validation results).")
            st.write(f"**Probabilities:** Home {p['1']:.1%}, Draw {p['X']:.1%}, Away {p['2']:.1%}.")
            if r["upset_alert"]:
                b=r["baseline"][r["underdog"]]
                st.write(f"**Experimental upset signal:** {r['underdog_team']} wins: ML {r['upset_chance']:.1%} vs Elo {b:.1%} ({r['upset_uplift']*100:+.1f} percentage points). This does not change the main forecast when Elo is selected.")
            st.caption(r["source_note"])

def _section(title,subtitle=None):
    st.markdown(f'<div class="fl-section">{_safe(title)}</div>',unsafe_allow_html=True)
    if subtitle:st.markdown(f'<div class="fl-kicker">{_safe(subtitle)}</div>',unsafe_allow_html=True)

def _stats(count,strong,upsets):
    st.markdown(f"""<div class="fl-grid">
      <div class="fl-stat"><div class="number">{count}</div><div class="label">MATCHES ANALYSED</div></div>
      <div class="fl-stat"><div class="number">{strong}</div><div class="label">60%+ ESTIMATES</div></div>
      <div class="fl-stat"><div class="number">{upsets}</div><div class="label">UPSET SIGNALS</div></div>
    </div>""",unsafe_allow_html=True)

def _scan(selected,days,token):
    all_cards=[];errors={};coverage={}
    now=datetime.now(timezone.utc)
    progress=st.progress(0,text="Preparing your weekly brief...")
    for i,league in enumerate(selected):
        code=LEAGUES[league]
        progress.progress(i/len(selected),text=f"{league} · loading history and fixtures ({i+1}/{len(selected)})")
        try:
            records,trained=club_model(code,CLUB_MODEL_VERSION)
            # Cached by league; we do not request from the provider every rerun.
            snapshot=live_matches(code,token)
            cards,counts=forecast_week(records,trained,snapshot,league,now=now,horizon_days=days)
            coverage[league]=counts
            all_cards.extend(cards)
        except (FootballDataError,HistoricalDataError,ValueError,TypeError,KeyError,RuntimeError) as exc:
            errors[league]=str(exc)
        except Exception as exc:
            errors[league]=f"{type(exc).__name__} while processing this competition"
        progress.progress((i+1)/len(selected),text=f"Processed {i+1} of {len(selected)} leagues")
    progress.empty()
    return dict(cards=sorted(all_cards,key=lambda x:x["datetime"]),errors=errors,
                coverage=coverage,leagues=list(selected),days=days,scanned_at=now.isoformat())

def render_home():
    apply_style();_header()
    now=datetime.now(SA_TIME)
    st.markdown(f"**{now.strftime('%A %d %B %Y')}** &nbsp; · &nbsp; South Africa time")
    st.radio("Navigate",PAGES,horizontal=True,key="scout_nav",label_visibility="collapsed")
    page=st.session_state.scout_nav
    if page==PAGES[4]:
        from research_ui import render as research_render
        research_render()
        return
    token=_secrets_token()
    if not token:
        st.error("The match-data token is missing. Add FOOTBALL_DATA_API_TOKEN in Streamlit → Manage app → Settings → Secrets. Never put the token in public GitHub.")
        return
    with st.expander("⚙️ Weekly scan settings",expanded="scout_snapshot" not in st.session_state):
        selected=st.multiselect("Leagues to analyse",list(LEAGUES),
            default=DEFAULT_LEAGUES,key="scout_leagues")
        days=st.selectbox("Upcoming window",options=(7,14),format_func=lambda n:f"Next {n} days",key="scout_days")
        st.caption("Five major leagues are selected by default. Free feeds have rate limits, so the first scan can take a few minutes. Nothing is fetched until you tap the scan button.")
    if st.button("⚡ Build / refresh weekly brief",type="primary",use_container_width=True,disabled=not bool(selected)):
        st.session_state.scout_snapshot=_scan(selected,days,token)
        st.session_state.scout_refreshed=True
    snapshot=st.session_state.get("scout_snapshot")
    if snapshot is None:
        _empty("Your weekly shortlist is ready to build. Choose your leagues, then tap ‘Build / refresh weekly brief’ once. Football Lab will check every upcoming match and rank the results for you.")
        st.caption("Free API access and historical team-name matching can restrict coverage. The app never invents predictions for unmapped teams.")
        return
    cards=snapshot["cards"]
    strong=strong_predictions(cards,.60)
    upsets=upset_predictions(cards)
    scanned=datetime.fromisoformat(snapshot["scanned_at"]).astimezone(SA_TIME)
    st.caption(f"Last scanned {scanned:%d %b at %H:%M SAST} · {snapshot['days']}-day window · {len(snapshot['leagues'])} selected leagues. Tap refresh for newer results.")
    if snapshot["errors"]:
        with st.expander(f"⚠️ {len(snapshot['errors'])} competition(s) could not be analysed"):
            for k,v in snapshot["errors"].items():st.warning(f"{k}: {v}")
    not_covered=sum(sum(int(n.get(k,0)) for k in ("unmapped","insufficient","failed")) for n in snapshot["coverage"].values())
    if not_covered:st.caption(f"{not_covered} upcoming match(es) were not forecast because names/data were missing or predictions failed; excluded, not guessed.")
    if not cards:
        _stats(0,0,0)
        _empty("No usable upcoming matches were found in the selected leagues, or the free API did not return the fixtures. Open scan errors or choose another league and refresh.")
        return
    if page==PAGES[0]:
        _stats(len(cards),len(strong),len(upsets))
        _section("Strongest this week","Highest single 1/X/2 model probabilities of 60% or more")
        if strong:
            for r in strong[:3]:_card(r,details=True)
            if len(strong)>3:st.caption(f"{len(strong)-3} more strong estimate(s) available under 60%+ picks.")
        else:_empty("No matches reach 60% this week. That's a valid result; Football Lab won't lower the standard just to show picks.")
        _section("Upset radar","An experimental ML-vs-Elo disagreement signal, not a recommendation to bet")
        if upsets:
            for r in upsets[:3]:_card(r,show_upset=True,details=True)
            if len(upsets)>3:st.caption(f"{len(upsets)-3} additional underdog signal(s) in Upset watch.")
        else:_empty("No fixture meets the predefined upset-alert conditions this week.")
        _section("Coming up","Next five modelled fixtures")
        for r in cards[:5]:_card(r,details=False)
    elif page==PAGES[1]:
        _section("60%+ picks","The highest estimated 1/X/2 outcome per fixture. Percentages are model estimates, not guarantees.")
        threshold=st.slider("Minimum model probability",60,85,60,5,format="%d%%")
        filtered=strong_predictions(cards,threshold/100)
        st.caption(f"{len(filtered)} of {len(cards)} forecast match(es) meet the threshold.")
        if not filtered:_empty("No predictions reach your selected threshold for these dates. Avoid forcing a pick.")
        for r in filtered:_card(r,details=True)
    elif page==PAGES[2]:
        _section("Potential upsets","A challenger model flags underdogs whose chance is at least 7 percentage points above Elo and at least 27%.")
        st.warning("RESEARCH ONLY: the upset model is not proven superior to Elo. An alert does not mean the underdog is favoured or that this is a profitable bet.")
        st.caption(f"{len(upsets)} alerts among {len(cards)} modelled fixture(s). Ranked by ML-vs-Elo disagreement.")
        if not upsets:_empty("No potential upsets meet the experimental criteria this week.")
        for r in upsets:_card(r,show_upset=True,details=True)
    elif page==PAGES[3]:
        _section("Upcoming matches","All mapped fixtures, ordered by kickoff. Tap Analysis for details.")
        for r in cards:_card(r,details=True)
    if page!=PAGES[4]:
        st.divider()
        st.caption("Data: football-data.org fixtures plus football-data.co.uk historic results. Main forecasts use Elo or results/form ML chosen by older validation; current xG, injury news and lineups are not included. Probabilities are research estimates, not financial advice. Odds and value are not assessed.")
        rows=[dict(league=r["league"],kickoff_sast=r["local_time"],home=r["home"],away=r["away"],
            home_win=r["probabilities"]["1"],draw=r["probabilities"]["X"],away_win=r["probabilities"]["2"],
            main_pick=r["pick"],main_probability=r["probability"],strong_60=r["probability"]>=.60,
            upset_research_alert=r["upset_alert"],ml_underdog_uplift=r["upset_uplift"],engine=r["engine"]) for r in cards]
        st.download_button("↓ Download this week's analysis (CSV)",pd.DataFrame(rows).to_csv(index=False),
            file_name=f"football_lab_weekly_{scanned:%Y%m%d}.csv",mime="text/csv",use_container_width=True)
