"""Optional Understat comparison panel; research only, never changes live predictor."""
import streamlit as st
from club_ui import club_model
from understat_data import retrieve_historical, UnderstatUnavailable
from advanced_club_experiment import run_xg_experiment

@st.cache_data(ttl=24*3600,max_entries=1,show_spinner=False)
def advanced_history():
    # Explicit user-initiated HTTP call only; no credential, bot evasion, or proxy.
    return retrieve_historical()

@st.cache_data(ttl=24*3600,max_entries=1,show_spinner=False)
def compare_advanced(rows,understat_matches):
    return run_xg_experiment(rows,understat_matches)

def render():
    st.subheader("V3.2 — Test advanced football data")
    st.write("Research question: do **real match-level xG and pressing/deep-completion metrics** improve historical Premier League predictions over Elo and results-only machine learning?")
    st.info("The research feed is provided by Understat via an unofficial, read-only interface. Data availability and permitted use can change. We stop on access errors and never work around restrictions. This is a research experiment, NOT a replacement for the current-fixtures model.")
    st.caption("All four models train and validate on the same matched games. Pre-match features only use earlier results. The comparison uses 2020/21–2023/24 for training, 2024/25 for validation and 2025/26 for retrospective testing.")
    st.warning("The 2025/26 season was already examined in Football Lab V3.1. It cannot be treated as a pristine holdout for newly developed xG features. This is diagnostic evidence; the decisive test will be a future prospective season.")
    if st.button("Run free Premier League xG experiment",type="primary",width="stretch"):
        try:
            with st.spinner("Reading six seasons of Premier League results and advanced metrics. This may take a few minutes..."):
                records,_=club_model("PL", "3.1-upset-challenger-20261011")
                events=advanced_history()
                report=compare_advanced(records,events)
                st.session_state.xg_comparison=report
                st.success("Advanced data joined and compared with the Elo baseline.")
        except (UnderstatUnavailable,ValueError,TypeError,KeyError,RuntimeError) as exc:
            st.session_state.pop("xg_comparison",None)
            st.error(f"Advanced research could not finish: {exc}")
            st.caption("Current men's club predictions remain available and unchanged.")
        except Exception as exc:
            st.session_state.pop("xg_comparison",None)
            st.error(f"Source or integration error: {type(exc).__name__}. Current predictions were not changed.")
    result=st.session_state.get("xg_comparison")
    if not result:
        st.caption("No experimental results yet. Press the button once; subsequent successful results are cached for 24 hours.")
        return
    st.subheader("Data coverage")
    a,b,c=st.columns(3)
    a.metric("Exactly matched matches",result["source_matched"])
    b.metric("Unmatched matches",result["source_unmatched"])
    c.metric("2025/26 comparisons",result["retrospective_matches"])
    st.write("Source coverage by season (matches with exact team, date and final-score matching):")
    st.json(result["matching_by_season"],expanded=False)
    st.subheader("Model comparison — same matches")
    st.dataframe(result["table"],hide_index=True,width="stretch")
    st.caption("Lower log loss and Brier are better; accuracy alone can mislead. Model selection occurs on validation results, NOT the retrospective test.")
    st.write(f"**Best validation log loss:** {result['selected']}")
    st.caption(result["notes"])
    st.subheader("Metrics actually used")
    st.write("Chance quality and underlying performance:",", ".join(result["used_features"]["xg"]))
    st.write("Pressing intensity and attacking depth:",", ".join(result["used_features"]["pressing"]))
    st.caption("Pressing here is PPDA and deep completions from Understat, not event-derived player tracking or a claim to full tactical intelligence.")
    st.download_button("Download research comparison (CSV)",result["table"].to_csv(index=False),
        file_name="football_lab_v32_xg_ablation.csv",mime="text/csv")
    st.warning("Even if the xG model wins this retrospective test, Football Lab will NOT automatically switch live picks. We need stable current advanced-data coverage and new pre-kickoff forecasts scored prospectively.")
