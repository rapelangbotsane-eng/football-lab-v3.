"""Read-only no-code view of GitHub Actions research training diagnostics."""
from datetime import datetime
import requests
import pandas as pd
import streamlit as st

URL="https://raw.githubusercontent.com/rapelangbotsane-eng/football-lab-v3./main/data/training_report.json"

@st.cache_data(ttl=600,show_spinner=False)
def fetch_report():
    response=requests.get(URL,timeout=12,headers={"User-Agent":"FootballLabTraining/1"})
    response.raise_for_status()
    report=response.json()
    if not isinstance(report,dict) or report.get("schema_version")!=1:
        raise ValueError("Training report has unexpected schema")
    return report

def render():
    st.subheader("Model Training Lab")
    st.write("Our independent research workbench: rolling-season training on earlier matches, selecting on a validation season and measuring errors on later games.")
    st.info("Training runs on GitHub rather than your phone. It **does not** replace your current live predictions without further testing.")
    if st.button("Refresh training results",key="load_training_report",type="primary"):
        fetch_report.clear()
        try:
            with st.spinner("Reading the latest public training report..."):
                st.session_state.training_lab_report=fetch_report()
        except (requests.RequestException,ValueError) as exc:
            st.error(f"Training report couldn't be read: {type(exc).__name__}. Try again shortly.")
    report=st.session_state.get("training_lab_report")
    if report is None:
        st.caption("Tap Refresh training results after the GitHub training workflow has completed. Results are saved as a public report, not embedded in the app code.")
        st.markdown("[View training workflow](https://github.com/rapelangbotsane-eng/football-lab-v3./actions/workflows/train-research.yml)")
        return
    status=report.get("status","unknown")
    if status=="queued":
        st.warning("Training job prepared. GitHub has not yet published completed results.")
        st.markdown("[Check GitHub Actions](https://github.com/rapelangbotsane-eng/football-lab-v3./actions/workflows/train-research.yml)")
        return
    if status=="failed":
        st.error("No league trained successfully. Inspect GitHub Actions logs and retry.")
    elif status=="partial":
        st.warning("Training completed for some leagues; other datasets or advanced metrics were unavailable.")
    else:st.success("The research training report was published.")
    st.caption(f"Report generated: {report.get('generated_at','unknown')} UTC · Live model promoted: {'yes' if report.get('live_model_promoted') else 'NO'}")
    st.warning(report.get("caution","Retrospective benchmarks are not equivalent to prospective validation."))
    if report.get("errors"):
        with st.expander("Unavailable leagues and retrieval errors"):
            st.json(report["errors"])
    results=report.get("leagues",{})
    if results:
        st.subheader("Rolling-season evaluation")
        league=st.selectbox("League",list(results),key="trained_league")
        info=results[league]
        st.caption(f'{info.get("historical_matches",0)} historical games, from {info.get("historical_first","?")} to {info.get("historical_last","?")}')
        folds=info.get("folds",[])
        rows=[]
        for fold in folds:
            base=fold["baseline_test"]
            best=fold["selected_test"]
            rows.append(dict(
                year=f'{fold["test_season"]}/{str(fold["test_season"]+1)[-2:]}',
                training_through=fold["training_through"],
                validation=fold["validation_season"],
                test_matches=fold["test_matches"],
                selected=fold["selected_by_validation"],
                accuracy=best["accuracy"],
                log_loss=best["log_loss"],
                elo_log_loss=base["log_loss"],
                brier=best["brier"],
                elo_brier=base["brier"],
                strong_60_count=fold["selected_60_plus"]["matches"],
                strong_60_hit_rate=fold["selected_60_plus"]["hit_rate"],
                upset_alerts=fold["selected_upset_vs_elo"]["alerts"],
                upset_wins=fold["selected_upset_vs_elo"]["captured"],
            ))
        if rows:
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        if folds:
            season=st.selectbox("Inspect year",list(range(len(folds))),
                format_func=lambda i:f"{folds[i]['test_season']}/{str(folds[i]['test_season']+1)[-2:]}",
                key="inspect_year")
            fold=folds[season]
            st.write("**Models compared on precisely the same matches**")
            table=[]
            for x in fold["competitors"]:
                table.append(dict(model=x["model"],temperature=x["temperature"],
                    val_log_loss=x["validation"]["log_loss"],
                    test_log_loss=x["test"]["log_loss"],
                    test_brier=x["test"]["brier"],
                    test_accuracy=x["test"]["accuracy"]))
            st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
            st.caption("Model selection and probability softening used validation results only. Later seasons were scored as retrospective diagnostics and do not certify future superiority.")
    else:
        st.info("No league reports have been published yet.")
    st.subheader("Free advanced xG experiment")
    advanced=report.get("advanced",{})
    if advanced.get("status")=="completed":
        st.write(f"**Exactly matched advanced-data matches:** {advanced['matched_matches']}")
        st.write(f"**Selected by earlier validation:** {advanced['selected_on_validation']}")
        st.dataframe(pd.DataFrame(advanced["comparison"]),hide_index=True,use_container_width=True)
        st.caption(advanced.get("note","Advanced metrics have not been promoted to live forecasts."))
    elif advanced.get("status")=="unavailable":
        st.warning(f"Advanced data wasn't available or matched insufficiently: {advanced.get('reason','Unknown source failure')}")
        st.caption("The ordinary results-based training is still useful. Missing xG/PPDA values were not fabricated.")
    else:st.caption("Advanced metrics have not been tested in this report.")
