# Football Lab V3 — Advanced Match Intelligence

A free-data football research prototype with a Streamlit mobile-friendly interface.

## Launch on Streamlit Community Cloud
1. Sign in to https://share.streamlit.io with GitHub.
2. Choose **Create app**, repository **rapelangbotsane-eng/football-lab-v3.** (note the final period), branch **main**, main file **app.py**.
3. If the private repository does not appear, grant Streamlit access to it through GitHub authorisation.
4. Deploy, open the resulting website on your iPhone, then Safari → Share → Add to Home Screen.
5. Open **Data & training**, select one or more historical StatsBomb seasons (80+ event-complete games total), and press **Sync event data + train model**. Results appear on the Backtest tab.

The app automatically downloads selected historical event datasets when the button is pressed. Hosting requires internet connectivity, and free-tier resource limits apply.

## What it actually measures
- Provider shot-level expected goals (xG) and xG conceded, high-quality chances
- Progressive passes/carries, box and final-third entries
- Pressure events, approximate pressure recoveries and PPDA proxy
- Chronological pre-match features, Elo, rolling recency
- 1/X/2 classifier, estimated goal rates, 10,000 probabilistic score simulations
- Held-out accuracy, log loss, Brier score and experimental upset flag

## Essential limitations
This is **historical research software**. StatsBomb's freely available data covers selected matches and seasons; it does **not** supply live fixtures, comprehensive current-league event data, injuries, lineups, or official PPDA. The prototype's upset-detection value and real match forecasting accuracy have **not yet been verified** on independent football match data. Pressing features are estimated and the goal simulation is statistical, not a possession-by-possession digital twin.

Data credits: https://github.com/hudl/open-data


## Connect the free football-data.org token (current fixtures)
1. Open https://share.streamlit.io/ and choose the Football Lab app.
2. Open Manage app → Settings → Secrets.
3. Add a TOML entry (replace the placeholder with your own token):
   `FOOTBALL_DATA_API_TOKEN = "YOUR_TOKEN_HERE"`
4. Save and let Streamlit reload the GitHub code.
5. Open the **Upcoming fixtures** tab, choose a league, and tap **Load current fixtures**.

Do not commit the real token to this public repository. Free football-data.org accounts are rate-limited (10 requests/minute). The app caches match lists for 30 minutes and requests only one selected league at a time.

**Forecast limitation:** The Upcoming fixtures tab makes *experimental Poisson goals/form estimates*, not forecasts from the advanced StatsBomb event-trained machine-learning model. This free API does not provide live xG, PPDA, ball progression, injuries, or lineups. These experimental probabilities have not been calibrated against held-out club-football results. Models will not be silently conflated.


## V3.1 — Men's club-football current predictions

Use the **⚽ Men's club predictions** tab and select a league. Press **Load & analyse matches** once; the app retrieves six historical seasons (2020/21–2025/26) from football-data.co.uk, learns a goals/form/Elo club model, evaluates 2025/26 without training on it, retrieves current fixtures and past results from the configured football-data.org free API, and estimates 1/X/2 for upcoming games.

### Rigorous evaluation design
- Train on seasons 2020/21–2023/24.
- Select the temperature calibration and best of trained-ML versus Elo-only using **2024/25 validation matches**.
- Report the winner's **2025/26 holdout** accuracy, log loss, Brier score, and compare both candidate models.
- Evaluate experimental alerts for Elo-underdog victories against historic results. Display false alarms.
- Only after evaluation, refit the final classifier on all 2020/21–2025/26 records for upcoming 2026/27 fixtures; form features are built chronologically.
- Never use a completed fixture on the same calendar date in that day's forecasts.
- Explicit alias mapping: if a live team cannot be reliably joined to its historical results, refuse to forecast rather than guessing.

### Data sources and limitations
Historical results: https://www.football-data.co.uk/downloadm.php (free historical CSV files, credited to source).
Current fixtures, dates and completed matches: https://www.football-data.org/documentation/quickstart (free plan, access-dependent).
No bookmaker odds, future match results, live StatsBomb events, xG, pressing efficiency, player availability or confirmed lineups are included in V3.1 current-game forecasts. The separate **historical event-data** engine remains available for research; its metrics are NOT silently mixed into live predictions.
Free-source availability, historical quality, missing team aliases and generalization to unseen seasons are still risks. Real current-game forecasting quality is unverified until prospective forecasts are logged and scored.


## V3.2 — Advanced xG / PPDA research experiment

Open the **🔬 Advanced xG experiment** tab and press **Run free Premier League xG experiment**. This is an opt-in research workflow. It fetches one EPL season per year (2020–2025) from the community-documented Understat AJAX feed, extracts xG, xGA, non-penalty xG, expected points, PPDA and deep completions, then matches the events to the existing historical results by *exact fixture date, normalized team names and final score*. Unmatched fixtures remain unfilled; we do NOT fabricate advanced data.

Every rolling advanced metric uses **only matches before the prediction day**. On the same eligible fixtures, the experiment compares four models:
1. Elo only.
2. Elo + basic result/venue features.
3. Elo + result/venue + advanced xG and expected points.
4. Elo + result/venue + xG + PPDA and deep completions.

Historical training: 2020/21–2023/24. Validation and temperature selection: 2024/25. Retrospective diagnosis: 2025/26. **This 2025/26 dataset has previously been studied in V3.1, so it is not a pristine final holdout.** Trust a prospective, timestamped 2026/27 evaluation over retrospective gains. The experiment will *not* switch current 1X2 predictions to the advanced model; those continue to use the prior validated Elo/form model until we prove the new data's availability and value on future results.

Data source: https://understat.com (unofficial read-only feed described by community tooling); see https://soccerdata.readthedocs.io/en/latest/datasources/Understat.html. No guaranteed API or general redistribution rights are asserted. If access fails, stops, or is disallowed, don't bypass restrictions or scrape through protective measures. The existing app works without Understat. All sources should be credited, and any commercial deployment requires a separate licensing review.

### Can models combine many data types?
Yes. They can combine match results, Elo, xG, pressing and progression if they refer to the same real entities, have reliable timestamps, and measure data available BEFORE kickoff. **Historical StatsBomb/Wyscout progression features cannot be silently treated as recent match features.** The next phase would be a source coverage matrix and an event-derived feature stream only if equivalent current and historical coverage exists. More predictors do not guarantee lower error; ablation studies test incremental value.
