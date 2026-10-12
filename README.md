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


## V3.3 — Mobile weekly scout

Open the **existing Streamlit app** on a phone, then select leagues under **Weekly scan settings** and tap **Build / refresh weekly brief**. The mobile-first landing page has:
- **🏠 Brief**: strongest upcoming estimates, upset-radar signals and next fixtures;
- **⭐ 60%+ picks**: only the most likely individual 1/X/2 outcomes at or above the chosen threshold (60% default). This is *not* a bet recommendation or guaranteed accuracy;
- **⚡ Upset watch**: research-only ML challenger signals with at least +7 percentage points over Elo for the underdog and at least a 27% ML underdog probability, independent of which ordinary 1/X/2 engine won validation. Signals are not bookmaker value or proof of an edge;
- **📅 Matches**: all successfully modeled matches, kickoff shown in SAST, and expandable source/engine information;
- **🔬 Research**: the original historical model, advanced xG experiment, event-data research, retrospective tests and exports are retained unchanged under a secondary section.

Runs within the football-data.org free API constraints. Default scan is five major European leagues for the next seven days; settings also offer 14 days or other supported leagues. One chronological team-state pass per league makes many upcoming predictions without re-training each fixture. The underlying per-league classifiers and API snapshots are cached; pressing **refresh** updates forecasts based on the source snapshot when the API cache expires. API errors and missing historical club matches are explicitly reported; those fixtures are excluded rather than guessed. Weekly shortlists persist for the current Streamlit browser session, not permanently across app restarts; V3.3 does NOT claim background automation or push notifications.

The responsive color system uses Streamlit's built-in dark theme with custom midnight/lime styling. Add the hosted URL to an iPhone Home Screen in Safari (Share → Add to Home Screen) for an app-like shortcut. No native iOS binary or App Store publication is claimed.

**Data caveat:** the live shortlisted probabilities are still the results-based Elo/form model selected on earlier validation. Understat's xG/PPDA research is not blended into live suggestions until it is verified on genuinely new results. Historical probability estimates are not verified betting odds or financial advice.


## V3.4 — Prediction tracker and monochrome interface

The mobile dashboard now has a Tracker tab. Each time Build / refresh weekly brief is pressed, the app records pre-kickoff match forecasts with snapshot time, all three original 1/X/2 probabilities, model version, strong-60% status and separate research-only upset signal. Re-scans NEVER overwrite an earlier saved probability. Finished matches can be checked against football-data.org by exact fixture ID and FINISHED full-time score. The tracker shows week/month accuracy, 60%+ hit rate, upset-alert hit rate, pending games, multiclass log loss, Brier score and emerging confidence calibration. CSV and JSON backups can be downloaded.

**Important limitation:** the local SQLite file on free Streamlit hosting is temporary and not a private account; it can disappear on app resets and may be shared between public app visitors. The optional GitHub Actions cloud archive avoids data loss after that one-time setup. Archive records are public in the GitHub repository, but contain only forecast data, not API credentials.

### Enable permanent automated daily records (one-time setup)

1. In GitHub, open this repository, then Settings → Secrets and variables → Actions → New repository secret.
2. Create a secret named FOOTBALL_DATA_API_TOKEN and insert the same football-data.org token already saved in Streamlit. NEVER commit this value to the repo or put it in a message.
3. On the GitHub Actions page, run the workflow named Archive Football Lab predictions once. It will use the GitHub-provided GITHUB_TOKEN to push prediction data to data/tracker_archive.json, subject to Actions permissions.
4. GitHub Actions then runs daily at 07:00 UTC (09:00 South Africa time) if Actions remain enabled. The read-only Streamlit tracker merges this public archive with local data. Free sources and scheduled task execution are not guaranteed; export an occasional manual backup.

The tracker stores snapshots only before kickoff and grades them only when the exact fixture's full-time score is present. It does not automatically backfill predictions for past matches. It is not a betting-profit tracker, and 60% probability is not a 60% proven success rate.

### Visual changes

A consistent monochrome SVG icon set replaces the colourful emoji in navigation, match alerts and hero. The browser icon uses a monochrome Material glyph until the user selects the official logo. The original historical research screens remain available from Research, with text-only secondary tabs.


## V3.6 — Faster iPhone navigation

The six white-icon navigation tiles now use browser-side Streamlit tabs with a fixed 3-column × 2-row mobile layout, rather than six buttons that called the server on every tap. This makes switching sections client-side once the dashboard is loaded. The trade-off is a slightly larger initial UI render, so the Matches view starts with 12 cards and lets users request more. The Tracker no longer automatically blocks tab rendering with a GitHub HTTP request: use **Sync cloud archive** inside Tracker when needed. Optional StatsBomb competition discovery starts only after tapping **Load historical event seasons** inside Research, and the full research workbench is opt-in. The normal prediction/accuracy engine and saved tracker records are unchanged. Since tab CSS depends on Streamlit's frontend markup, visual deployment testing on iPhone is still required.
