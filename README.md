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
