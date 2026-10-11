"""Current-season fixtures and basic-form forecasts from football-data.org.

These forecasts are a separate, unvalidated Poisson baseline: NOT the StatsBomb
event-trained model. No xG, pressing, injuries, or player availability is inferred.
"""
from datetime import datetime, timedelta, timezone
from math import exp, factorial
import requests

API_BASE = "https://api.football-data.org/v4/"
LEAGUES = {
    "Premier League · England": "PL",
    "La Liga · Spain": "PD",
    "Bundesliga · Germany": "BL1",
    "Serie A · Italy": "SA",
    "Ligue 1 · France": "FL1",
    "Eredivisie · Netherlands": "DED",
    "Primeira Liga · Portugal": "PPL",
}

class FootballDataError(Exception):
    pass

def fetch_season_matches(token, league_code):
    """One API request per league; callers must cache results to respect free limits."""
    if league_code not in LEAGUES.values():
        raise FootballDataError("Unknown league code")
    try:
        response = requests.get(
            API_BASE + f"competitions/{league_code}/matches",
            headers={"X-Auth-Token": token, "Accept": "application/json"},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise FootballDataError(f"Network connection failed: {type(exc).__name__}") from exc
    if response.status_code == 401:
        raise FootballDataError("Token rejected (401). Check Streamlit Secrets.")
    if response.status_code == 403:
        raise FootballDataError("This league is unavailable on your free plan (403). Choose another.")
    if response.status_code == 429:
        raise FootballDataError("Free API rate limit reached (429). Wait a minute before trying again.")
    if response.status_code >= 400:
        raise FootballDataError(f"Provider returned HTTP {response.status_code}.")
    data = response.json()
    if not isinstance(data.get("matches"), list):
        raise FootballDataError("Provider returned no usable match list.")
    return {"matches": data["matches"], "fetched": datetime.now(timezone.utc).isoformat()}

def kickoff(match):
    try:
        return datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00"))
    except (KeyError, AttributeError, ValueError, TypeError):
        return None

def scoreline(match):
    score = (match.get("score") or {}).get("fullTime") or {}
    home, away = score.get("home"), score.get("away")
    if isinstance(home, int) and isinstance(away, int):
        return (home, away)
    return None

def historical_results(matches, before):
    # For a forecast, never use fixtures played on/after its kickoff.
    results = []
    for match in matches:
        time = kickoff(match)
        if (match.get("status") == "FINISHED" and time
                and time < before and scoreline(match)):
            results.append(match)
    return sorted(results, key=kickoff)

def next_fixtures(matches, now=None, horizon_days=21):
    now = now or datetime.now(timezone.utc)
    until = now + timedelta(days=horizon_days)
    eligible = []
    for m in matches:
        dt = kickoff(m)
        if dt and now <= dt <= until and m.get("status") in ("SCHEDULED", "TIMED"):
            if (m.get("homeTeam") or {}).get("id") and (m.get("awayTeam") or {}).get("id"):
                eligible.append(m)
    return sorted(eligible, key=kickoff)

def team_record(team_id, matches, n=8):
    relevant = []
    for m in matches:
        home = (m.get("homeTeam") or {}).get("id")
        away = (m.get("awayTeam") or {}).get("id")
        result = scoreline(m)
        if not result: continue
        a, b = result
        if team_id == home:
            relevant.append((a, b))
        elif team_id == away:
            relevant.append((b, a))
    return relevant[-n:]

def forecast_fixture(fixture, matches):
    """A transparent basic Poisson baseline. Not trained or performance-validated."""
    dt = kickoff(fixture)
    h = (fixture.get("homeTeam") or {}).get("id")
    a = (fixture.get("awayTeam") or {}).get("id")
    if not dt or not h or not a or h == a:
        return None
    past = historical_results(matches, dt)
    hs, as_ = team_record(h, past), team_record(a, past)
    if min(len(hs), len(as_)) < 5:
        return {"insufficient": True, "home_samples": len(hs), "away_samples": len(as_)}
    scored = [scoreline(m) for m in past]
    avg_home = sum(s[0] for s in scored) / len(scored) if scored else 1.45
    avg_away = sum(s[1] for s in scored) / len(scored) if scored else 1.18
    avg_home = max(.3, avg_home)
    avg_away = max(.3, avg_away)
    # Shrink team form to same-league averages to avoid excessive confidence.
    def smoothed_rate(values, baseline):
        return (sum(values) + 4 * baseline) / (len(values) + 4)
    h_attack = smoothed_rate([x[0] for x in hs], avg_home)
    h_defense = smoothed_rate([x[1] for x in hs], avg_away)
    a_attack = smoothed_rate([x[0] for x in as_], avg_away)
    a_defense = smoothed_rate([x[1] for x in as_], avg_home)
    lam_h = max(.25, min(3.5, (h_attack + a_defense) / 2))
    lam_a = max(.25, min(3.5, (a_attack + h_defense) / 2))
    ph = [exp(-lam_h)*lam_h**k/factorial(k) for k in range(11)]
    pa = [exp(-lam_a)*lam_a**k/factorial(k) for k in range(11)]
    probs = [0., 0., 0.]
    scores = []
    for i in range(11):
        for j in range(11):
            p = ph[i] * pa[j]
            probs[0 if i > j else 1 if i == j else 2] += p
            scores.append((p, f"{i}-{j}"))
    probs = [x / sum(probs) for x in probs]
    return {
        "insufficient": False,
        "home_samples": len(hs), "away_samples": len(as_),
        "home_xg": lam_h, "away_xg": lam_a,
        "p_home": probs[0], "p_draw": probs[1], "p_away": probs[2],
        "top_score": max(scores)[1],
        "pick": ("1", "X", "2")[probs.index(max(probs))],
        "note": "Basic goals-and-form Poisson baseline, not the advanced event-trained model."
    }
