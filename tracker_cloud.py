"""Public read-only GitHub archive for durable, scheduled prediction history.

A GitHub Actions job writes this if the owner configures FOOTBALL_DATA_API_TOKEN
as a GitHub Actions secret. No unauthenticated write is ever attempted here.
"""
import json
import requests

ARCHIVE_URL="https://raw.githubusercontent.com/rapelangbotsane-eng/football-lab-v3./main/data/tracker_archive.json"

def fetch_cloud_archive(timeout=12):
    try:
        res=requests.get(ARCHIVE_URL,timeout=timeout,
            headers={"User-Agent":"FootballLabTracker/3.4","Accept":"application/json"})
        res.raise_for_status()
        rows=res.json()
        if not isinstance(rows,list):return [],"Cloud archive returned an unexpected format"
        return [r for r in rows if isinstance(r,dict) and
                all(k in r for k in ("key","archived_at","kickoff","probabilities","pick","league_code"))],None
    except (requests.RequestException,ValueError) as error:
        return [],f"Could not retrieve the cloud archive ({type(error).__name__})"
