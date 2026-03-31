import os
import json
import requests
from datetime import datetime, timedelta
import math
import google.generativeai as genai

# ── KEYS ──
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
FOOTBALL_KEY = os.environ.get("FOOTBALL_API_KEY")

# ── SETUP GEMINI ──
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ── DATES ──
today = datetime.utcnow()
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)

def fmt(d): return d.strftime("%Y-%m-%d")

HEADERS = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": FOOTBALL_KEY
}

BASE = "https://v3.football.api-sports.io"

LEAGUE_MAP = {
    39: "pl",
    140: "laliga",
    2: "ucl",
    78: "bundesliga",
    135: "seriea",
    61: "ligue1"
}

ROLE_FACTORS = {
    "Attacker": 1.25,
    "Midfielder": 0.85,
    "Defender": 0.25,
    "Goalkeeper": 0.05
}

def get_fixtures(date_str):
    r = requests.get(f"{BASE}/fixtures", headers=HEADERS, params={"date": date_str, "season": 2024})
    data = r.json()
    return data.get("response", [])

def calc_xsot(avg_shots, role, team_attack, opp_defense, minutes):
    role_mult = ROLE_FACTORS.get(role, 1.0)
    team_mult = 0.6 + team_attack * 0.8
    opp_mult = 1.0 - opp_defense * 0.5
    min_mult = 1.0 if minutes >= 80 else 0.85 if minutes >= 60 else 0.6
    return round(avg_shots * role_mult * team_mult * opp_mult * min_mult, 2)

def calc_prob(xsot, line=0.5):
    lam = xsot
    if line == 0.5:
        p0 = math.exp(-lam)
        return min(0.97, max(0.03, 1 - p0))
    elif line == 1.5:
        p0 = math.exp(-lam)
        p1 = lam * math.exp(-lam)
        return min(0.95, max(0.02, 1 - p0 - p1))
    return 0.3

def calc_ev(prob, odds):
    return round((prob * odds) - 1, 3)

def get_verdict(prob, conf, ev, high_variance):
    if high_variance:
        return "AVOID"
    if prob >= 0.75 and conf >= 8 and ev > 0:
        return "SAFE"
    if prob >= 0.65 or conf >= 6:
        return "RISKY"
    return "AVOID"

def process_fixtures(fixtures, day_label):
    matches = []
    count = 0
    for fix in fixtures:
        league_id = fix["league"]["id"]
        if league_id not in LEAGUE_MAP:
            continue
        if count >= 6:
            break
        count += 1

        home = fix["teams"]["home"]["name"]
        away = fix["teams"]["away"]["name"]
        kickoff = fix["fixture"]["date"][11:16]
        fixture_id = fix["fixture"]["id"]
        league_key = LEAGUE_MAP[league_id]

        players = []
        for team_att, team_def in [(0.80, 0.70), (0.75, 0.68)]:
            r = requests.get(f"{BASE}/players/topscorers", headers=HEADERS,
                           params={"league": league_id, "season": 2024})
            top = r.json().get("response", [])[:3]
            for p in top:
                pname = p["player"]["name"]
                role = "Attacker"
                if p.get("statistics"):
                    pos = p["statistics"][0]["games"].get("position", "Attacker")
                    role = pos if pos else "Attacker"
                goals = 0
                apps = 1
                minutes = 80
                if p.get("statistics"):
                    goals = p["statistics"][0]["goals"]["total"] or 0
                    apps = p["statistics"][0]["games"]["appearences"] or 1
                    minutes = p["statistics"][0]["games"].get("minutes", 80) or 80
                    minutes = min(90, max(45, minutes // max(apps, 1)))

                avg_shots = max(0.5, goals / max(apps, 1) * 2.5)
                line = 1.5 if avg_shots > 1.5 else 0.5
                xsot = calc_xsot(avg_shots, role, team_att, team_def, minutes)
                prob = calc_prob(xsot, line)
                conf = min(10, max(1, round(prob * 10 + (1 if minutes >= 80 else 0))))
                odds = round(1.5 + (1 - prob) * 1.5, 2)
                ev = calc_ev(prob, odds)
                implied = round(1 / odds, 3)
                is_value = (prob - implied) > 0.05
                verdict = get_verdict(prob, conf, ev, False)

                role_short = "ST" if "Forward" in role or "Attacker" in role else "CAM" if "Midfielder" in role else "CB"

                players.append({
                    "name": pname,
                    "role": role_short,
                    "minutes": minutes,
                    "line": line,
                    "xsot": xsot,
                    "prob": round(prob, 3),
                    "conf": conf,
                    "oddsCurrent": odds,
                    "oddsOpen": round(odds + 0.05, 2),
                    "ev": ev,
                    "isValue": is_value,
                    "verdict": verdict,
                    "status": "confirmed",
                    "form": [1 if i % 2 == 0 else 0 for i in range(6)]
                })

        matches.append({
            "id": fixture_id,
            "league": league_key,
            "day": day_label,
            "homeTeam": home,
            "awayTeam": away,
            "kickoff": kickoff,
            "homeAttack": 0.80,
            "awayAttack": 0.75,
            "homeDefense": 0.70,
            "awayDefense": 0.68,
            "isLive": False,
            "isHighVariance": False,
            "players": players[:6]
        })

    return matches

def get_ai_insight(matches):
    if not matches:
        return "No matches today for analysis."
    summary = []
    for m in matches[:3]:
        safe = [p["name"] for p in m["players"] if p["verdict"] == "SAFE"]
        summary.append(f"{m['homeTeam']} vs {m['awayTeam']}: Safe picks: {', '.join(safe) if safe else 'None'}")

    prompt = f"""You are a football shots-on-target betting analyst.
Today's top matches: {'. '.join(summary)}
Give a 2-3 sentence sharp insight about the best value picks today and any risks to watch.
Be specific and confident. No fluff."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"AI insight unavailable: {str(e)}"

# ── MAIN ──
print("Fetching fixtures...")
all_matches = []

all_matches += process_fixtures(get_fixtures(fmt(yesterday)), "yesterday")
all_matches += process_fixtures(get_fixtures(fmt(today)), "today")
all_matches += process_fixtures(get_fixtures(fmt(tomorrow)), "tomorrow")

print(f"Got {len(all_matches)} matches")

today_matches = [m for m in all_matches if m["day"] == "today"]
ai_insight = get_ai_insight(today_matches)

output = {
    "updated": datetime.utcnow().isoformat() + "Z",
    "aiInsight": ai_insight,
    "matches": all_matches
}

with open("data.json", "w") as f:
    json.dump(output, f, indent=2)

print("data.json written successfully!")
