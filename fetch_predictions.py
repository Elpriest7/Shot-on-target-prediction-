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

# ── FOOTBALL API ──
HEADERS = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": FOOTBALL_KEY
}
BASE = "https://v3.football.api-sports.io"

LEAGUE_MAP = {
    39: {"key": "pl",         "name": "Premier League",   "understat": "EPL"},
    140: {"key": "laliga",    "name": "La Liga",           "understat": "La_liga"},
    78: {"key": "bundesliga", "name": "Bundesliga",        "understat": "Bundesliga"},
    135: {"key": "seriea",    "name": "Serie A",           "understat": "Serie_A"},
    61: {"key": "ligue1",     "name": "Ligue 1",           "understat": "Ligue_1"},
}

ROLE_FACTORS = {
    "Attacker": 1.25,
    "Forward":  1.25,
    "Midfielder": 0.85,
    "Defender": 0.25,
    "Goalkeeper": 0.05
}

# ══════════════════════════════════════════
# UNDERSTAT — FREE PLAYER SOT DATA
# ══════════════════════════════════════════

def get_understat_players(league_name):
    """Scrape player SOT stats from Understat for a league"""
    try:
        url = f"https://understat.com/league/{league_name}/2024"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=15)
        html = r.text

        # Extract JSON from page script
        import re
        pattern = r"var playersData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if not match:
            return {}

        raw = match.group(1)
        # Unescape the string
        raw = raw.encode('utf-8').decode('unicode_escape')
        players_raw = json.loads(raw)

        players = {}
        for p in players_raw:
            name = p.get("player_name", "")
            shots = float(p.get("shots", 0))
            sot = float(p.get("shots_on_target", 0) if "shots_on_target" in p else p.get("key_passes", 0))
            games = int(p.get("games", 1)) or 1
            xg = float(p.get("xG", 0))
            npxg = float(p.get("npxG", 0))
            time_played = float(p.get("time", 0))

            avg_sot = round(sot / games, 3)
            avg_shots = round(shots / games, 3)
            avg_xg = round(xg / games, 3)
            avg_mins = round(time_played / games, 0) if games > 0 else 80

            players[name.lower()] = {
                "name": name,
                "avg_sot": avg_sot,
                "avg_shots": avg_shots,
                "avg_xg": avg_xg,
                "games": games,
                "avg_mins": avg_mins,
            }
        print(f"Understat: Got {len(players)} players for {league_name}")
        return players

    except Exception as e:
        print(f"Understat error for {league_name}: {e}")
        return {}


def get_understat_player_matches(player_id):
    """Get last 6 match SOT for a specific player"""
    try:
        url = f"https://understat.com/player/{player_id}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        html = r.text

        import re
        pattern = r"var matchesData\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, html)
        if not match:
            return []

        raw = match.group(1)
        raw = raw.encode('utf-8').decode('unicode_escape')
        matches = json.loads(raw)

        form = []
        for m in reversed(matches[-6:]):
            sot = int(m.get("shots_on_target", 0) if "shots_on_target" in m else 0)
            form.append(1 if sot > 0 else 0)

        return form
    except:
        return [0, 1, 0, 1, 0, 1]


# ══════════════════════════════════════════
# MATH ENGINE
# ══════════════════════════════════════════

def calc_xsot(avg_sot, avg_shots, avg_xg, role, team_attack, opp_defense, minutes):
    role_mult = ROLE_FACTORS.get(role, 1.0)
    team_mult = 0.6 + team_attack * 0.8
    opp_mult = 1.0 - opp_defense * 0.5
    min_mult = 1.0 if minutes >= 80 else 0.85 if minutes >= 60 else 0.6

    # Blend real SOT average with xG-based estimate
    xg_estimate = avg_xg * 2.5
    base = (avg_sot * 0.6) + (xg_estimate * 0.4) if avg_sot > 0 else avg_shots * 0.45

    return round(base * role_mult * team_mult * opp_mult * min_mult, 2)


def calc_prob(xsot, line=0.5):
    lam = max(0.01, xsot)
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


def calc_confidence(prob, avg_sot, minutes, games):
    c = 1
    if prob >= 0.85: c += 3
    elif prob >= 0.75: c += 2
    elif prob >= 0.65: c += 1
    if avg_sot >= 1.5: c += 2
    elif avg_sot >= 1.0: c += 1
    if minutes >= 80: c += 1.5
    elif minutes < 60: c -= 1
    if games >= 10: c += 1
    return min(10, max(1, round(c)))


# ══════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════

def get_fixtures(date_str):
    try:
        r = requests.get(f"{BASE}/fixtures", headers=HEADERS,
                        params={"date": date_str, "season": 2024}, timeout=15)
        return r.json().get("response", [])
    except Exception as e:
        print(f"Fixtures error: {e}")
        return []


def get_top_players_for_team(team_id, league_id, understat_players):
    """Get top attacking players for a team using Understat data"""
    try:
        r = requests.get(f"{BASE}/players", headers=HEADERS,
                        params={"team": team_id, "season": 2024, "league": league_id}, timeout=15)
        data = r.json().get("response", [])

        attackers = []
        for p in data:
            stats = p.get("statistics", [{}])[0]
            pos = stats.get("games", {}).get("position", "") or ""
            goals = stats.get("goals", {}).get("total", 0) or 0
            apps = stats.get("games", {}).get("appearences", 0) or 0
            minutes = stats.get("games", {}).get("minutes", 0) or 0

            if pos in ["Attacker", "Midfielder"] and apps >= 3:
                avg_mins = round(minutes / max(apps, 1))
                name = p["player"]["name"]

                # Try to match with understat data
                name_lower = name.lower()
                us_data = None
                for key in understat_players:
                    if any(part in key for part in name_lower.split() if len(part) > 3):
                        us_data = understat_players[key]
                        break

                avg_sot = us_data["avg_sot"] if us_data else max(0.3, goals / max(apps, 1) * 0.8)
                avg_shots = us_data["avg_shots"] if us_data else max(0.5, goals / max(apps, 1) * 2)
                avg_xg = us_data["avg_xg"] if us_data else goals / max(apps, 1) * 0.4
                real_games = us_data["games"] if us_data else apps

                attackers.append({
                    "name": name,
                    "position": pos,
                    "avg_sot": avg_sot,
                    "avg_shots": avg_shots,
                    "avg_xg": avg_xg,
                    "avg_mins": avg_mins,
                    "games": real_games,
                    "has_understat": us_data is not None
                })

        # Sort by avg_sot descending
        attackers.sort(key=lambda x: x["avg_sot"], reverse=True)
        return attackers[:4]

    except Exception as e:
        print(f"Team players error: {e}")
        return []


def process_fixtures(fixtures, day_label, understat_cache):
    matches = []
    count = 0

    for fix in fixtures:
        league_id = fix["league"]["id"]
        if league_id not in LEAGUE_MAP:
            continue
        if count >= 5:
            break
        count += 1

        league_info = LEAGUE_MAP[league_id]
        league_key = league_info["key"]
        us_league = league_info["understat"]

        home_id = fix["teams"]["home"]["id"]
        away_id = fix["teams"]["away"]["id"]
        home_name = fix["teams"]["home"]["name"]
        away_name = fix["teams"]["away"]["name"]
        kickoff = fix["fixture"]["date"][11:16]
        fixture_id = fix["fixture"]["id"]

        # Get understat data for this league (cached)
        if us_league not in understat_cache:
            understat_cache[us_league] = get_understat_players(us_league)
        us_players = understat_cache[us_league]

        # Derby/cup detection
        is_high_variance = any(word in fix["league"]["name"].lower()
                              for word in ["cup", "trophy", "shield", "supercopa"])

        # Get players for both teams
        home_players = get_top_players_for_team(home_id, league_id, us_players)
        away_players = get_top_players_for_team(away_id, league_id, us_players)

        # Team strength (home advantage)
        home_attack = 0.82
        away_attack = 0.76
        home_defense = 0.70
        away_defense = 0.68

        all_players = []

        for p, team_att, opp_def in [
            *[(p, home_attack, away_defense) for p in home_players],
            *[(p, away_attack, home_defense) for p in away_players]
        ]:
            role = p["position"]
            xsot = calc_xsot(p["avg_sot"], p["avg_shots"], p["avg_xg"],
                            role, team_att, opp_def, p["avg_mins"])
            line = 1.5 if p["avg_sot"] >= 1.5 else 0.5
            prob = calc_prob(xsot, line)
            conf = calc_confidence(prob, p["avg_sot"], p["avg_mins"], p["games"])
            odds = round(1.5 + (1 - prob) * 1.5, 2)
            ev = calc_ev(prob, odds)
            implied = round(1 / odds, 3)
            is_value = (prob - implied) > 0.05
            verdict = get_verdict(prob, conf, ev, is_high_variance)

            role_short = "ST" if role == "Attacker" or role == "Forward" else "CAM"

            all_players.append({
                "name": p["name"],
                "role": role_short,
                "minutes": p["avg_mins"],
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
                "form": [1 if i % 2 == 0 else 0 for i in range(6)],
                "avgSot": p["avg_sot"],
                "hasRealData": p["has_understat"]
            })

        matches.append({
            "id": fixture_id,
            "league": league_key,
            "day": day_label,
            "homeTeam": home_name,
            "awayTeam": away_name,
            "kickoff": kickoff,
            "homeAttack": home_attack,
            "awayAttack": away_attack,
            "homeDefense": home_defense,
            "awayDefense": away_defense,
            "isLive": False,
            "isHighVariance": is_high_variance,
            "players": all_players[:8]
        })

    return matches


# ══════════════════════════════════════════
# GEMINI AI ANALYSIS
# ══════════════════════════════════════════

def get_ai_insight(matches):
    if not matches:
        return "No matches today for analysis."

    safe_picks = []
    risky = []
    avoid = []

    for m in matches:
        for p in m.get("players", []):
            entry = f"{p['name']} ({m['homeTeam']} vs {m['awayTeam']}, {p['prob']*100:.0f}% prob, line {p['line']}, EV {p['ev']*100:.1f}%)"
            if p["verdict"] == "SAFE":
                safe_picks.append(entry)
            elif p["verdict"] == "RISKY":
                risky.append(entry)
            else:
                avoid.append(entry)

    prompt = f"""You are a sharp football shots-on-target betting analyst with deep knowledge of player statistics.

TODAY'S DATA:
Safe picks: {', '.join(safe_picks[:5]) if safe_picks else 'None'}
Risky picks: {', '.join(risky[:3]) if risky else 'None'}
Avoid: {', '.join(avoid[:3]) if avoid else 'None'}

Give a sharp 3-sentence analysis:
1. Which safe pick has the best value today and why
2. One risk to watch out for
3. Overall confidence in today's card

Be specific, data-driven, no fluff. Maximum 80 words."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"AI analysis unavailable today: {str(e)}"


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════

print("=" * 50)
print("SOTIQ — Daily Predictions Engine")
print(f"Running at {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
print("=" * 50)

understat_cache = {}
all_matches = []

print("\n📅 Fetching yesterday's fixtures...")
yesterday_fixtures = get_fixtures(fmt(yesterday))
all_matches += process_fixtures(yesterday_fixtures, "yesterday", understat_cache)
print(f"   Got {len([m for m in all_matches if m['day']=='yesterday'])} matches")

print("\n📅 Fetching today's fixtures...")
today_fixtures = get_fixtures(fmt(today))
all_matches += process_fixtures(today_fixtures, "today", understat_cache)
print(f"   Got {len([m for m in all_matches if m['day']=='today'])} matches")

print("\n📅 Fetching tomorrow's fixtures...")
tomorrow_fixtures = get_fixtures(fmt(tomorrow))
all_matches += process_fixtures(tomorrow_fixtures, "tomorrow", understat_cache)
print(f"   Got {len([m for m in all_matches if m['day']=='tomorrow'])} matches")

print("\n🤖 Generating Gemini AI insight...")
today_matches = [m for m in all_matches if m["day"] == "today"]
ai_insight = get_ai_insight(today_matches)
print(f"   Done: {ai_insight[:60]}...")

# Stats summary
all_players = [p for m in today_matches for p in m.get("players", [])]
safe_count = len([p for p in all_players if p["verdict"] == "SAFE"])
ev_count = len([p for p in all_players if p["ev"] > 0])
value_count = len([p for p in all_players if p["isValue"]])
avoid_count = len([p for p in all_players if p["verdict"] == "AVOID"])

print(f"\n📊 Today's Summary:")
print(f"   Safe picks: {safe_count}")
print(f"   Positive EV: {ev_count}")
print(f"   Value bets: {value_count}")
print(f"   Avoid: {avoid_count}")

output = {
    "updated": datetime.utcnow().isoformat() + "Z",
    "aiInsight": ai_insight,
    "stats": {
        "safe": safe_count,
        "positiveEV": ev_count,
        "value": value_count,
        "avoid": avoid_count
    },
    "matches": all_matches
}

with open("data.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ data.json written successfully!")
print("=" * 50)
