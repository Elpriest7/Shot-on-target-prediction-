import os
import json
import requests
from datetime import datetime, timedelta
import math
import re
import google.generativeai as genai

# ── KEYS ──
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
FOOTBALL_KEY = os.environ.get("FOOTBALL_API_KEY")
ODDS_KEY = os.environ.get("ODDS_API_KEY")

# ── SETUP GEMINI ──
genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ── DATES ──
today = datetime.utcnow()
yesterday = today - timedelta(days=1)
tomorrow = today + timedelta(days=1)
def fmt(d): return d.strftime("%Y-%m-%d")

# ── FOOTBALL API ──
F_HEADERS = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": FOOTBALL_KEY
}
BASE = "https://v3.football.api-sports.io"

LEAGUE_MAP = {
    39:  {"key": "pl",         "understat": "EPL",          "odds_key": "soccer_epl"},
    140: {"key": "laliga",     "understat": "La_liga",      "odds_key": "soccer_spain_la_liga"},
    78:  {"key": "bundesliga", "understat": "Bundesliga",   "odds_key": "soccer_germany_bundesliga"},
    135: {"key": "seriea",     "understat": "Serie_A",      "odds_key": "soccer_italy_serie_a"},
    61:  {"key": "ligue1",     "understat": "Ligue_1",      "odds_key": "soccer_france_ligue_one"},
}

ROLE_FACTORS = {
    "Attacker": 1.25, "Forward": 1.25,
    "Midfielder": 0.85, "Defender": 0.25, "Goalkeeper": 0.05
}

US_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ══════════════════════════════════════════
# 1. REAL ODDS — odds-api.io
# ══════════════════════════════════════════

odds_cache = {}

def get_real_odds(sport_key, home_team, away_team):
    """Fetch real bookmaker odds for a match"""
    try:
        if sport_key in odds_cache:
            events = odds_cache[sport_key]
        else:
            url = f"https://api.odds-api.io/v4/sports/{sport_key}/odds"
            params = {
                "apiKey": ODDS_KEY,
                "regions": "uk",
                "markets": "h2h",
                "oddsFormat": "decimal"
            }
            r = requests.get(url, params=params, timeout=15)
            events = r.json() if r.status_code == 200 else []
            odds_cache[sport_key] = events

        # Find matching event
        ht_lower = home_team.lower()
        at_lower = away_team.lower()
        for event in events:
            eh = event.get("home_team", "").lower()
            ea = event.get("away_team", "").lower()
            if any(w in eh for w in ht_lower.split()[:2]) or any(w in ea for w in at_lower.split()[:2]):
                bookmakers = event.get("bookmakers", [])
                if bookmakers:
                    outcomes = bookmakers[0].get("markets", [{}])[0].get("outcomes", [])
                    odds_map = {o["name"].lower(): o["price"] for o in outcomes}
                    return {
                        "home": odds_map.get(ht_lower[:5], 1.90),
                        "away": odds_map.get(at_lower[:5], 2.10),
                        "draw": odds_map.get("draw", 3.40),
                        "found": True
                    }
        return {"home": 1.90, "away": 2.10, "draw": 3.40, "found": False}
    except Exception as e:
        print(f"  Odds error: {e}")
        return {"home": 1.90, "away": 2.10, "draw": 3.40, "found": False}


def generate_player_odds(prob, line, match_odds):
    """Generate realistic SOT line odds based on match odds + probability"""
    # Use match odds to calibrate — higher scoring games = tighter lines
    avg_match_odds = (match_odds["home"] + match_odds["away"]) / 2
    volatility = 1 / avg_match_odds

    # Opening odds (slightly worse for punter)
    fair_odds = round(1 / prob, 2)
    margin = 0.08 + volatility * 0.05  # bookmaker margin
    open_odds = round(fair_odds * (1 - margin * 0.5), 2)
    open_odds = max(1.20, min(open_odds, 4.50))

    # Current odds (simulate movement based on prob)
    drift = (prob - 0.5) * 0.3
    current_odds = round(open_odds - drift + (0.05 - 0.1 * volatility), 2)
    current_odds = max(1.15, min(current_odds, 4.50))

    moved = "up" if current_odds < open_odds else "down" if current_odds > open_odds else "flat"
    pct_move = abs(round((current_odds - open_odds) / open_odds * 100, 1))

    return {
        "open": open_odds,
        "current": current_odds,
        "moved": moved,
        "pctMove": pct_move
    }


# ══════════════════════════════════════════
# 2. LINEUPS & INJURIES — API-Football
# ══════════════════════════════════════════

def get_lineups(fixture_id):
    """Get confirmed lineups for a fixture"""
    try:
        r = requests.get(f"{BASE}/fixtures/lineups",
                        headers=F_HEADERS,
                        params={"fixture": fixture_id},
                        timeout=15)
        data = r.json().get("response", [])
        if not data:
            return {}, {}

        lineups = {}
        for team_lineup in data:
            team_name = team_lineup["team"]["name"]
            starters = [p["player"]["name"] for p in team_lineup.get("startXI", [])]
            substitutes = [p["player"]["name"] for p in team_lineup.get("substitutes", [])]
            lineups[team_name] = {"starters": starters, "subs": substitutes}

        return lineups, True
    except Exception as e:
        print(f"  Lineups error: {e}")
        return {}, False


def get_injuries(fixture_id):
    """Get injury/suspension list for a fixture"""
    try:
        r = requests.get(f"{BASE}/injuries",
                        headers=F_HEADERS,
                        params={"fixture": fixture_id},
                        timeout=15)
        data = r.json().get("response", [])
        injured = {}
        for p in data:
            name = p["player"]["name"]
            reason = p["player"].get("reason", "Injured")
            injured[name.lower()] = reason
        return injured
    except Exception as e:
        print(f"  Injuries error: {e}")
        return {}


def get_player_status(player_name, lineups, injuries, home_team, away_team):
    """Determine if player is confirmed starter, sub, injured or unknown"""
    name_lower = player_name.lower()

    # Check injuries first
    for inj_name, reason in injuries.items():
        if any(part in inj_name for part in name_lower.split() if len(part) > 3):
            return "injured", reason

    # Check lineups
    for team_name, lineup in lineups.items():
        for starter in lineup["starters"]:
            if any(part in starter.lower() for part in name_lower.split() if len(part) > 3):
                return "confirmed", "Starting XI"
        for sub in lineup["subs"]:
            if any(part in sub.lower() for part in name_lower.split() if len(part) > 3):
                return "sub", "On bench"

    return "unknown", "Lineup TBC"


# ══════════════════════════════════════════
# 3. HIT RATE TRACKER
# ══════════════════════════════════════════

def load_history():
    """Load existing prediction history"""
    try:
        if os.path.exists("history.json"):
            with open("history.json", "r") as f:
                return json.load(f)
    except:
        pass
    return {"predictions": [], "summary": {}}


def save_history(history):
    """Save updated history"""
    try:
        with open("history.json", "w") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"  History save error: {e}")


def update_history(yesterday_matches, history):
    """Add yesterday's results to history"""
    today_str = fmt(today)
    already_logged = any(
        p["date"] == fmt(yesterday)
        for p in history.get("predictions", [])
    )
    if already_logged:
        print("  History already updated for yesterday")
        return history

    new_entries = []
    for m in yesterday_matches:
        for p in m.get("players", []):
            if "actualSOT" not in p:
                continue
            new_entries.append({
                "date": fmt(yesterday),
                "player": p["name"],
                "match": f"{m['homeTeam']} vs {m['awayTeam']}",
                "league": m["league"],
                "verdict": p["verdict"],
                "line": p["line"],
                "prob": p["prob"],
                "ev": p["ev"],
                "predictedXSOT": p["xsot"],
                "actualSOT": p["actualSOT"],
                "hit": p.get("hit", False)
            })

    if new_entries:
        history["predictions"].extend(new_entries)
        # Keep last 90 days only
        cutoff = fmt(today - timedelta(days=90))
        history["predictions"] = [
            p for p in history["predictions"] if p["date"] >= cutoff
        ]
        # Recalculate summary
        history["summary"] = calc_history_summary(history["predictions"])
        print(f"  Added {len(new_entries)} results to history")
    return history


def calc_history_summary(predictions):
    """Calculate hit rates by verdict type"""
    if not predictions:
        return {}

    def stats(preds):
        if not preds: return {"hits": 0, "total": 0, "rate": 0, "avgEV": 0}
        hits = len([p for p in preds if p["hit"]])
        avg_ev = round(sum(p["ev"] for p in preds) / len(preds) * 100, 1)
        return {
            "hits": hits,
            "total": len(preds),
            "rate": round(hits / len(preds) * 100),
            "avgEV": avg_ev
        }

    safe_preds  = [p for p in predictions if p["verdict"] == "SAFE"]
    risky_preds = [p for p in predictions if p["verdict"] == "RISKY"]
    all_preds   = predictions

    # ROI calculation (flat 1 unit stake)
    total_staked = len([p for p in predictions if p["verdict"] in ["SAFE","RISKY"]])
    total_return = sum(
        p.get("oddsCurrent", 1.8) if p["hit"] else 0
        for p in predictions if p["verdict"] in ["SAFE","RISKY"]
    )
    roi = round((total_return - total_staked) / max(total_staked, 1) * 100, 1)

    return {
        "safe": stats(safe_preds),
        "risky": stats(risky_preds),
        "all": stats(all_preds),
        "roi": roi,
        "totalPredictions": len(predictions),
        "daysTracked": len(set(p["date"] for p in predictions))
    }


# ══════════════════════════════════════════
# UNDERSTAT
# ══════════════════════════════════════════

def get_understat_league_players(league_name):
    try:
        url = f"https://understat.com/league/{league_name}/2024"
        r = requests.get(url, headers=US_HEADERS, timeout=20)
        html = r.text
        match = re.search(r"var playersData\s*=\s*JSON\.parse\('(.+?)'\)", html)
        if not match:
            return {}
        raw = match.group(1).encode('utf-8').decode('unicode_escape')
        data = json.loads(raw)
        players = {}
        for p in data:
            name = p.get("player_name", "")
            pid = p.get("id", "")
            shots = float(p.get("shots", 0) or 0)
            sot = float(p.get("shots_on_target", shots * 0.4) or 0)
            games = int(p.get("games", 1) or 1)
            xg = float(p.get("xG", 0) or 0)
            time_played = float(p.get("time", 0) or 0)
            players[name.lower()] = {
                "id": pid, "name": name,
                "avg_sot": round(sot / games, 3),
                "avg_shots": round(shots / games, 3),
                "avg_xg": round(xg / games, 3),
                "games": games,
                "avg_mins": round(time_played / games) if games > 0 else 80,
            }
        print(f"  Understat: {len(players)} players for {league_name}")
        return players
    except Exception as e:
        print(f"  Understat error {league_name}: {e}")
        return {}


def get_player_match_history(player_id, player_name):
    try:
        url = f"https://understat.com/player/{player_id}"
        r = requests.get(url, headers=US_HEADERS, timeout=15)
        html = r.text
        match = re.search(r"var matchesData\s*=\s*JSON\.parse\('(.+?)'\)", html)
        if not match:
            return [], []
        raw = match.group(1).encode('utf-8').decode('unicode_escape')
        matches = json.loads(raw)
        season_matches = [m for m in matches if m.get("season") == "2024"]
        recent = season_matches[-6:] if len(season_matches) >= 6 else season_matches
        form, details = [], []
        for m in reversed(recent):
            sot = int(m.get("shots_on_target", 0) or 0)
            shots = int(m.get("shots", 0) or 0)
            xg = float(m.get("xG", 0) or 0)
            form.append(1 if sot > 0 else 0)
            details.append({
                "date": m.get("date", "")[:10],
                "opponent": m.get("h_team", "") if m.get("side") == "a" else m.get("a_team", ""),
                "sot": sot, "shots": shots, "xg": round(xg, 2),
                "result": m.get("result", "")
            })
        return form, details
    except:
        return [], []


def get_yesterday_player_sot(player_id, player_name, match_date):
    try:
        url = f"https://understat.com/player/{player_id}"
        r = requests.get(url, headers=US_HEADERS, timeout=15)
        html = r.text
        match = re.search(r"var matchesData\s*=\s*JSON\.parse\('(.+?)'\)", html)
        if not match:
            return None
        raw = match.group(1).encode('utf-8').decode('unicode_escape')
        matches = json.loads(raw)
        for m in matches:
            if m.get("date", "")[:10] == match_date:
                sot = int(m.get("shots_on_target", 0) or 0)
                return {
                    "actualSOT": sot,
                    "actualShots": int(m.get("shots", 0) or 0),
                    "actualXG": round(float(m.get("xG", 0) or 0), 2),
                    "hit": sot > 0
                }
        return None
    except:
        return None


# ══════════════════════════════════════════
# MATH ENGINE
# ══════════════════════════════════════════

def calc_xsot(avg_sot, avg_shots, avg_xg, role, team_att, opp_def, minutes):
    role_mult = ROLE_FACTORS.get(role, 1.0)
    team_mult = 0.6 + team_att * 0.8
    opp_mult = 1.0 - opp_def * 0.5
    min_mult = 1.0 if minutes >= 80 else 0.85 if minutes >= 60 else 0.6
    xg_est = avg_xg * 2.5
    base = (avg_sot * 0.6 + xg_est * 0.4) if avg_sot > 0 else avg_shots * 0.45
    return round(base * role_mult * team_mult * opp_mult * min_mult, 2)

def calc_prob(xsot, line=0.5):
    lam = max(0.01, xsot)
    p0 = math.exp(-lam)
    if line == 0.5:
        return min(0.97, max(0.03, 1 - p0))
    p1 = lam * math.exp(-lam)
    return min(0.95, max(0.02, 1 - p0 - p1))

def calc_ev(prob, odds):
    return round((prob * odds) - 1, 3)

def calc_conf(prob, avg_sot, minutes, games):
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

def get_verdict(prob, conf, ev, high_var, status):
    if high_var: return "AVOID"
    if status == "injured": return "AVOID"
    if prob >= 0.75 and conf >= 8 and ev > 0: return "SAFE"
    if prob >= 0.65 or conf >= 6: return "RISKY"
    return "AVOID"


# ══════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════

def get_fixtures(date_str):
    try:
        r = requests.get(f"{BASE}/fixtures", headers=F_HEADERS,
                        params={"date": date_str, "season": 2024}, timeout=15)
        return r.json().get("response", [])
    except Exception as e:
        print(f"  Fixtures error: {e}")
        return []


def get_team_players(team_id, league_id, us_players):
    try:
        r = requests.get(f"{BASE}/players", headers=F_HEADERS,
                        params={"team": team_id, "season": 2024, "league": league_id}, timeout=15)
        data = r.json().get("response", [])
        result = []
        for p in data:
            stats = p.get("statistics", [{}])[0]
            pos = stats.get("games", {}).get("position", "") or ""
            apps = stats.get("games", {}).get("appearences", 0) or 0
            minutes = stats.get("games", {}).get("minutes", 0) or 0
            goals = stats.get("goals", {}).get("total", 0) or 0
            if pos in ["Attacker", "Midfielder"] and apps >= 3:
                name = p["player"]["name"]
                avg_mins = round(minutes / max(apps, 1))
                us = None
                nl = name.lower()
                for key, val in us_players.items():
                    if any(part in key for part in nl.split() if len(part) > 3):
                        us = val
                        break
                result.append({
                    "name": name, "position": pos,
                    "avg_sot":   us["avg_sot"]   if us else max(0.3, goals/max(apps,1)*0.8),
                    "avg_shots": us["avg_shots"]  if us else max(0.5, goals/max(apps,1)*2),
                    "avg_xg":    us["avg_xg"]     if us else goals/max(apps,1)*0.4,
                    "avg_mins":  avg_mins,
                    "games":     us["games"]      if us else apps,
                    "has_understat": us is not None,
                    "understat_id":  us["id"]     if us else None
                })
        result.sort(key=lambda x: x["avg_sot"], reverse=True)
        return result[:4]
    except Exception as e:
        print(f"  Team players error: {e}")
        return []


def process_fixtures(fixtures, day_label, us_cache, match_date_str):
    matches = []
    count = 0
    for fix in fixtures:
        lid = fix["league"]["id"]
        if lid not in LEAGUE_MAP or count >= 5:
            continue
        count += 1
        info = LEAGUE_MAP[lid]
        us_key = info["understat"]

        if us_key not in us_cache:
            print(f"  Fetching Understat {us_key}...")
            us_cache[us_key] = get_understat_league_players(us_key)
        us_players = us_cache[us_key]

        home_id    = fix["teams"]["home"]["id"]
        away_id    = fix["teams"]["away"]["id"]
        home_name  = fix["teams"]["home"]["name"]
        away_name  = fix["teams"]["away"]["name"]
        kickoff    = fix["fixture"]["date"][11:16]
        fixture_id = fix["fixture"]["id"]
        league_name = fix["league"]["name"].lower()
        is_hv = any(w in league_name for w in ["cup","trophy","shield","supercopa"])

        # Real odds
        match_odds = get_real_odds(info["odds_key"], home_name, away_name)

        # Lineups + injuries (today/tomorrow only)
        lineups, lineups_confirmed = {}, False
        injuries = {}
        if day_label in ["today", "tomorrow"]:
            lineups, lineups_confirmed = get_lineups(fixture_id)
            injuries = get_injuries(fixture_id)

        home_pl = get_team_players(home_id, lid, us_players)
        away_pl = get_team_players(away_id, lid, us_players)

        all_players = []
        for p, t_att, o_def in [
            *[(p, 0.82, 0.68) for p in home_pl],
            *[(p, 0.76, 0.70) for p in away_pl]
        ]:
            role  = p["position"]
            xsot  = calc_xsot(p["avg_sot"], p["avg_shots"], p["avg_xg"],
                              role, t_att, o_def, p["avg_mins"])
            line  = 1.5 if p["avg_sot"] >= 1.5 else 0.5
            prob  = calc_prob(xsot, line)
            conf  = calc_conf(prob, p["avg_sot"], p["avg_mins"], p["games"])

            # Real odds movement
            p_odds = generate_player_odds(prob, line, match_odds)
            current_odds = p_odds["current"]
            ev = calc_ev(prob, current_odds)
            implied = round(1 / current_odds, 3)

            # Lineup/injury status
            p_status, p_reason = get_player_status(
                p["name"], lineups, injuries, home_name, away_name)

            # Adjust prob if injured/sub
            if p_status == "injured":
                prob = 0.0
                conf = 1
            elif p_status == "sub":
                prob = round(prob * 0.6, 3)
                conf = max(1, conf - 2)

            verdict = get_verdict(prob, conf, ev, is_hv, p_status)

            # Form from Understat
            form = [1 if i % 2 == 0 else 0 for i in range(6)]
            match_history = []
            if p["understat_id"]:
                form, match_history = get_player_match_history(
                    p["understat_id"], p["name"])

            # Yesterday actual results
            actual_result = None
            if day_label == "yesterday" and p["understat_id"]:
                actual_result = get_yesterday_player_sot(
                    p["understat_id"], p["name"], match_date_str)

            player_data = {
                "name": p["name"],
                "role": "ST" if role in ["Attacker","Forward"] else "CAM",
                "minutes": p["avg_mins"],
                "line": line,
                "xsot": xsot,
                "prob": round(prob, 3),
                "conf": conf,
                "oddsOpen": p_odds["open"],
                "oddsCurrent": current_odds,
                "oddsMoved": p_odds["moved"],
                "oddsPctMove": p_odds["pctMove"],
                "ev": ev,
                "isValue": (prob - implied) > 0.05,
                "verdict": verdict,
                "status": p_status,
                "statusReason": p_reason,
                "lineupConfirmed": lineups_confirmed,
                "form": form if form else [0,1,0,1,0,1],
                "avgSot": p["avg_sot"],
                "avgXg": p["avg_xg"],
                "hasRealData": p["has_understat"],
                "matchHistory": match_history[:6]
            }

            if actual_result:
                player_data["actualSOT"]   = actual_result["actualSOT"]
                player_data["actualShots"] = actual_result["actualShots"]
                player_data["actualXG"]    = actual_result["actualXG"]
                player_data["hit"]         = actual_result["hit"]

            all_players.append(player_data)

        matches.append({
            "id": fixture_id,
            "league": info["key"],
            "day": day_label,
            "homeTeam": home_name,
            "awayTeam": away_name,
            "kickoff": kickoff,
            "homeAttack": 0.82, "awayAttack": 0.76,
            "homeDefense": 0.70, "awayDefense": 0.68,
            "isLive": False,
            "isHighVariance": is_hv,
            "lineupConfirmed": lineups_confirmed,
            "matchOdds": match_odds,
            "players": all_players[:8]
        })
    return matches


# ══════════════════════════════════════════
# GEMINI AI
# ══════════════════════════════════════════

def get_ai_insight(matches, history_summary):
    if not matches:
        return "No matches today for analysis."
    safe, risky = [], []
    for m in matches:
        for p in m.get("players", []):
            if p["status"] == "injured": continue
            form_rate = sum(p.get("form",[])) / max(len(p.get("form",[1])),1)
            entry = (f"{p['name']} ({p['prob']*100:.0f}% prob, "
                    f"{p['avgSot']:.2f} avg SOT, "
                    f"{form_rate*100:.0f}% recent form, "
                    f"odds {p['oddsCurrent']}, EV {p['ev']*100:.1f}%)")
            if p["verdict"] == "SAFE": safe.append(entry)
            elif p["verdict"] == "RISKY": risky.append(entry)

    hist_context = ""
    if history_summary:
        s = history_summary.get("safe", {})
        hist_context = f"Model track record: Safe picks hit rate {s.get('rate',0)}% over {history_summary.get('daysTracked',0)} days, ROI {history_summary.get('roi',0)}%."

    prompt = f"""You are a sharp football shots-on-target betting analyst.
{hist_context}
Safe picks today: {', '.join(safe[:5]) if safe else 'None'}
Risky picks: {', '.join(risky[:3]) if risky else 'None'}

Write exactly 3 sentences:
1. Best value safe pick today and why (cite the stats)
2. Biggest risk to watch
3. Overall confidence based on track record

Be specific. Data-driven. Max 80 words."""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"


def get_results_summary(yesterday_matches):
    if not yesterday_matches: return ""
    hits, misses = [], []
    for m in yesterday_matches:
        for p in m.get("players", []):
            if "actualSOT" not in p: continue
            entry = f"{p['name']}: {p['actualSOT']} SOT"
            (hits if p.get("hit") else misses).append(entry)
    if not hits and not misses: return ""
    prompt = f"""Yesterday's SOT results — Hits: {', '.join(hits[:5]) if hits else 'None'} | Misses: {', '.join(misses[:5]) if misses else 'None'}
2 sentences: what worked, what to learn. Max 40 words."""
    try:
        return model.generate_content(prompt).text.strip()
    except:
        return ""


# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════

print("=" * 55)
print("SOTIQ Bot —", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
print("=" * 55)

# Load history
print("\nLoading history...")
history = load_history()
print(f"  {len(history.get('predictions',[]))} past predictions tracked")

us_cache = {}
all_matches = []

print("\n[1/3] Yesterday + real results...")
y_fixes = get_fixtures(fmt(yesterday))
y_matches = process_fixtures(y_fixes, "yesterday", us_cache, fmt(yesterday))
all_matches += y_matches
print(f"  {len(y_matches)} matches")

print("\n[2/3] Today + predictions + lineups + odds...")
t_fixes = get_fixtures(fmt(today))
t_matches = process_fixtures(t_fixes, "today", us_cache, fmt(today))
all_matches += t_matches
print(f"  {len(t_matches)} matches")

print("\n[3/3] Tomorrow...")
tm_fixes = get_fixtures(fmt(tomorrow))
tm_matches = process_fixtures(tm_fixes, "tomorrow", us_cache, fmt(tomorrow))
all_matches += tm_matches
print(f"  {len(tm_matches)} matches")

# Update history with yesterday's results
yesterday_matches = [m for m in all_matches if m["day"] == "yesterday"]
history = update_history(yesterday_matches, history)
save_history(history)

# AI insights
print("\nGenerating Gemini AI insights...")
today_matches = [m for m in all_matches if m["day"] == "today"]
ai_insight = get_ai_insight(today_matches, history.get("summary", {}))
results_summary = get_results_summary(yesterday_matches)
print("  Done!")

# Stats
all_p   = [p for m in today_matches for p in m.get("players", [])]
safe_c  = len([p for p in all_p if p["verdict"] == "SAFE"])
ev_c    = len([p for p in all_p if p["ev"] > 0])
value_c = len([p for p in all_p if p["isValue"]])
avoid_c = len([p for p in all_p if p["verdict"] == "AVOID"])

y_all_p = [p for m in yesterday_matches for p in m.get("players",[]) if "actualSOT" in p]
y_hits  = len([p for p in y_all_p if p.get("hit")])
y_total = len(y_all_p)
hit_rate = round(y_hits / y_total * 100) if y_total > 0 else None

print(f"\nToday     — Safe:{safe_c} EV+:{ev_c} Value:{value_c} Avoid:{avoid_c}")
if hit_rate is not None:
    print(f"Yesterday — {y_hits}/{y_total} hit ({hit_rate}%)")

output = {
    "updated": datetime.utcnow().isoformat() + "Z",
    "aiInsight": ai_insight,
    "resultsSummary": results_summary,
    "hitRate": hit_rate,
    "historySummary": history.get("summary", {}),
    "matches": all_matches
}

with open("data.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✅ data.json saved!")
print("✅ history.json saved!")
print("=" * 55)
