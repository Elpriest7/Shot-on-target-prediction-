import os, json, requests, math, re, time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import google.generativeai as genai

# ── KEYS ──
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY")
FOOTBALL_KEY  = os.environ.get("FOOTBALL_API_KEY")
ODDS_KEY      = os.environ.get("ODDS_API_KEY")
THEODDS_KEY   = os.environ.get("THEODDS_API_KEY")  # Real player SOT lines

genai.configure(api_key=GEMINI_KEY)
# gemini-2.5-flash = current free tier model (250 req/day)
model = genai.GenerativeModel("gemini-2.5-flash")

today     = datetime.utcnow()
yesterday = today - timedelta(days=1)
tomorrow  = today + timedelta(days=1)
def fmt(d): return d.strftime("%Y-%m-%d")

# ══════════════════════════════════════════
# LEAGUE REGISTRY
# Data quality tiers:
#   A = Understat (best - real SOT per game)
#   B = FBref     (great - real SOT per game)
#   C = API-Football season averages (good)
# ══════════════════════════════════════════

# ── FOCUSED ON 7 BEST LEAGUES WITH REAL PER-GAME SOT DATA ──
# Tier A = Understat (real SOT per game — top 5 leagues)
# Tier B = FBref (real SOT per game — European competitions)
LEAGUES = {
    39:  {"key":"pl",         "name":"Premier League",   "flag":"🏴󠁧󠁢󠁥󠁮󠁧󠁿", "country":"England", "tier":"A",
          "understat":"EPL",       "odds":"soccer_epl"},
    140: {"key":"laliga",     "name":"La Liga",          "flag":"🇪🇸", "country":"Spain",   "tier":"A",
          "understat":"La_liga",   "odds":"soccer_spain_la_liga"},
    78:  {"key":"bundesliga", "name":"Bundesliga",       "flag":"🇩🇪", "country":"Germany", "tier":"A",
          "understat":"Bundesliga","odds":"soccer_germany_bundesliga"},
    135: {"key":"seriea",     "name":"Serie A",          "flag":"🇮🇹", "country":"Italy",   "tier":"A",
          "understat":"Serie_A",   "odds":"soccer_italy_serie_a"},
    61:  {"key":"ligue1",     "name":"Ligue 1",          "flag":"🇫🇷", "country":"France",  "tier":"A",
          "understat":"Ligue_1",   "odds":"soccer_france_ligue_one"},
    2:   {"key":"ucl",        "name":"Champions League", "flag":"⭐", "country":"Europe",  "tier":"B",
          "fbref":"8",             "odds":"soccer_uefa_champs_league"},
    3:   {"key":"uel",        "name":"Europa League",    "flag":"🟠", "country":"Europe",  "tier":"B",
          "fbref":"19",            "odds":"soccer_uefa_europa_league"},
}

# ── THE ODDS API — PLAYER SOT MARKET KEYS ──
THEODDS_LEAGUES = {
    39:  "soccer_epl",
    140: "soccer_spain_la_liga",
    78:  "soccer_germany_bundesliga",
    135: "soccer_italy_serie_a",
    61:  "soccer_france_ligue_one",
    2:   "soccer_uefa_champs_league",
    3:   "soccer_uefa_europa_league",
}

ROLE_FACTORS = {
    "Attacker":1.25,"Forward":1.25,
    "Midfielder":0.85,"Defender":0.25,"Goalkeeper":0.05
}

F_HEADERS = {
    "x-rapidapi-host":"v3.football.api-sports.io",
    "x-rapidapi-key": FOOTBALL_KEY
}
BASE = "https://v3.football.api-sports.io"
US_HEADERS = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ══════════════════════════════════════════
# UNDERSTAT (Tier A)
# ══════════════════════════════════════════

us_cache = {}

def get_understat_players(league_name):
    if league_name in us_cache:
        return us_cache[league_name]
    try:
        url = f"https://understat.com/league/{league_name}/2025"
        r = requests.get(url, headers=US_HEADERS, timeout=20)
        match = re.search(r"var playersData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
        if not match: return {}
        raw  = match.group(1)
        # Safe unicode decode
        try:
            raw = raw.encode('utf-8').decode('unicode_escape')
        except Exception:
            raw = raw.encode('raw_unicode_escape').decode('unicode_escape')
        data = json.loads(raw)
        out  = {}
        for p in data:
            shots= float(p.get("shots",0) or 0)
            sot  = float(p.get("shots_on_target", shots*0.4) or 0)
            g    = int(p.get("games",1) or 1)
            xg   = float(p.get("xG",0) or 0)
            t    = float(p.get("time",0) or 0)
            out[p.get("player_name","").lower()] = {
                "id": p.get("id",""), "name": p.get("player_name",""),
                "avg_sot":   round(sot/g,3),
                "avg_shots": round(shots/g,3),
                "avg_xg":    round(xg/g,3),
                "games": g,
                "avg_mins":  round(t/g) if g>0 else 80,
            }
        us_cache[league_name] = out
        print(f"  Understat {league_name}: {len(out)} players")
        return out
    except Exception as e:
        print(f"  Understat err {league_name}: {e}")
        return {}

def us_player_history(pid, pname):
    try:
        r = requests.get(f"https://understat.com/player/{pid}", headers=US_HEADERS, timeout=15)
        m = re.search(r"var matchesData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
        if not m: return [],[]
        raw = m.group(1)
        try:
            raw = raw.encode('utf-8').decode('unicode_escape')
        except Exception:
            raw = raw.encode('raw_unicode_escape').decode('unicode_escape')
        matches = [x for x in json.loads(raw) if x.get("season") in ["2025","2024"]]
        recent = matches[-6:] if len(matches)>=6 else matches
        form, details = [],[]
        for x in reversed(recent):
            sot   = int(x.get("shots_on_target",0) or 0)
            shots = int(x.get("shots",0) or 0)
            xg    = float(x.get("xG",0) or 0)
            form.append(1 if sot>0 else 0)
            details.append({"date":x.get("date","")[:10],
                "opponent": x.get("h_team","") if x.get("side")=="a" else x.get("a_team",""),
                "sot":sot,"shots":shots,"xg":round(xg,2)})
        return form, details
    except: return [],[]

def us_yesterday_sot(pid, match_date):
    try:
        r = requests.get(f"https://understat.com/player/{pid}", headers=US_HEADERS, timeout=15)
        m = re.search(r"var matchesData\s*=\s*JSON\.parse\('(.+?)'\)", r.text)
        if not m: return None
        raw = m.group(1)
        try:
            raw = raw.encode('utf-8').decode('unicode_escape')
        except Exception:
            raw = raw.encode('raw_unicode_escape').decode('unicode_escape')
        for x in json.loads(raw):
            if x.get("date","")[:10] == match_date:
                sot = int(x.get("shots_on_target",0) or 0)
                return {"actualSOT":sot,"actualShots":int(x.get("shots",0) or 0),
                        "actualXG":round(float(x.get("xG",0) or 0),2),"hit":sot>0}
        return None
    except: return None

# ══════════════════════════════════════════
# FBREF (Tier B)
# ══════════════════════════════════════════

fbref_cache = {}

FBREF_LEAGUE_URLS = {
    "8":  "https://fbref.com/en/comps/8/2025-2026/shooting/2025-2026-Champions-League-Stats",
    "19": "https://fbref.com/en/comps/19/2025-2026/shooting/2025-2026-Europa-League-Stats",
    "882":"https://fbref.com/en/comps/882/shooting/2025-2026/UEFA-Europa-Conference-League-Stats",
}

def get_fbref_players(fbref_id):
    if fbref_id in fbref_cache:
        return fbref_cache[fbref_id]
    url = FBREF_LEAGUE_URLS.get(fbref_id)
    if not url: return {}
    try:
        time.sleep(4)  # FBref rate limit
        r = requests.get(url, headers={
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept":"text/html,application/xhtml+xml"
        }, timeout=25)
        # FBref hides table in HTML comments — strip them
        html = r.text.replace("<!--","").replace("-->","")
        soup = BeautifulSoup(html, "html.parser")
        table = (soup.find("table", {"id": re.compile("stats_shooting")}) or
                 soup.find("table", id=lambda x: x and "shooting" in x))
        if not table:
            print(f"  FBref: no shooting table for {fbref_id}")
            return {}
        rows = table.find("tbody").find_all("tr")
        out = {}
        for row in rows:
            if "thead" in row.get("class",[]): continue
            cells = row.find_all(["td","th"])
            if len(cells) < 10: continue
            try:
                name  = row.find("td", {"data-stat":"player"})
                if not name: continue
                name  = name.get_text(strip=True)
                pos   = row.find("td", {"data-stat":"position"})
                pos   = pos.get_text(strip=True) if pos else "FW"
                shots = row.find("td", {"data-stat":"shots"})
                shots = float(shots.get_text(strip=True) or 0) if shots else 0
                sot   = row.find("td", {"data-stat":"shots_on_target"})
                sot   = float(sot.get_text(strip=True) or 0) if sot else 0
                mins  = row.find("td", {"data-stat":"minutes"})
                mins  = float(mins.get_text(strip=True).replace(",","") or 0) if mins else 0
                xg    = row.find("td", {"data-stat":"xg"})
                xg    = float(xg.get_text(strip=True) or 0) if xg else 0
                games = row.find("td", {"data-stat":"games"})
                games = int(games.get_text(strip=True) or 1) if games else 1
                games = max(games, 1)
                out[name.lower()] = {
                    "name": name, "pos": pos,
                    "avg_sot":   round(sot/games,3),
                    "avg_shots": round(shots/games,3),
                    "avg_xg":    round(xg/games,3),
                    "games": games,
                    "avg_mins":  round(mins/games) if games>0 else 80,
                }
            except: continue
        fbref_cache[fbref_id] = out
        print(f"  FBref {fbref_id}: {len(out)} players")
        return out
    except Exception as e:
        print(f"  FBref err {fbref_id}: {e}")
        return {}

# ══════════════════════════════════════════
# MATH ENGINE — ELITE UPGRADES
# ══════════════════════════════════════════

# ── PUBLIC BIAS PLAYERS & MATCHES (Upgrade 4) ──
# These attract public money → lines shorter than fair value
# We discount model probability to avoid overconfidence
PUBLIC_BIAS_PLAYERS = {
    # name fragment : discount factor
    "haaland":  0.94,
    "mbappe":   0.93,
    "salah":    0.95,
    "vinicius": 0.94,
    "bellingham":0.95,
    "ronaldo":  0.93,
    "neymar":   0.93,
    "lewandowski":0.95,
    "kane":     0.95,
    "yamal":    0.94,
    "saka":     0.96,
    "osimhen":  0.95,
}

PUBLIC_BIAS_LEAGUES = {
    "ucl": 0.96,   # Champions League — biggest public bias
    "uel": 0.97,   # Europa League
}

def apply_public_bias(prob, player_name, league_key):
    """Discount probability for public bias players/leagues (Upgrade 4)"""
    nl = player_name.lower()
    factor = 1.0
    # Check player bias
    for fragment, disc in PUBLIC_BIAS_PLAYERS.items():
        if fragment in nl:
            factor = min(factor, disc)
            break
    # Check league bias
    league_disc = PUBLIC_BIAS_LEAGUES.get(league_key, 1.0)
    factor = min(factor, league_disc)
    return round(prob * factor, 3)

# ── CALIBRATION LAYER (Upgrade 2) ──
# Corrects model overconfidence based on historical results
# calibration_factor = actual_hit_rate / predicted_probability_avg
# Starts neutral (1.0) and adjusts as history builds

def get_calibration_factor(history_summary, verdict_type="SAFE"):
    """
    Calculate calibration factor from historical results.
    If model predicts 75% avg but hits 68% → factor = 0.68/0.75 = 0.907
    Requires at least 30 predictions to be meaningful.
    """
    if not history_summary:
        return 1.0
    s = history_summary.get(verdict_type.lower(), {})
    total = s.get("total", 0)
    if total < 30:
        # Not enough data yet — return slight conservative adjustment
        return 0.97
    hit_rate  = s.get("rate", 75) / 100
    # Estimate avg predicted probability for this verdict type
    avg_pred = 0.75 if verdict_type == "SAFE" else 0.65
    if avg_pred <= 0:
        return 1.0
    factor = hit_rate / avg_pred
    # Clamp between 0.80 and 1.10 to prevent wild swings
    return round(max(0.80, min(1.10, factor)), 3)

def apply_calibration(prob, calibration_factor):
    """Apply calibration correction to model probability (Upgrade 2)"""
    calibrated = prob * calibration_factor
    return round(max(0.01, min(0.97, calibrated)), 3)

# ── PLAYER SPECIFIC MODIFIERS (Upgrade 3) ──

def calc_finishing_efficiency(avg_sot, avg_shots):
    """
    SOT/shots ratio — finishing efficiency modifier.
    Elite: >0.55 (55%+ of shots on target)
    Average: 0.35-0.55
    Poor: <0.35
    """
    if avg_shots <= 0:
        return 1.0
    ratio = avg_sot / avg_shots
    if ratio >= 0.60: return 1.12   # Elite converter
    if ratio >= 0.50: return 1.06   # Above average
    if ratio >= 0.40: return 1.0    # Average
    if ratio >= 0.30: return 0.94   # Below average
    return 0.88                      # Poor converter

def calc_overperformance_trend(avg_sot, avg_xg):
    """
    Check if player consistently overperforms xG → SOT gap.
    avg_sot >> xG*2 means player shoots more than xG suggests.
    """
    if avg_xg <= 0:
        return 1.0
    xg_implied_sot = avg_xg * 2.5  # Expected SOT from xG
    if avg_sot <= 0:
        return 1.0
    ratio = avg_sot / xg_implied_sot
    if ratio >= 1.3: return 1.08    # Consistently overperforms
    if ratio >= 1.1: return 1.03
    if ratio >= 0.9: return 1.0     # In line with xG
    if ratio >= 0.7: return 0.96    # Underperforms
    return 0.92                      # Consistent underperformer

def apply_player_modifiers(xsot, avg_sot, avg_shots, avg_xg):
    """Apply finishing efficiency and overperformance trend (Upgrade 3)"""
    fin_eff  = calc_finishing_efficiency(avg_sot, avg_shots)
    overperf = calc_overperformance_trend(avg_sot, avg_xg)
    modified = xsot * fin_eff * overperf
    return round(max(0.01, modified), 2)

# ── CONFIDENCE TIGHTENING (Upgrade 5) ──
# Stricter criteria — fewer but better picks

TIGHT_THRESHOLDS = {
    "SAFE_PROB":     0.75,   # Minimum probability for SAFE
    "SAFE_CONF":     8,      # Minimum confidence for SAFE
    "SAFE_EV":       0.03,   # Minimum EV for SAFE (3%)
    "BANKER_PROB":   0.82,   # Minimum probability for BANKER
    "BANKER_CONF":   9,      # Minimum confidence for BANKER
    "BANKER_EV":     0.06,   # Minimum EV for BANKER (6%)
    "BANKER_SOT":    1.2,    # Minimum avg SOT for BANKER
    "BANKER_FORM":   0.55,   # Minimum form rate for BANKER
    "VALUE_EDGE":    0.05,   # Minimum edge over implied prob
    "REAL_ODDS_BONUS": 0.02, # Extra EV buffer required without real odds
}

def get_tight_verdict(prob, conf, ev, high_var, status, real_odds, calibration_factor, league_key):
    """
    Tightened verdict system (Upgrade 5).
    Fewer picks but much higher quality.
    """
    if high_var or status == "injured":
        return "AVOID"

    # Extra penalty if no real odds available
    ev_threshold = TIGHT_THRESHOLDS["SAFE_EV"]
    if not real_odds:
        ev_threshold += TIGHT_THRESHOLDS["REAL_ODDS_BONUS"]

    # UCL/derby extra caution
    if league_key in ["ucl","uel"]:
        ev_threshold += 0.01

    if (prob >= TIGHT_THRESHOLDS["SAFE_PROB"] and
        conf >= TIGHT_THRESHOLDS["SAFE_CONF"] and
        ev  >= ev_threshold):
        return "SAFE"

    if prob >= 0.65 or conf >= 6:
        return "RISKY"

    return "AVOID"

def is_tight_banker(prob, conf, ev, avg_sot, form, real_odds):
    """Stricter banker criteria (Upgrade 5)"""
    form_rate = sum(form) / max(len(form), 1)
    ev_min = TIGHT_THRESHOLDS["BANKER_EV"]
    if not real_odds:
        ev_min += 0.02  # Require more edge without real odds
    return (
        prob      >= TIGHT_THRESHOLDS["BANKER_PROB"] and
        conf      >= TIGHT_THRESHOLDS["BANKER_CONF"] and
        ev        >= ev_min and
        avg_sot   >= TIGHT_THRESHOLDS["BANKER_SOT"] and
        form_rate >= TIGHT_THRESHOLDS["BANKER_FORM"]
    )

# ── CLV TRACKING (Upgrade 1) ──
# Closing Line Value: compare prediction odds vs closing odds
# If you consistently beat closing line → long term profitable

def calc_clv(open_odds, closing_odds):
    """
    CLV = (closing_odds - open_odds) / open_odds × 100
    Positive CLV means you got better odds than market settled at.
    This is the strongest predictor of long-term profitability.
    """
    if not closing_odds or not open_odds or open_odds <= 0:
        return None
    clv = (closing_odds - open_odds) / open_odds * 100
    return round(clv, 2)

def update_clv_in_history(history, closing_odds_data):
    """
    Update yesterday's predictions with closing odds for CLV calculation.
    closing_odds_data: {player_name: closing_odds}
    """
    updated = 0
    for pred in history.get("predictions", []):
        pname = pred.get("player","").lower()
        if pname in closing_odds_data and "clv" not in pred:
            closing = closing_odds_data.get(pname)
            open_o  = pred.get("oddsCurrent", 0)
            clv     = calc_clv(open_o, closing)
            if clv is not None:
                pred["clv"] = clv
                updated += 1
    if updated:
        print(f"  CLV updated for {updated} predictions")
    return history

def calc_clv_summary(predictions):
    """
    Calculate average CLV from predictions that have closing odds.
    Positive avg CLV = profitable model regardless of short-term results.
    """
    clv_preds = [p for p in predictions if "clv" in p]
    if not clv_preds:
        return {"avg_clv": None, "positive_clv_rate": None, "total": 0}
    avg_clv = round(sum(p["clv"] for p in clv_preds) / len(clv_preds), 2)
    pos_rate = round(len([p for p in clv_preds if p["clv"] > 0]) / len(clv_preds) * 100)
    return {
        "avg_clv":         avg_clv,
        "positive_clv_rate": pos_rate,
        "total":           len(clv_preds),
        "beating_market":  avg_clv > 0
    }

def fetch_closing_odds(sport_key, event_id, player_names):
    """
    Fetch closing odds (post-kickoff) for CLV calculation.
    Uses The Odds API — same endpoint but called after match starts.
    """
    if not THEODDS_KEY or not event_id or not sport_key:
        return {}
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds",
            params={
                "apiKey":    THEODDS_KEY,
                "regions":   "uk,eu",
                "markets":   "player_shots_on_target",
                "oddsFormat":"decimal"
            },
            timeout=15)
        if r.status_code != 200:
            return {}
        data    = r.json()
        closing = {}
        for bk in data.get("bookmakers", []):
            for market in bk.get("markets", []):
                if market.get("key") != "player_shots_on_target":
                    continue
                for outcome in market.get("outcomes", []):
                    pname = outcome.get("description","").lower()
                    side  = outcome.get("name","").lower()
                    price = outcome.get("price", 0)
                    if "over" in side and pname:
                        # Keep highest closing odds found
                        if pname not in closing or price > closing[pname]:
                            closing[pname] = price
        return closing
    except Exception as e:
        print(f"  CLV fetch err: {e}")
        return {}

def calc_xsot(avg_sot, avg_shots, avg_xg, role, t_att, o_def, mins):
    rm = ROLE_FACTORS.get(role, 1.0)
    tm = 0.6 + t_att * 0.8
    om = 1.0 - o_def * 0.5
    mm = 1.0 if mins>=80 else 0.85 if mins>=60 else 0.6
    xg_est = avg_xg * 2.5
    base = (avg_sot*0.6 + xg_est*0.4) if avg_sot>0 else avg_shots*0.45
    return round(base * rm * tm * om * mm, 2)

def calc_prob(xsot, line=0.5):
    lam = max(0.01, xsot)
    p0  = math.exp(-lam)
    if line==0.5: return min(0.97, max(0.03, 1-p0))
    p1  = lam * math.exp(-lam)
    return min(0.95, max(0.02, 1-p0-p1))

def calc_ev(prob, odds): return round((prob*odds)-1, 3)

def calc_conf(prob, avg_sot, mins, games, data_tier):
    c = 1
    if prob>=0.85: c+=3
    elif prob>=0.75: c+=2
    elif prob>=0.65: c+=1
    if avg_sot>=1.5: c+=2
    elif avg_sot>=1.0: c+=1
    if mins>=80: c+=1.5
    elif mins<60: c-=1
    if games>=10: c+=1
    # Bonus for higher quality data
    if data_tier=="A": c+=1
    elif data_tier=="B": c+=0.5
    return min(10, max(1, round(c)))

def get_verdict(prob, conf, ev, high_var, status):
    if high_var or status=="injured": return "AVOID"
    if prob>=0.75 and conf>=8 and ev>0: return "SAFE"
    if prob>=0.65 or conf>=6: return "RISKY"
    return "AVOID"

def is_banker(prob, conf, ev, avg_sot, form):
    """Banker = highest confidence picks of the day"""
    form_rate = sum(form)/max(len(form),1)
    return (prob>=0.82 and conf>=9 and ev>0.05
            and avg_sot>=1.2 and form_rate>=0.6)

# ══════════════════════════════════════════
# ODDS — THE ODDS API (Real Player SOT Lines)
# ══════════════════════════════════════════

theodds_events_cache = {}   # sport_key -> list of events
theodds_props_cache  = {}   # event_id  -> player SOT props

def get_theodds_events(sport_key):
    """Get all upcoming events for a sport from The Odds API"""
    if sport_key in theodds_events_cache:
        return theodds_events_cache[sport_key]
    if not THEODDS_KEY:
        return []
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/events",
            params={"apiKey": THEODDS_KEY},
            timeout=15)
        if r.status_code == 200:
            data = r.json()
            theodds_events_cache[sport_key] = data
            remaining = r.headers.get("x-requests-remaining","?")
            print(f"  TheOddsAPI events {sport_key}: {len(data)} events | Credits left: {remaining}")
            return data
        else:
            print(f"  TheOddsAPI events error: {r.status_code}")
            return []
    except Exception as e:
        print(f"  TheOddsAPI events err: {e}")
        return []

# ── Nigerian bookmaker keys on The Odds API ──
NIGERIAN_BOOKS = {
    "onexbet":   "1xBet",
    "parimatch": "Parimatch",
    "betano":    "Betano",
}

def get_player_sot_lines(sport_key, event_id, home_team, away_team):
    """
    Fetch real player_shots_on_target lines from The Odds API.
    Collects odds from 1xBet, Parimatch and Betano specifically.
    Returns dict: {
        player_name_lower: {
            line, best_odds, best_book,
            books: {1xBet: odds, Parimatch: odds, Betano: odds},
            found: True
        }
    }
    """
    cache_key = f"{sport_key}:{event_id}"
    if cache_key in theodds_props_cache:
        return theodds_props_cache[cache_key]

    if not THEODDS_KEY or not event_id:
        return {}

    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds",
            params={
                "apiKey":    THEODDS_KEY,
                "regions":   "uk,eu",
                "markets":   "player_shots_on_target",
                "oddsFormat":"decimal",
                "bookmakers":"onexbet,parimatch,betano"
            },
            timeout=15)

        remaining = r.headers.get("x-requests-remaining","?")
        print(f"  SOT lines {home_team} vs {away_team} | Credits left: {remaining}")

        if r.status_code != 200:
            print(f"  TheOddsAPI error: {r.status_code} — {r.text[:120]}")
            theodds_props_cache[cache_key] = {}
            return {}

        data = r.json()
        props = {}

        for bookmaker in data.get("bookmakers", []):
            bk_key  = bookmaker.get("key","")
            bk_name = NIGERIAN_BOOKS.get(bk_key, bookmaker.get("title",""))
            if bk_key not in NIGERIAN_BOOKS:
                continue  # skip non-Nigerian books

            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_shots_on_target":
                    continue
                for outcome in market.get("outcomes", []):
                    pname = outcome.get("description","").lower()
                    side  = outcome.get("name","").lower()
                    price = outcome.get("price", 0)
                    point = float(outcome.get("point", 0.5))
                    if not pname or "over" not in side:
                        continue

                    if pname not in props:
                        props[pname] = {
                            "lines":     {},   # {0.5: best_odds, 1.5: best_odds}
                            "best_odds": 0,
                            "best_book": "",
                            "best_line": 0.5,
                            # per bookmaker: {bk_name: {0.5: odds, 1.5: odds}}
                            "books":     {},
                            "found":     True
                        }

                    # Store per-bookmaker per-line odds
                    if bk_name not in props[pname]["books"]:
                        props[pname]["books"][bk_name] = {}
                    props[pname]["books"][bk_name][point] = price

                    # Store best odds across all lines/books
                    if price > props[pname]["best_odds"]:
                        props[pname]["best_odds"] = price
                        props[pname]["best_book"] = bk_name
                        props[pname]["best_line"] = point

                    # Store best odds per line
                    if point not in props[pname]["lines"] or price > props[pname]["lines"][point]:
                        props[pname]["lines"][point] = price

        # Fallback: capture from any bookmaker if none of the 3 found
        if not props:
            for bookmaker in data.get("bookmakers", []):
                bk_name = bookmaker.get("title","")
                for market in bookmaker.get("markets", []):
                    if market.get("key") != "player_shots_on_target":
                        continue
                    for outcome in market.get("outcomes", []):
                        pname = outcome.get("description","").lower()
                        side  = outcome.get("name","").lower()
                        price = outcome.get("price", 0)
                        point = float(outcome.get("point", 0.5))
                        if not pname or "over" not in side:
                            continue
                        if pname not in props:
                            props[pname] = {"lines":{},"best_odds":0,"best_book":"","best_line":0.5,"books":{},"found":True}
                        if bk_name not in props[pname]["books"]:
                            props[pname]["books"][bk_name] = {}
                        props[pname]["books"][bk_name][point] = price
                        if price > props[pname]["best_odds"]:
                            props[pname]["best_odds"] = price
                            props[pname]["best_book"] = bk_name
                            props[pname]["best_line"] = point
                        if point not in props[pname]["lines"] or price > props[pname]["lines"][point]:
                            props[pname]["lines"][point] = price

        theodds_props_cache[cache_key] = props
        print(f"  Found SOT lines for {len(props)} players")
        return props

    except Exception as e:
        print(f"  SOT lines err: {e}")
        theodds_props_cache[cache_key] = {}
        return {}

def match_event_id(sport_key, home_team, away_team):
    """Find The Odds API event ID for a given match"""
    events = get_theodds_events(sport_key)
    hl = home_team.lower()
    al = away_team.lower()
    for ev in events:
        eh = ev.get("home_team","").lower()
        ea = ev.get("away_team","").lower()
        # Fuzzy match — check if any word from team name matches
        h_match = any(w in eh for w in hl.split() if len(w)>3)
        a_match = any(w in ea for w in al.split() if len(w)>3)
        if h_match or a_match:
            return ev.get("id","")
    return ""

def get_match_h2h_odds(sport_key, home, away):
    """Get match H2H odds from The Odds API as fallback"""
    if not THEODDS_KEY: 
        return {"home":1.9,"away":2.1,"draw":3.4,"found":False}
    try:
        events = get_theodds_events(sport_key)
        hl = home.lower(); al = away.lower()
        for ev in events:
            eh = ev.get("home_team","").lower()
            ea = ev.get("away_team","").lower()
            if any(w in eh for w in hl.split() if len(w)>3) or any(w in ea for w in al.split() if len(w)>3):
                # Fetch odds for this event
                r = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{ev['id']}/odds",
                    params={"apiKey":THEODDS_KEY,"regions":"uk","markets":"h2h","oddsFormat":"decimal"},
                    timeout=15)
                if r.status_code==200:
                    bks = r.json().get("bookmakers",[])
                    if bks:
                        outs = bks[0].get("markets",[{}])[0].get("outcomes",[])
                        om = {o["name"].lower():o["price"] for o in outs}
                        vals = list(om.values())
                        return {"home":vals[0] if vals else 1.9,
                                "away":vals[1] if len(vals)>1 else 2.1,
                                "draw":om.get("draw",3.4),"found":True}
        return {"home":1.9,"away":2.1,"draw":3.4,"found":False}
    except:
        return {"home":1.9,"away":2.1,"draw":3.4,"found":False}

def resolve_player_odds(player_name, sot_lines, prob, match_odds):
    """
    Try to get real SOT line odds for a player from Nigerian bookmakers.
    Falls back to estimated odds if not found.
    Returns odds info including per-bookmaker breakdown.
    """
    nl = player_name.lower()
    real_line = None

    # Fuzzy match player name against SOT lines
    for key, val in sot_lines.items():
        name_parts = [p for p in nl.split() if len(p)>3]
        if any(part in key for part in name_parts):
            real_line = val
            break

    if real_line and real_line.get("best_odds",0) > 0:
        # Real bookmaker SOT line found!
        best_o   = real_line["best_odds"]
        best_line= real_line.get("best_line", 0.5)
        open_o   = round(best_o * 1.03, 2)
        moved    = "up" if best_o < open_o else "flat"
        pct_move = abs(round((best_o - open_o) / open_o * 100, 1))
        return {
            "open":      open_o,
            "current":   best_o,
            "line":      best_line,
            "moved":     moved,
            "pctMove":   pct_move,
            "realOdds":  True,
            "best_book": real_line.get("best_book",""),
            "best_line": best_line,
            "books":     real_line.get("books",{}),   # {bk: {0.5: odds, 1.5: odds}}
            "lines":     real_line.get("lines",{}),   # {0.5: best_odds, 1.5: best_odds}
        }
    else:
        # Fallback: estimate from match odds + probability
        fair   = round(1/max(prob,0.01), 2)
        margin = 0.08
        avg_mo = (match_odds.get("home",1.9)+match_odds.get("away",2.1))/2
        drift  = (prob-0.5)*0.3 - (0.05/avg_mo)
        open_o = max(1.15, min(round(fair*(1-margin*0.5),2), 5.0))
        curr_o = max(1.10, min(round(open_o-drift,2), 5.0))
        moved  = "up" if curr_o<open_o else "down" if curr_o>open_o else "flat"
        return {
            "open":      open_o,
            "current":   curr_o,
            "line":      None,
            "moved":     moved,
            "pctMove":   abs(round((curr_o-open_o)/open_o*100,1)),
            "realOdds":  False,
            "best_book": "",
            "books":     {},
        }

# ══════════════════════════════════════════
# LINEUPS & INJURIES
# ══════════════════════════════════════════

def get_lineups(fixture_id):
    try:
        r = requests.get(f"{BASE}/fixtures/lineups",
                        headers=F_HEADERS, params={"fixture":fixture_id}, timeout=15)
        data = r.json().get("response",[])
        if not data: return {}, False
        out = {}
        for tl in data:
            tn = tl["team"]["name"]
            out[tn] = {
                "starters":[p["player"]["name"] for p in tl.get("startXI",[])],
                "subs":    [p["player"]["name"] for p in tl.get("substitutes",[])]
            }
        return out, True
    except: return {}, False

def get_injuries(fixture_id):
    try:
        r = requests.get(f"{BASE}/injuries",
                        headers=F_HEADERS, params={"fixture":fixture_id}, timeout=15)
        out = {}
        for p in r.json().get("response",[]):
            out[p["player"]["name"].lower()] = p["player"].get("reason","Injured")
        return out
    except: return {}

def player_status(name, lineups, injuries):
    nl = name.lower()
    for inj,reason in injuries.items():
        if any(part in inj for part in nl.split() if len(part)>3):
            return "injured", reason
    for tl in lineups.values():
        for s in tl["starters"]:
            if any(part in s.lower() for part in nl.split() if len(part)>3):
                return "confirmed","Starting XI"
        for s in tl["subs"]:
            if any(part in s.lower() for part in nl.split() if len(part)>3):
                return "sub","On bench"
    return "unknown","Lineup TBC"

# ══════════════════════════════════════════
# HIT RATE HISTORY
# ══════════════════════════════════════════

def load_history():
    try:
        if os.path.exists("history.json"):
            with open("history.json") as f: return json.load(f)
    except: pass
    return {"predictions":[],"summary":{}}

def save_history(h):
    try:
        with open("history.json","w") as f: json.dump(h,f,indent=2)
    except Exception as e: print(f"  History err: {e}")

def update_history(y_matches, history):
    already = any(p["date"]==fmt(yesterday) for p in history.get("predictions",[]))
    if already: return history
    entries = []
    for m in y_matches:
        for p in m.get("players",[]):
            if "actualSOT" not in p: continue
            entries.append({
                "date":fmt(yesterday),"player":p["name"],
                "match":f"{m['homeTeam']} vs {m['awayTeam']}",
                "league":m["league"],"verdict":p["verdict"],
                "line":p["line"],"prob":p["prob"],"ev":p["ev"],
                "xsot":p["xsot"],"actualSOT":p["actualSOT"],
                "hit":p.get("hit",False),"banker":p.get("banker",False),
                "oddsCurrent":p.get("oddsCurrent",1.8)
            })
    if entries:
        history["predictions"].extend(entries)
        cutoff = fmt(today-timedelta(days=90))
        history["predictions"] = [p for p in history["predictions"] if p["date"]>=cutoff]
        history["summary"] = calc_summary(history["predictions"])
        print(f"  History: +{len(entries)} results")
    return history

def calc_summary(preds):
    if not preds: return {}
    def stats(ps):
        if not ps: return {"hits":0,"total":0,"rate":0,"avgEV":0}
        hits = len([p for p in ps if p["hit"]])
        return {"hits":hits,"total":len(ps),
                "rate":round(hits/len(ps)*100),
                "avgEV":round(sum(p["ev"] for p in ps)/len(ps)*100,1)}
    safe_p   = [p for p in preds if p["verdict"]=="SAFE"]
    risky_p  = [p for p in preds if p["verdict"]=="RISKY"]
    banker_p = [p for p in preds if p.get("banker")]
    staked   = len([p for p in preds if p["verdict"] in ["SAFE","RISKY"]])
    ret      = sum(p.get("oddsCurrent",1.8) for p in preds if p["verdict"] in ["SAFE","RISKY"] and p["hit"])
    roi      = round((ret-staked)/max(staked,1)*100,1)
    clv_summary = calc_clv_summary(preds)
    return {
        "safe":stats(safe_p),"risky":stats(risky_p),
        "banker":stats(banker_p),"all":stats(preds),
        "roi":roi,"totalPredictions":len(preds),
        "daysTracked":len(set(p["date"] for p in preds)),
        "clv": clv_summary,
    }

# ══════════════════════════════════════════
# FIXTURES PROCESSOR
# ══════════════════════════════════════════

def get_fixtures(date_str):
    try:
        # No season filter - get all fixtures for the date across all seasons
        r = requests.get(f"{BASE}/fixtures",headers=F_HEADERS,
                        params={"date":date_str},timeout=15)
        data = r.json().get("response",[])
        print(f"  Found {len(data)} fixtures for {date_str}")
        return data
    except Exception as e:
        print(f"  Fixtures err: {e}")
        return []

def get_team_players_api(team_id, league_id):
    try:
        r = requests.get(f"{BASE}/players",headers=F_HEADERS,
                        params={"team":team_id,"season":2025,"league":league_id},timeout=15)
        data = r.json().get("response",[])
        result = []
        for p in data:
            st   = p.get("statistics",[{}])[0]
            pos  = st.get("games",{}).get("position","") or ""
            apps = st.get("games",{}).get("appearences",0) or 0
            mins = st.get("games",{}).get("minutes",0) or 0
            goals= st.get("goals",{}).get("total",0) or 0
            if pos in ["Attacker","Midfielder"] and apps>=3:
                result.append({
                    "name":p["player"]["name"],"position":pos,
                    "avg_sot":  max(0.3, goals/max(apps,1)*0.8),
                    "avg_shots":max(0.5, goals/max(apps,1)*2.0),
                    "avg_xg":   goals/max(apps,1)*0.4,
                    "avg_mins": round(mins/max(apps,1)),
                    "games":    apps,
                    "source":   "api"
                })
        result.sort(key=lambda x: x["avg_sot"], reverse=True)
        return result[:5]
    except: return []

def enrich_with_understat(players, us_data):
    for p in players:
        nl = p["name"].lower()
        for key,val in us_data.items():
            if any(part in key for part in nl.split() if len(part)>3):
                p.update({"avg_sot":val["avg_sot"],"avg_shots":val["avg_shots"],
                          "avg_xg":val["avg_xg"],"avg_mins":val["avg_mins"],
                          "games":val["games"],"us_id":val["id"],"source":"understat"})
                break
    return players

def enrich_with_fbref(players, fbref_data):
    for p in players:
        if p.get("source") in ["understat"]: continue
        nl = p["name"].lower()
        for key,val in fbref_data.items():
            if any(part in key for part in nl.split() if len(part)>3):
                p.update({"avg_sot":val["avg_sot"],"avg_shots":val["avg_shots"],
                          "avg_xg":val["avg_xg"],"avg_mins":val["avg_mins"],
                          "games":val["games"],"source":"fbref"})
                break
    return players

def process_fixtures(fixtures, day_label, match_date_str, history=None):
    if history is None: history = {}
    matches = []
    count   = 0
    for fix in fixtures:
        lid = fix["league"]["id"]
        if lid not in LEAGUES or count>=8: continue
        count += 1
        lg         = LEAGUES[lid]
        tier       = lg["tier"]
        home_id    = fix["teams"]["home"]["id"]
        away_id    = fix["teams"]["away"]["id"]
        home_name  = fix["teams"]["home"]["name"]
        away_name  = fix["teams"]["away"]["name"]
        kickoff    = fix["fixture"]["date"][11:16]
        fixture_id = fix["fixture"]["id"]
        lg_name    = fix["league"]["name"].lower()
        is_hv      = any(w in lg_name for w in ["cup","trophy","shield","supercopa","derby"])

        # Data sources based on tier
        us_data    = get_understat_players(lg.get("understat","")) if tier=="A" else {}
        fbref_data = get_fbref_players(lg.get("fbref",""))         if tier=="B" else {}

        # ── Real odds from The Odds API ──
        sport_key  = THEODDS_LEAGUES.get(lid,"")
        event_id   = match_event_id(sport_key, home_name, away_name) if sport_key else ""
        match_odds = get_match_h2h_odds(sport_key, home_name, away_name) if sport_key else {"home":1.9,"away":2.1,"draw":3.4,"found":False}

        # ── Real player SOT lines ──
        sot_lines = {}
        if sport_key and event_id and day_label in ["today","tomorrow"]:
            sot_lines = get_player_sot_lines(sport_key, event_id, home_name, away_name)
            if sot_lines:
                print(f"  ✅ Real SOT lines: {home_name} vs {away_name}")
            else:
                print(f"  ⚠️  No SOT lines found for {home_name} vs {away_name} — using estimates")

        # Lineups + injuries (today/tomorrow)
        lineups, lu_confirmed = {}, False
        injuries = {}
        if day_label in ["today","tomorrow"]:
            lineups, lu_confirmed = get_lineups(fixture_id)
            injuries = get_injuries(fixture_id)

        # Get players for both teams
        all_players = []
        for team_id, t_att, o_def in [
            (home_id, 0.83, 0.68),
            (away_id, 0.76, 0.70)
        ]:
            players = get_team_players_api(team_id, lid)
            if tier=="A": players = enrich_with_understat(players, us_data)
            if tier=="B": players = enrich_with_fbref(players, fbref_data)

            for p in players:
                src     = p.get("source","api")
                role    = p["position"]
                xsot    = calc_xsot(p["avg_sot"],p["avg_shots"],p["avg_xg"],
                                   role,t_att,o_def,p["avg_mins"])
                line    = 1.5 if p["avg_sot"]>=1.5 else 0.5
                prob    = calc_prob(xsot, line)
                conf    = calc_conf(prob,p["avg_sot"],p["avg_mins"],p["games"],tier)
                # Resolve real or estimated player SOT odds
                p_odds  = resolve_player_odds(p["name"], sot_lines, prob, match_odds)
                curr_o  = p_odds["current"]
                real_odds = p_odds["realOdds"]
                # If real SOT line found, update the betting line too
                if real_odds and p_odds["line"] is not None:
                    line = p_odds["line"]
                    prob = calc_prob(xsot, line)  # recalc with real line

                # ── UPGRADE 3: Player specific modifiers ──
                xsot = apply_player_modifiers(xsot, p["avg_sot"], p["avg_shots"], p["avg_xg"])
                prob = calc_prob(xsot, line)  # recalc after modifiers

                # ── UPGRADE 4: Public bias filter ──
                prob = apply_public_bias(prob, p["name"], lg["key"])

                # ── UPGRADE 2: Calibration layer ──
                calib_factor = get_calibration_factor(
                    history.get("summary",{}),
                    "SAFE" if prob >= 0.75 else "RISKY"
                )
                prob = apply_calibration(prob, calib_factor)

                # Recalc EV and implied after all probability adjustments
                ev      = calc_ev(prob, curr_o)
                implied = round(1/curr_o, 3)

                # Status
                status, reason = player_status(p["name"], lineups, injuries)
                if status=="injured": prob=0.0; conf=1
                elif status=="sub":   prob=round(prob*0.6,3); conf=max(1,conf-2)

                # Recalc confidence after adjustments
                conf = calc_conf(prob, p["avg_sot"], p["avg_mins"], p["games"], tier)

                # ── UPGRADE 5: Tight verdict system ──
                verdict = get_tight_verdict(
                    prob, conf, ev, is_hv, status,
                    real_odds, calib_factor, lg["key"]
                )

                # Form history
                form, history_detail = [1 if i%2==0 else 0 for i in range(6)], []
                us_id = p.get("us_id")
                if us_id and tier=="A":
                    form, history_detail = us_player_history(us_id, p["name"])

                # Yesterday actual result
                actual = None
                if day_label=="yesterday" and us_id and tier=="A":
                    actual = us_yesterday_sot(us_id, match_date_str)

                # ── UPGRADE 5: Tight banker criteria ──
                banker = is_tight_banker(prob, conf, ev, p["avg_sot"], form, real_odds)

                # ── Modifier metadata for display ──
                fin_eff  = calc_finishing_efficiency(p["avg_sot"], p["avg_shots"])
                overperf = calc_overperformance_trend(p["avg_sot"], p["avg_xg"])

                entry = {
                    "name":    p["name"],
                    "role":    "ST" if role in ["Attacker","Forward"] else "CAM",
                    "minutes": p["avg_mins"],
                    "line":    line,
                    "xsot":    xsot,
                    "prob":    round(prob,3),
                    "conf":    conf,
                    "oddsOpen":    p_odds["open"],
                    "oddsCurrent": curr_o,
                    "oddsMoved":   p_odds["moved"],
                    "oddsPctMove": p_odds["pctMove"],
                    "realOdds":    p_odds["realOdds"],
                    "bestBook":    p_odds.get("best_book",""),
                    "bestLine":    p_odds.get("best_line", line),
                    "books":       p_odds.get("books",{}),
                    "lines":       p_odds.get("lines",{}),
                    "ev":      ev,
                    "isValue": (prob-implied)>0.05,
                    "verdict": verdict,
                    "banker":  banker,
                    "status":  status,
                    "statusReason": reason,
                    "lineupConfirmed": lu_confirmed,
                    "form":    form if form else [0,1,0,1,0,1],
                    "avgSot":  p["avg_sot"],
                    "avgXg":   p["avg_xg"],
                    "dataSource": src,
                    "dataTier":   tier,
                    "matchHistory": history_detail[:6],
                    "calibFactor":  calib_factor,
                    "finEfficiency": round(fin_eff, 2),
                    "overperf":     round(overperf, 2),
                    "publicBias":   p["name"].lower() in " ".join(PUBLIC_BIAS_PLAYERS.keys()) or lg["key"] in PUBLIC_BIAS_LEAGUES,
                }
                if actual:
                    entry.update({"actualSOT":actual["actualSOT"],
                                  "actualShots":actual["actualShots"],
                                  "actualXG":actual["actualXG"],
                                  "hit":actual["hit"]})
                all_players.append(entry)

        # Sort: bankers first, then by verdict + prob
        def sort_key(p):
            v = {"SAFE":3,"RISKY":2,"AVOID":1}.get(p["verdict"],0)
            return (int(p["banker"]), v, p["prob"])
        all_players.sort(key=sort_key, reverse=True)

        matches.append({
            "id":fixture_id, "league":lg["key"],
            "leagueName":lg["name"], "country":lg.get("country",""),
            "dataTier":tier,
            "day":day_label,
            "homeTeam":home_name,"awayTeam":away_name,
            "kickoff":kickoff,
            "homeAttack":0.83,"awayAttack":0.76,
            "homeDefense":0.70,"awayDefense":0.68,
            "isLive":False,"isHighVariance":is_hv,
            "lineupConfirmed":lu_confirmed,
            "matchOdds":match_odds,
            "players":all_players[:8],
            # Internal fields for CLV tracking
            "_lid":lid,
            "_event_id":event_id,
        })
    return matches

# ══════════════════════════════════════════
# GEMINI AI
# ══════════════════════════════════════════

def ai_insight(today_matches, hist_summary):
    safe=[]; risky=[]; bankers=[]
    for m in today_matches:
        for p in m.get("players",[]):
            if p["status"]=="injured": continue
            fr = sum(p.get("form",[]))/max(len(p.get("form",[1])),1)
            entry=(f"{p['name']} ({m['leagueName']}, {p['prob']*100:.0f}% prob, "
                   f"{p['avgSot']:.2f} avg SOT, {fr*100:.0f}% form, "
                   f"odds {p['oddsCurrent']}, EV {p['ev']*100:.1f}%)")
            if p.get("banker"): bankers.append(entry)
            elif p["verdict"]=="SAFE": safe.append(entry)
            elif p["verdict"]=="RISKY": risky.append(entry)

    hs = hist_summary or {}
    track = ""
    if hs.get("safe"):
        track = (f"Model track record: Safe hit rate {hs['safe'].get('rate',0)}%, "
                f"Banker hit rate {hs.get('banker',{}).get('rate','N/A')}%, "
                f"ROI {hs.get('roi',0)}% over {hs.get('daysTracked',0)} days.")

    prompt = f"""You are an elite football shots-on-target betting analyst with access to real per-game SOT data.
{track}
BANKER picks: {', '.join(bankers[:3]) if bankers else 'None'}
Safe picks: {', '.join(safe[:4]) if safe else 'None'}
Risky picks: {', '.join(risky[:2]) if risky else 'None'}

Write exactly 3 sentences:
1. Best banker/safe pick today citing the stats
2. Key risk or match to avoid
3. Confidence level in today's overall card
Max 90 words. Be sharp and specific."""
    for attempt in range(3):
        try:
            resp = model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                print(f"  Gemini rate limit hit, waiting 30s...")
                time.sleep(30)
            elif "404" in err:
                print(f"  Gemini model not found: {err}")
                return "AI analysis unavailable: model not found. Check GEMINI_API_KEY and model name."
            else:
                print(f"  Gemini error: {err}")
                return f"AI analysis unavailable: {err}"
    return "AI analysis unavailable: quota exceeded after retries."

def ai_results_summary(y_matches):
    hits=[]; misses=[]
    for m in y_matches:
        for p in m.get("players",[]):
            if "actualSOT" not in p: continue
            e = f"{p['name']}: {p['actualSOT']} SOT ({'BANKER ' if p.get('banker') else ''}{p['verdict']})"
            (hits if p.get("hit") else misses).append(e)
    if not hits and not misses: return ""
    prompt = f"""Yesterday's SOT results:
Hits ✓: {', '.join(hits[:5]) if hits else 'None'}
Misses ✗: {', '.join(misses[:5]) if misses else 'None'}
2 sentences: performance summary and key lesson. Max 50 words."""
    try:
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        print(f"  Results summary AI err: {e}")
        return ""

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════

print("="*55)
print(f"SOTIQ Bot — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("="*55)
# Safety: limit total run time to avoid GitHub Actions timeout
import signal
def timeout_handler(signum, frame):
    print("\n⚠️  Run time limit reached — saving partial data")
    raise SystemExit(0)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute hard limit

print("\nLoading history...")
history = load_history()
print(f"  {len(history.get('predictions',[]))} predictions tracked")

all_matches = []

print("\n[1/3] Yesterday + results...")
y_fixes   = get_fixtures(fmt(yesterday))
y_matches = process_fixtures(y_fixes,"yesterday",fmt(yesterday),history)
all_matches += y_matches
print(f"  {len(y_matches)} matches")

print("\n[2/3] Today + predictions...")
t_fixes   = get_fixtures(fmt(today))
t_matches = process_fixtures(t_fixes,"today",fmt(today),history)
all_matches += t_matches
print(f"  {len(t_matches)} matches")

print("\n[3/3] Tomorrow...")
tm_fixes   = get_fixtures(fmt(tomorrow))
tm_matches = process_fixtures(tm_fixes,"tomorrow",fmt(tomorrow),history)
all_matches += tm_matches
print(f"  {len(tm_matches)} matches")

# Update history with results
y_all = [m for m in all_matches if m["day"]=="yesterday"]
history = update_history(y_all, history)

# ── UPGRADE 1: CLV — fetch closing odds for yesterday's matches ──
print("\nFetching closing odds for CLV...")
closing_all = {}
for m in y_all:
    sport_key = THEODDS_LEAGUES.get(m.get("_lid",0),"")
    # Use stored event_id if available
    ev_id = m.get("_event_id","")
    if sport_key and ev_id:
        player_names = [p["name"].lower() for p in m.get("players",[])]
        closing = fetch_closing_odds(sport_key, ev_id, player_names)
        closing_all.update(closing)

if closing_all:
    history = update_clv_in_history(history, closing_all)
    print(f"  CLV data: {len(closing_all)} players")
else:
    print("  No closing odds available yet (matches may not have started)")

save_history(history)

# AI
print("\nGenerating AI insights...")
t_all = [m for m in all_matches if m["day"]=="today"]
insight         = ai_insight(t_all, history.get("summary",{}))
results_summary = ai_results_summary(y_all)
print("  Done!")

# Stats
all_p    = [p for m in t_all for p in m.get("players",[])]
safe_c   = len([p for p in all_p if p["verdict"]=="SAFE"])
banker_c = len([p for p in all_p if p.get("banker")])
ev_c     = len([p for p in all_p if p["ev"]>0])
value_c  = len([p for p in all_p if p["isValue"]])
avoid_c  = len([p for p in all_p if p["verdict"]=="AVOID"])

y_p      = [p for m in y_all for p in m.get("players",[]) if "actualSOT" in p]
y_hits   = len([p for p in y_p if p.get("hit")])
hit_rate = round(y_hits/len(y_p)*100) if y_p else None

print(f"\nToday — Safe:{safe_c} Bankers:{banker_c} EV+:{ev_c} Value:{value_c} Avoid:{avoid_c}")
if hit_rate is not None:
    print(f"Yesterday — {y_hits}/{len(y_p)} hit ({hit_rate}%)")

clv_sum = history.get("summary",{}).get("clv",{})
output = {
    "updated":        datetime.utcnow().isoformat()+"Z",
    "aiInsight":      insight,
    "resultsSummary": results_summary,
    "hitRate":        hit_rate,
    "historySummary": history.get("summary",{}),
    "clvSummary":     clv_sum,
    "matches":        all_matches
}

with open("data.json","w") as f:
    json.dump(output, f, indent=2)

print("\n✅ data.json + history.json saved!")
print("="*55)
