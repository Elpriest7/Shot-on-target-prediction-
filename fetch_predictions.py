import os, json, requests, math, re, time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import google.generativeai as genai

# ── KEYS ──
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY")
FOOTBALL_KEY = os.environ.get("FOOTBALL_API_KEY")
ODDS_KEY     = os.environ.get("ODDS_API_KEY")

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel("gemini-1.5-flash-latest")

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

LEAGUES = {
    # API-Football ID : config
    39:  {"key":"pl",          "name":"Premier League",      "country":"England",     "tier":"A", "understat":"EPL",         "fbref":"9",  "odds":"soccer_epl"},
    140: {"key":"laliga",      "name":"La Liga",             "country":"Spain",       "tier":"A", "understat":"La_liga",     "fbref":"12", "odds":"soccer_spain_la_liga"},
    78:  {"key":"bundesliga",  "name":"Bundesliga",          "country":"Germany",     "tier":"A", "understat":"Bundesliga",  "fbref":"20", "odds":"soccer_germany_bundesliga"},
    135: {"key":"seriea",      "name":"Serie A",             "country":"Italy",       "tier":"A", "understat":"Serie_A",     "fbref":"11", "odds":"soccer_italy_serie_a"},
    61:  {"key":"ligue1",      "name":"Ligue 1",             "country":"France",      "tier":"A", "understat":"Ligue_1",     "fbref":"13", "odds":"soccer_france_ligue_one"},
    # FBref tier B
    2:   {"key":"ucl",         "name":"Champions League",    "country":"Europe",      "tier":"B", "fbref":"8",  "odds":"soccer_uefa_champs_league"},
    3:   {"key":"uel",         "name":"Europa League",       "country":"Europe",      "tier":"B", "fbref":"19", "odds":"soccer_uefa_europa_league"},
    848: {"key":"uecl",        "name":"Conference League",   "country":"Europe",      "tier":"B", "fbref":"882","odds":"soccer_uefa_europa_conference_league"},
    # API-Football tier C (season averages)
    40:  {"key":"championship","name":"Championship",        "country":"England",     "tier":"C", "odds":"soccer_england_championship"},
    41:  {"key":"league1",     "name":"League One",          "country":"England",     "tier":"C"},
    88:  {"key":"eredivisie",  "name":"Eredivisie",          "country":"Netherlands", "tier":"C", "odds":"soccer_netherlands_eredivisie"},
    144: {"key":"jupiler",     "name":"Jupiler Pro League",  "country":"Belgium",     "tier":"C", "odds":"soccer_belgium_first_div_a"},
    203: {"key":"superlig",    "name":"Süper Lig",           "country":"Turkey",      "tier":"C", "odds":"soccer_turkey_super_league"},
    94:  {"key":"primavera",   "name":"Primeira Liga",       "country":"Portugal",    "tier":"C", "odds":"soccer_portugal_primeira_liga"},
    119: {"key":"superligadk", "name":"Superliga",           "country":"Denmark",     "tier":"C"},
    113: {"key":"allsvenskan", "name":"Allsvenskan",         "country":"Sweden",      "tier":"C"},
    235: {"key":"rpl",         "name":"Russian PL",          "country":"Russia",      "tier":"C"},
    197: {"key":"csl",         "name":"Super Lig",           "country":"Greece",      "tier":"C"},
    218: {"key":"ligamx",      "name":"Liga MX",             "country":"Mexico",      "tier":"C"},
    253: {"key":"mls",         "name":"MLS",                 "country":"USA",         "tier":"C", "odds":"soccer_usa_mls"},
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
        raw  = match.group(1).encode('utf-8').decode('unicode_escape')
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
        raw = m.group(1).encode('utf-8').decode('unicode_escape')
        matches = [x for x in json.loads(raw) if x.get("season")=="2025"]
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
        raw = m.group(1).encode('utf-8').decode('unicode_escape')
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
    "8":  "https://fbref.com/en/comps/8/shooting/Champions-League-Stats",
    "19": "https://fbref.com/en/comps/19/shooting/Europa-League-Stats",
    "882":"https://fbref.com/en/comps/882/shooting/UEFA-Europa-Conference-League-Stats",
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
        table = soup.find("table", {"id": re.compile("stats_shooting")})
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
# MATH ENGINE
# ══════════════════════════════════════════

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
# ODDS
# ══════════════════════════════════════════

odds_api_cache = {}

def get_match_odds(odds_key, home, away):
    if not ODDS_KEY or not odds_key: 
        return {"home":1.9,"away":2.1,"draw":3.4,"found":False}
    try:
        if odds_key not in odds_api_cache:
            r = requests.get(
                f"https://api.odds-api.io/v4/sports/{odds_key}/odds",
                params={"apiKey":ODDS_KEY,"regions":"uk","markets":"h2h","oddsFormat":"decimal"},
                timeout=15)
            odds_api_cache[odds_key] = r.json() if r.status_code==200 else []
        for ev in odds_api_cache.get(odds_key,[]):
            eh = ev.get("home_team","").lower()
            ea = ev.get("away_team","").lower()
            if (any(w in eh for w in home.lower().split()[:2]) or
                any(w in ea for w in away.lower().split()[:2])):
                bk = ev.get("bookmakers",[])
                if bk:
                    outs = bk[0].get("markets",[{}])[0].get("outcomes",[])
                    om = {o["name"].lower():o["price"] for o in outs}
                    return {"home": list(om.values())[0] if om else 1.9,
                            "away": list(om.values())[1] if len(om)>1 else 2.1,
                            "draw": om.get("draw",3.4), "found":True}
        return {"home":1.9,"away":2.1,"draw":3.4,"found":False}
    except Exception as e:
        return {"home":1.9,"away":2.1,"draw":3.4,"found":False}

def player_odds(prob, match_odds):
    fair    = round(1/max(prob,0.01), 2)
    margin  = 0.08
    avg_mo  = (match_odds["home"]+match_odds["away"])/2
    drift   = (prob-0.5)*0.3 - (0.05/avg_mo)
    open_o  = max(1.15, min(round(fair*(1-margin*0.5),2), 5.0))
    curr_o  = max(1.10, min(round(open_o-drift,2), 5.0))
    moved   = "up" if curr_o<open_o else "down" if curr_o>open_o else "flat"
    return {"open":open_o,"current":curr_o,"moved":moved,
            "pctMove":abs(round((curr_o-open_o)/open_o*100,1))}

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
                "hit":p.get("hit",False),"banker":p.get("banker",False)
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
    return {
        "safe":stats(safe_p),"risky":stats(risky_p),
        "banker":stats(banker_p),"all":stats(preds),
        "roi":roi,"totalPredictions":len(preds),
        "daysTracked":len(set(p["date"] for p in preds))
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

def process_fixtures(fixtures, day_label, match_date_str):
    matches = []
    count   = 0
    for fix in fixtures:
        lid = fix["league"]["id"]
        if lid not in LEAGUES or count>=6: continue
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

        # Real odds
        match_odds = get_match_odds(lg.get("odds",""), home_name, away_name)

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
                p_odds  = player_odds(prob, match_odds)
                curr_o  = p_odds["current"]
                ev      = calc_ev(prob, curr_o)
                implied = round(1/curr_o, 3)

                # Status
                status, reason = player_status(p["name"], lineups, injuries)
                if status=="injured": prob=0.0; conf=1
                elif status=="sub":   prob=round(prob*0.6,3); conf=max(1,conf-2)

                verdict = get_verdict(prob,conf,ev,is_hv,status)

                # Form history
                form, history_detail = [1 if i%2==0 else 0 for i in range(6)], []
                us_id = p.get("us_id")
                if us_id and tier=="A":
                    form, history_detail = us_player_history(us_id, p["name"])

                # Yesterday actual result
                actual = None
                if day_label=="yesterday" and us_id and tier=="A":
                    actual = us_yesterday_sot(us_id, match_date_str)

                # Banker detection
                banker = is_banker(prob,conf,ev,p["avg_sot"],form)

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
                    "matchHistory": history_detail[:6]
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
            "players":all_players[:8]
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

    prompt = f"""You are a sharp football shots-on-target betting analyst.
{track}
BANKER picks: {', '.join(bankers[:3]) if bankers else 'None'}
Safe picks: {', '.join(safe[:4]) if safe else 'None'}
Risky picks: {', '.join(risky[:2]) if risky else 'None'}

Write exactly 3 sentences:
1. Best banker/safe pick today citing the stats
2. Key risk or match to avoid
3. Confidence level in today's overall card
Max 90 words. Be sharp and specific."""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

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
        return model.generate_content(prompt).text.strip()
    except: return ""

# ══════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════

print("="*55)
print(f"SOTIQ Bot — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
print("="*55)

print("\nLoading history...")
history = load_history()
print(f"  {len(history.get('predictions',[]))} predictions tracked")

all_matches = []

print("\n[1/3] Yesterday + results...")
y_fixes   = get_fixtures(fmt(yesterday))
y_matches = process_fixtures(y_fixes,"yesterday",fmt(yesterday))
all_matches += y_matches
print(f"  {len(y_matches)} matches")

print("\n[2/3] Today + predictions...")
t_fixes   = get_fixtures(fmt(today))
t_matches = process_fixtures(t_fixes,"today",fmt(today))
all_matches += t_matches
print(f"  {len(t_matches)} matches")

print("\n[3/3] Tomorrow...")
tm_fixes   = get_fixtures(fmt(tomorrow))
tm_matches = process_fixtures(tm_fixes,"tomorrow",fmt(tomorrow))
all_matches += tm_matches
print(f"  {len(tm_matches)} matches")

# Update history
y_all = [m for m in all_matches if m["day"]=="yesterday"]
history = update_history(y_all, history)
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

output = {
    "updated":        datetime.utcnow().isoformat()+"Z",
    "aiInsight":      insight,
    "resultsSummary": results_summary,
    "hitRate":        hit_rate,
    "historySummary": history.get("summary",{}),
    "matches":        all_matches
}

with open("data.json","w") as f:
    json.dump(output, f, indent=2)

print("\n✅ data.json + history.json saved!")
print("="*55)
