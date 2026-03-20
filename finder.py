"""S&P 500 Market Leader Screener
================================
Swing Leader vs Benchmark — Full Relative Strength Stack (RS 1-4)

Translates the PineScript indicator into a Python screener that evaluates
all S&P 500 constituents and serves filtered results via a local web dashboard.

Usage
-----
    pip install -r requirements.txt
    python finder.py              # → http://127.0.0.1:5000
    python finder.py --port 8080
"""

import argparse
import datetime
import os
import sqlite3
import ssl
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def _make_session():
    """Create an SSL-verification-disabled session compatible with yfinance."""
    try:
        from curl_cffi.requests import Session as CfSession
        return CfSession(verify=False, impersonate="chrome131")
    except ImportError:
        import requests as _rq
        s = _rq.Session()
        s.verify = False
        return s

_YF_SESSION = _make_session()

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template_string


def _log(msg):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode(), flush=True)

# ═══════════════════════════════════════════════════════════════════
# Parameters (mirror PineScript inputs)
# ═══════════════════════════════════════════════════════════════════
BENCH = "SPY"
W_FAST = 5
W_SLOW = 10
W_JUDGE = 20
W_FILTER = 60
W_BETA = 252
K_MOM = 20
LOG_RET = True
BETA_ADJ = False
ANN = 252.0
INITIAL_PERIOD = "3y"
MIN_BARS = W_BETA + 2 * K_MOM + 10
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prices.db")
DL_DELAY = 0.35


# ═══════════════════════════════════════════════════════════════════
# S&P 500 constituent list
# ═══════════════════════════════════════════════════════════════════
def get_sp500():
    _log("[1/3] Fetching S&P 500 list ...")
    try:
        import requests as _req

        resp = _req.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
            verify=False,
        )
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        df = tables[0]
        df["sym"] = df["Symbol"].str.replace(".", "-", regex=False)
        info = {
            row["sym"]: {
                "name": row.get("Security", ""),
                "sector": row.get("GICS Sector", ""),
            }
            for _, row in df.iterrows()
        }
        _log(f"    {len(info)} constituents")
        return list(info.keys()), info
    except Exception as exc:
        _log(f"[!] Failed to fetch S&P 500 list: {exc}")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# SQLite price cache
# ═══════════════════════════════════════════════════════════════════
def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS prices "
        "(ticker TEXT NOT NULL, date TEXT NOT NULL, close REAL, "
        "PRIMARY KEY (ticker, date))"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_tk ON prices(ticker)")
    conn.commit()
    return conn


def _save_close(conn, ticker, series):
    if series is None or series.empty:
        return
    rows = [
        (ticker, d.strftime("%Y-%m-%d"), float(v))
        for d, v in series.items()
        if pd.notna(v)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO prices(ticker,date,close) VALUES(?,?,?)", rows
    )
    conn.commit()


def _download_one(sym, period=None, start=None):
    try:
        tk = yf.Ticker(sym, session=_YF_SESSION)
        kw = {"auto_adjust": True}
        if start:
            kw["start"] = start
        else:
            kw["period"] = period or INITIAL_PERIOD
        hist = tk.history(**kw)
        if not hist.empty and "Close" in hist.columns:
            return hist["Close"]
    except Exception:
        pass
    return None


def sync_prices(tickers):
    """First run: download 3 yr history → SQLite.
    Subsequent runs: only fetch the gap since the last stored date."""
    conn = _init_db()
    syms = sorted(set(tickers + [BENCH]))

    rows = conn.execute(
        "SELECT ticker, MAX(date) FROM prices GROUP BY ticker"
    ).fetchall()
    db_last = {r[0]: r[1] for r in rows}

    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)

    need_full, need_gap, up_to_date = [], [], []
    for sym in syms:
        if sym not in db_last:
            need_full.append(sym)
        else:
            last = datetime.datetime.strptime(db_last[sym], "%Y-%m-%d").date()
            if last >= yesterday:
                up_to_date.append(sym)
            else:
                need_gap.append((sym, last))

    _log(
        f"[2/3] Price sync - "
        f"{len(up_to_date)} current, "
        f"{len(need_gap)} need update, "
        f"{len(need_full)} new"
    )

    # ── full download for tickers not in DB ────────────────────────
    if need_full:
        _log(f"    Downloading {len(need_full)} new tickers ({INITIAL_PERIOD}) ...")
        ok = fail = 0
        for i, sym in enumerate(need_full):
            s = _download_one(sym, period=INITIAL_PERIOD)
            if s is not None:
                _save_close(conn, sym, s)
                ok += 1
            else:
                fail += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(need_full):
                _log(f"      {i+1}/{len(need_full)}  ({ok} ok, {fail} fail)")
            time.sleep(DL_DELAY)

    # ── incremental update for existing tickers ────────────────────
    if need_gap:
        _log(f"    Updating {len(need_gap)} tickers ...")
        ok = fail = 0
        for i, (sym, last) in enumerate(need_gap):
            start = (last + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            s = _download_one(sym, start=start)
            if s is not None:
                _save_close(conn, sym, s)
                ok += 1
            else:
                fail += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(need_gap):
                _log(f"      {i+1}/{len(need_gap)}  ({ok} ok, {fail} fail)")
            time.sleep(DL_DELAY)

    # ── load everything from DB ────────────────────────────────────
    _log("    Loading prices from local DB ...")
    ph = ",".join(["?"] * len(syms))
    df = pd.read_sql(
        f"SELECT ticker, date, close FROM prices WHERE ticker IN ({ph})",
        conn,
        params=syms,
    )
    conn.close()

    if df.empty:
        _log("[!] No price data in DB")
        sys.exit(1)

    df["date"] = pd.to_datetime(df["date"])
    close = df.pivot(index="date", columns="ticker", values="close").sort_index()
    good = close.columns[close.count() >= MIN_BARS]
    close = close[good]
    _log(f"    {close.shape[0]} days x {close.shape[1]} symbols ready")
    return close


# ═══════════════════════════════════════════════════════════════════
# Core screening logic — fully vectorised across all stocks
# ═══════════════════════════════════════════════════════════════════
def screen(close: pd.DataFrame, beta_adj: bool = BETA_ADJ) -> pd.DataFrame:
    if BENCH not in close.columns:
        _log(f"[!] Benchmark '{BENCH}' not found in downloaded data")
        sys.exit(1)

    pb = close[BENCH]
    cols = [c for c in close.columns if c != BENCH]
    p = close[cols]

    # ── returns ────────────────────────────────────────────────────
    if LOG_RET:
        r = np.log(p / p.shift(1))
        rb = np.log(pb / pb.shift(1))
    else:
        r = p.pct_change()
        rb = pb.pct_change()

    # ── rolling beta (W_BETA window) ──────────────────────────────
    mr = r.rolling(W_BETA).mean()
    mrb = rb.rolling(W_BETA).mean()
    cov = (r.mul(rb, axis=0)).rolling(W_BETA).mean() - mr.mul(mrb, axis=0)
    vb = (rb**2).rolling(W_BETA).mean() - mrb**2
    beta = cov.div(vb, axis=0).replace([np.inf, -np.inf], np.nan)

    # ── daily excess return (raw or beta-adjusted) ────────────────
    e_raw = r.sub(rb, axis=0)
    if beta_adj:
        e = (r - beta.mul(rb, axis=0)).where(beta.notna(), e_raw)
    else:
        e = e_raw

    # ── cumulative RS by window ───────────────────────────────────
    RS = {w: e.rolling(w).sum() for w in (W_FILTER, W_JUDGE, W_SLOW, W_FAST)}

    # ── information ratio (consistency) ───────────────────────────
    def _ir(series, w):
        return (series.rolling(w).mean() / series.rolling(w).std()) * np.sqrt(ANN)

    ir20 = _ir(e, W_JUDGE)
    ir60 = _ir(e, W_FILTER)

    # ── alpha (annualised, filter window) ─────────────────────────
    mr60 = r.rolling(W_FILTER).mean()
    mrb60 = rb.rolling(W_FILTER).mean()
    alpha_d = mr60 - beta.mul(mrb60, axis=0)
    alpha_a = alpha_d * ANN

    # ── alpha consistency ─────────────────────────────────────────
    eps = r - (alpha_d + beta.mul(rb, axis=0))
    alpha_ir = _ir(eps, W_FILTER)

    # ── trend / momentum / acceleration ───────────────────────────
    L = np.log(p).sub(np.log(pb), axis=0)
    M = L - L.shift(K_MOM)
    A = M - M.shift(K_MOM)

    # ── latest snapshot → DataFrame ───────────────────────────────
    _pct = (lambda x: (np.exp(x) - 1) * 100) if LOG_RET else (lambda x: x * 100)

    o = pd.DataFrame(index=cols)
    for tag, w in [("60", W_FILTER), ("20", W_JUDGE), ("10", W_SLOW), ("5", W_FAST)]:
        o[f"rs_{tag}_raw"] = RS[w].iloc[-1]
        o[f"rs_{tag}"] = _pct(RS[w].iloc[-1])

    o["ir_60"] = ir60.iloc[-1]
    o["ir_20"] = ir20.iloc[-1]
    o["beta"] = beta.iloc[-1]
    o["alpha_ann"] = alpha_a.iloc[-1] * 100
    o["alpha_ir"] = alpha_ir.iloc[-1]
    o["mom"] = M.iloc[-1]
    o["acc"] = A.iloc[-1]

    # ── swing rules ───────────────────────────────────────────────
    o["filter_pass"] = o["rs_60_raw"] > 0
    o["judge_pass"] = o["rs_20_raw"] > 0
    o["alarm_fast"] = o["rs_5_raw"] < 0
    o["alarm_slow"] = o["rs_10_raw"] < 0
    o["early_trend"] = (o["mom"] > 0) & (o["acc"] > 0)
    o["consistency_pass"] = o["ir_20"] > 0
    o["leader"] = (
        o["filter_pass"] & o["judge_pass"] & ~o["alarm_fast"] & ~o["alarm_slow"]
        & o["consistency_pass"]
    )

    o["status"] = np.select(
        [
            o["leader"],
            o["early_trend"] & o["judge_pass"],
            (o["alarm_fast"] | o["alarm_slow"]) & o["judge_pass"],
        ],
        ["LEADER", "Early Trend", "Alarm"],
        "Not Leading",
    )

    o = o.dropna(subset=["rs_60", "rs_20"])

    rs20_pct = o["rs_20"].rank(pct=True, na_option="bottom")
    ir20_pct = o["ir_20"].rank(pct=True, na_option="bottom")
    o["composite"] = 0.5 * rs20_pct + 0.5 * ir20_pct

    rank = {"LEADER": 0, "Early Trend": 1, "Alarm": 2, "Not Leading": 3}
    o["_r"] = o["status"].map(rank)
    o = o.sort_values(["_r", "composite"], ascending=[True, False]).drop(columns=["_r"])
    o.index.name = "ticker"

    cnt = o["status"].value_counts()
    _log(
        f"    Done - {int(cnt.get('LEADER', 0))} leaders, "
        f"{int(cnt.get('Early Trend', 0))} early trend, "
        f"{int(cnt.get('Alarm', 0))} alarm, "
        f"{int(cnt.get('Not Leading', 0))} not leading"
    )
    return o


# ═══════════════════════════════════════════════════════════════════
# Flask dashboard
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__)
G: dict = {}

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>S&amp;P 500 Leader Screener</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background:#f0f2f5;color:#1a1a2e}
.hdr{background:linear-gradient(135deg,#1e293b 0%,#334155 100%);color:#fff;padding:2rem 2rem 1.6rem}
.hdr h1{font-size:1.7rem;font-weight:700}
.hdr p{opacity:.75;margin-top:.25rem;font-size:.92rem}
.wrap{max-width:1700px;margin:0 auto;padding:1.4rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:.9rem;margin-bottom:1.3rem}
.card{background:#fff;border-radius:12px;padding:1.1rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.card .lb{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em}
.card .vl{font-size:1.7rem;font-weight:700;margin-top:.2rem}
.card.c-ld .vl{color:#059669}
.card.c-et .vl{color:#0d9488}
.card.c-al .vl{color:#d97706}
.filters{display:flex;gap:.45rem;margin-bottom:1rem;flex-wrap:wrap}
.filters button{padding:.45rem .95rem;border:1px solid #e2e8f0;border-radius:8px;background:#fff;cursor:pointer;font-size:.82rem;transition:all .15s}
.filters button:hover{background:#f1f5f9}
.filters button.on{background:#1e293b;color:#fff;border-color:#1e293b}
.tw{background:#fff;border-radius:12px;box-shadow:0 1px 3px rgba(0,0,0,.08);overflow-x:auto}
table{width:100%;border-collapse:collapse;font-size:.82rem}
thead{background:#f8fafc;position:sticky;top:0;z-index:2}
th{padding:.7rem .55rem;text-align:left;font-weight:600;color:#475569;cursor:pointer;user-select:none;white-space:nowrap;border-bottom:2px solid #e2e8f0}
th:hover{color:#1e293b}
th .si{margin-left:3px;opacity:.35;font-size:.7rem}
td{padding:.55rem;border-bottom:1px solid #f1f5f9;white-space:nowrap}
tr:hover td{background:#f8fafc}
.pos{color:#059669}.neg{color:#dc2626}
.badge{display:inline-block;padding:.18rem .55rem;border-radius:6px;font-size:.72rem;font-weight:600}
.b-ld{background:#dcfce7;color:#166534}
.b-et{background:#ccfbf1;color:#115e59}
.b-al{background:#fef3c7;color:#92400e}
.b-no{background:#f3f4f6;color:#6b7280}
.ft{text-align:center;padding:1.6rem;color:#94a3b8;font-size:.78rem;line-height:1.7}
.srch{margin-bottom:1rem}
.srch input{padding:.5rem .9rem;border:1px solid #e2e8f0;border-radius:8px;font-size:.85rem;width:260px;outline:none}
.srch input:focus{border-color:#94a3b8;box-shadow:0 0 0 2px rgba(148,163,184,.2)}
</style>
</head>
<body>
<div class="hdr"><div style="max-width:1700px;margin:0 auto">
  <h1>S&amp;P 500 Market Leader Screener</h1>
  <p>Swing Leader vs Benchmark — Full Relative Strength Stack (RS 1‑4)</p>
</div></div>

<div class="wrap">
  <div class="cards">
    <div class="card"><div class="lb">Total Screened</div><div class="vl">{{ total }}</div></div>
    <div class="card c-ld"><div class="lb">Leaders</div><div class="vl">{{ n_leader }}</div></div>
    <div class="card c-et"><div class="lb">Early Trend</div><div class="vl">{{ n_early }}</div></div>
    <div class="card c-al"><div class="lb">Alarm</div><div class="vl">{{ n_alarm }}</div></div>
    <div class="card"><div class="lb">Benchmark</div><div class="vl" style="font-size:1.15rem">{{ bench }}</div></div>
    <div class="card"><div class="lb">Last Updated</div><div class="vl" style="font-size:.95rem">{{ updated }}</div></div>
    <div class="card" style="cursor:pointer" onclick="location.href='/toggle_beta'">
      <div class="lb">Beta Adjustment</div>
      <div class="vl" style="font-size:1.1rem;color:{% if beta_adj %}#059669{% else %}#6b7280{% endif %}">
        {% if beta_adj %}ON{% else %}OFF{% endif %}
      </div>
      <div style="font-size:.7rem;color:#94a3b8;margin-top:.2rem">click to toggle</div>
    </div>
  </div>

  <div class="filters">
    <button class="on" onclick="filt('all',this)">All ({{ total }})</button>
    <button onclick="filt('LEADER',this)">Leaders ({{ n_leader }})</button>
    <button onclick="filt('Early Trend',this)">Early Trend ({{ n_early }})</button>
    <button onclick="filt('Alarm',this)">Alarm ({{ n_alarm }})</button>
    <button onclick="filt('Not Leading',this)">Not Leading ({{ n_none }})</button>
  </div>

  <div class="srch"><input id="q" type="text" placeholder="Search ticker or company…" oninput="search(this.value)"></div>

  <div class="tw">
  <table id="tbl">
    <thead><tr>
      <th onclick="srt(0,0)">#<span class="si">⇅</span></th>
      <th onclick="srt(1,0)">Ticker<span class="si">⇅</span></th>
      <th onclick="srt(2,1)">Score<span class="si">⇅</span></th>
      <th onclick="srt(3,0)">Company<span class="si">⇅</span></th>
      <th onclick="srt(4,0)">Sector<span class="si">⇅</span></th>
      <th onclick="srt(5,1)">RS 60d %<span class="si">⇅</span></th>
      <th onclick="srt(6,1)">RS 20d %<span class="si">⇅</span></th>
      <th onclick="srt(7,1)">RS 10d %<span class="si">⇅</span></th>
      <th onclick="srt(8,1)">RS 5d %<span class="si">⇅</span></th>
      <th onclick="srt(9,1)">IR 60d<span class="si">⇅</span></th>
      <th onclick="srt(10,1)">IR 20d<span class="si">⇅</span></th>
      <th onclick="srt(11,1)">Beta<span class="si">⇅</span></th>
      <th onclick="srt(12,1)">Alpha %<span class="si">⇅</span></th>
      <th onclick="srt(13,0)">Momentum<span class="si">⇅</span></th>
      <th onclick="srt(14,0)">Status<span class="si">⇅</span></th>
    </tr></thead>
    <tbody>
    {% for r in rows %}
    <tr data-st="{{ r.status }}" data-sk="{{ r.ticker }}|{{ r.company }}">
      <td>{{ loop.index }}</td>
      <td><strong>{{ r.ticker }}</strong></td>
      <td style="font-weight:600">{% if r.composite is not none %}{{ "%.2f"|format(r.composite) }}{% else %}--{% endif %}</td>
      <td>{{ r.company }}</td>
      <td>{{ r.sector }}</td>
      {% if r.rs_60 is not none %}<td class="{{ 'pos' if r.rs_60 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_60) }}</td>{% else %}<td>—</td>{% endif %}
      {% if r.rs_20 is not none %}<td class="{{ 'pos' if r.rs_20 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_20) }}</td>{% else %}<td>—</td>{% endif %}
      {% if r.rs_10 is not none %}<td class="{{ 'pos' if r.rs_10 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_10) }}</td>{% else %}<td>—</td>{% endif %}
      {% if r.rs_5 is not none %}<td class="{{ 'pos' if r.rs_5 > 0 else 'neg' }}">{{ "%.2f"|format(r.rs_5) }}</td>{% else %}<td>—</td>{% endif %}
      {% if r.ir_60 is not none %}<td class="{{ 'pos' if r.ir_60 > 0 else 'neg' }}">{{ "%.2f"|format(r.ir_60) }}</td>{% else %}<td>—</td>{% endif %}
      {% if r.ir_20 is not none %}<td class="{{ 'pos' if r.ir_20 > 0 else 'neg' }}">{{ "%.2f"|format(r.ir_20) }}</td>{% else %}<td>—</td>{% endif %}
      {% if r.beta is not none %}<td>{{ "%.2f"|format(r.beta) }}</td>{% else %}<td>—</td>{% endif %}
      {% if r.alpha_ann is not none %}<td class="{{ 'pos' if r.alpha_ann > 0 else 'neg' }}">{{ "%.2f"|format(r.alpha_ann) }}</td>{% else %}<td>—</td>{% endif %}
      <td>{% if r.early_trend %}<span style="color:#0d9488;font-weight:600">▲ Accel</span>{% else %}<span style="color:#94a3b8">—</span>{% endif %}</td>
      <td>{% if r.status=='LEADER' %}<span class="badge b-ld">LEADER</span>{% elif r.status=='Early Trend' %}<span class="badge b-et">Early Trend</span>{% elif r.status=='Alarm' %}<span class="badge b-al">Alarm</span>{% else %}<span class="badge b-no">—</span>{% endif %}</td>
    </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
</div>

<div class="ft">
  <p>Methodology: Multi-timeframe relative strength · Consistency (IR) · Momentum &amp; acceleration filters</p>
  <p>Benchmark: {{ bench }} · RS Mode: {{ 'Beta-neutral' if beta_adj else 'Raw excess' }} · Log returns: {{ 'Yes' if log_ret else 'No' }}</p>
</div>

<script>
let cs={col:-1,asc:true};
function srt(ci,num){
  const tb=document.querySelector('#tbl tbody');
  const rows=Array.from(tb.rows);
  const asc=cs.col===ci?!cs.asc:!num;
  cs={col:ci,asc:asc};
  rows.sort((a,b)=>{
    let va=a.cells[ci].textContent.trim(), vb=b.cells[ci].textContent.trim();
    if(num){va=parseFloat(va)||-1e9;vb=parseFloat(vb)||-1e9;return asc?va-vb:vb-va;}
    return asc?va.localeCompare(vb):vb.localeCompare(va);
  });
  rows.forEach(r=>tb.appendChild(r));
}
function filt(st,btn){
  document.querySelectorAll('.filters button').forEach(b=>b.classList.remove('on'));
  btn.classList.add('on');
  document.querySelectorAll('#tbl tbody tr').forEach(r=>{
    r.style.display=(st==='all'||r.dataset.st===st)?'':'none';
  });
}
function search(q){
  q=q.toLowerCase();
  document.querySelectorAll('#tbl tbody tr').forEach(r=>{
    r.style.display=r.dataset.sk.toLowerCase().includes(q)?'':'none';
  });
}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# Route
# ═══════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    df = G["results"]
    info = G["info"]

    def _safe(v):
        if v is None:
            return None
        if isinstance(v, (float, np.floating)):
            if np.isnan(v) or np.isinf(v):
                return None
            return round(float(v), 2)
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        return v

    rows = []
    for tk, row in df.iterrows():
        rows.append(
            {
                "ticker": tk,
                "company": info.get(tk, {}).get("name", ""),
                "sector": info.get(tk, {}).get("sector", ""),
                **{
                    k: _safe(row[k])
                    for k in [
                        "composite",
                        "rs_60", "rs_20", "rs_10", "rs_5",
                        "ir_60", "ir_20", "beta", "alpha_ann", "alpha_ir",
                        "early_trend", "status",
                    ]
                },
            }
        )

    cnt = df["status"].value_counts()
    return render_template_string(
        TEMPLATE,
        rows=rows,
        total=len(df),
        n_leader=int(cnt.get("LEADER", 0)),
        n_early=int(cnt.get("Early Trend", 0)),
        n_alarm=int(cnt.get("Alarm", 0)),
        n_none=int(cnt.get("Not Leading", 0)),
        bench=BENCH,
        updated=G["updated"],
        beta_adj=G["beta_adj"],
        log_ret=LOG_RET,
    )


@app.route("/toggle_beta")
def toggle_beta():
    from flask import redirect
    G["beta_adj"] = not G["beta_adj"]
    _log(f"[*] Beta adjustment toggled -> {G['beta_adj']}")
    G["results"] = screen(G["close"], beta_adj=G["beta_adj"])
    G["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return redirect("/")


# ═══════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="S&P 500 Leader Screener")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    tickers, info = get_sp500()
    close = sync_prices(tickers)
    results = screen(close, beta_adj=BETA_ADJ)

    G["close"] = close
    G["results"] = results
    G["info"] = info
    G["beta_adj"] = BETA_ADJ
    G["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    _log(f"\n* Dashboard ready -> http://127.0.0.1:{args.port}\n")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
