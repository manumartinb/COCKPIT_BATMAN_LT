#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
update_dashboard.py -- COCKPIT Batman LT
========================================
Genera data.json del dashboard COCKPIT y hace push a GitHub Pages.
https://manumartinb.github.io/COCKPIT_BATMAN_LT/

COCKPIT_SCORE = mean( PUT_pct, THETA_pct, SKTS_pct )   [APR 2026-07-07: MAXOR excluido]
  PUT   = skew_25d_vs50_pct_expanding @dte60/10:30/PUT de SKEW_PUT_ENRICHED (ya asof)
  THETA = pct expanding (por filas, min_periods=250) del day-mean theta_k2/SPX
          (historico Gen3 RAND + acumulador LIVE del hook V51; escala backtester)
  SKTS  = serie pct del data.json LOCAL del dashboard SKEW_TS (una fuente de verdad)
  score_pct = expanding rank del score. Zonas: ROJA<=20 / NEUTRA / VERDE>=80 / TURBO>=95.
MAXOR = max(PUT, TENSION pctE asof) solo como FLAG informativo (>=80), fuera del score.

Politica de frescura: score solo en dias con los 3 diales (sin imputacion, nunca
mean-de-2); latest report por-dial con su fecha; score INDETERMINADO si algun dial
supera 5 dias HABILES de retraso.
ASSERT de distribucion: la interseccion historica debe rondar los ~1.558 dias del APR.
Exit codes: 0 ok / 1 error. Auth push: remote SSH del repo propio (guard .git).
Docs: memory/analisis_predictabilidad_robustez_cockpit_20260707.md
"""
from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
BASE = Path(r"C:\Users\Administrator\Desktop\BULK OPTIONSTRAT\ESTRATEGIAS")
PUT_CSV = BASE / "Skew" / "SKEW_PUT_ENRICHED.csv"
TEN_CSV = BASE / "Skew" / "SURFACE_SKEW_CONCAVITY_COMPONENTS_DAILY.csv"
SKTS_DATA_JSON = BASE / "Skew" / "dashboards" / "SKEW_TS_CALL_BATMAN_LT_DASHBOARD" / "data.json"
THETA_HIST = BASE / "Skew" / "dashboards" / "_private_feeds" / "THETA_K2_SPX_DAILY_HIST.csv"
THETA_LIVE = BASE / "Skew" / "dashboards" / "_private_feeds" / "THETA_K2_SPX_DAILY_LIVE.csv"

DASHBOARD_DIR = BASE / "Skew" / "dashboards" / "COCKPIT_BATMAN_LT_DASHBOARD"
DATA_JSON = DASHBOARD_DIR / "data.json"

GH_USER_NAME = "manumartinb"
GH_USER_EMAIL = "manuelmartinbarranco@gmail.com"
BRANCH = "main"
TZ = ZoneInfo("Europe/Madrid")

MIN_PERIODS = 250
ROJA_MAX, VERDE_MIN, TURBO_MIN = 20.0, 80.0, 95.0
STALE_WARN = {"PUT": 2, "SKTS": 2, "THETA": 3}   # dias HABILES por dial
STALE_SCORE_MAX = 5                                # habiles: score INDETERMINADO
APR_INTERSECT_MIN, APR_INTERSECT_MAX = 1480, 4000  # assert distribucion (crece con LIVE)


def zone_label(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "INDETERMINADO"
    if v >= TURBO_MIN:
        return "TURBO"
    if v >= VERDE_MIN:
        return "VERDE"
    if v <= ROJA_MAX:
        return "ROJA"
    return "NEUTRA"


def _rn(v, prec=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None
    return round(float(v), prec)


def busdays_since(date_str: str) -> int:
    try:
        return int(np.busday_count(np.datetime64(date_str, "D"),
                                   np.datetime64(datetime.now(TZ).date(), "D")))
    except Exception:
        return 999


# ---------------- series por dial ----------------
def serie_put() -> pd.Series:
    df = pd.read_csv(PUT_CSV, usecols=["trade_date", "snapshot_time", "side", "dte_target",
                                       "skew_25d_vs50_pct_expanding"], low_memory=False)
    df = df[(df.side.astype(str).str.upper() == "PUT")
            & (df.snapshot_time.astype(str) == "10:30:00")
            & (pd.to_numeric(df.dte_target, errors="coerce") == 60)]
    df["d"] = pd.to_datetime(df.trade_date, errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["d", "skew_25d_vs50_pct_expanding"])
    return df.sort_values("d").drop_duplicates("d", keep="last").set_index("d")[
        "skew_25d_vs50_pct_expanding"].astype(float)


def serie_skts() -> pd.Series:
    d = json.loads(SKTS_DATA_JSON.read_text(encoding="utf-8"))
    s = pd.Series(d["pct"], index=d["dates"], dtype=float)
    return s.dropna()


def serie_theta_pct() -> pd.Series:
    hist = pd.read_csv(THETA_HIST)
    datecol = next((c for c in ("d", "dia", "date") if c in hist.columns), hist.columns[0])
    hist = hist.set_index(datecol)
    hist.index.name = "date"
    parts = [hist[["theta_day_mean_ref"]]]
    if THETA_LIVE.exists():
        live = pd.read_csv(THETA_LIVE)
        if len(live):
            live = live.set_index("date")[["theta_day_mean_ref"]]
            parts.append(live)
    raw = pd.concat(parts)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()["theta_day_mean_ref"].astype(float)
    pct = raw.expanding(min_periods=MIN_PERIODS).rank(pct=True) * 100.0
    return pct.dropna()


def serie_maxor(put: pd.Series) -> pd.Series:
    te = pd.read_csv(TEN_CSV, usecols=["trade_date", "TENSION_3WAY_MIN"])
    te["d"] = pd.to_datetime(te.trade_date, errors="coerce").dt.strftime("%Y-%m-%d")
    te = te.dropna().sort_values("d").set_index("d")["TENSION_3WAY_MIN"].astype(float)
    ten_pct = te.expanding(min_periods=50).rank(pct=True) * 100.0
    return pd.concat([put.rename("p"), ten_pct.rename("t")], axis=1).max(axis=1).dropna()


# ---------------- payload ----------------
def build_data_payload() -> dict:
    put = serie_put()
    skts = serie_skts()
    theta = serie_theta_pct()
    maxor = serie_maxor(put)

    idx = put.index.intersection(skts.index).intersection(theta.index).sort_values()
    n_int = len(idx)
    if not (APR_INTERSECT_MIN <= n_int <= APR_INTERSECT_MAX):
        raise RuntimeError(f"interseccion {n_int} dias fuera del rango certificado APR "
                           f"[{APR_INTERSECT_MIN}, {APR_INTERSECT_MAX}]: distribucion distinta")

    score = pd.DataFrame({"PUT": put[idx], "THETA": theta[idx], "SKTS": skts[idx]}).mean(axis=1)
    score_pct = score.expanding(min_periods=MIN_PERIODS).rank(pct=True) * 100.0
    body = pd.DataFrame({"score": score, "score_pct": score_pct}).dropna(subset=["score_pct"])

    # latest por dial (cada uno con SU fecha)
    dials = {}
    for name, s in (("PUT", put), ("THETA", theta), ("SKTS", skts)):
        last_d = str(s.index[-1])
        lag = busdays_since(last_d)
        dials[name] = {"date": last_d, "pct": _rn(s.iloc[-1]),
                       "stale_busdays": lag,
                       "warn": bool(lag > STALE_WARN[name])}
    mx_last = float(maxor.iloc[-1])
    flag = {"date": str(maxor.index[-1]), "value": _rn(mx_last),
            "on": bool(mx_last >= 80.0)}

    any_dead = any(d["stale_busdays"] > STALE_SCORE_MAX for d in dials.values())
    if any_dead or body.empty:
        latest_score = {"date": str(body.index[-1]) if len(body) else None,
                        "score": _rn(body.score.iloc[-1]) if len(body) else None,
                        "score_pct": _rn(body.score_pct.iloc[-1]) if len(body) else None,
                        "zone": "INDETERMINADO",
                        "reason": "dial(es) con mas de %d dias habiles de retraso" % STALE_SCORE_MAX}
    else:
        latest_score = {"date": str(body.index[-1]),
                        "score": _rn(body.score.iloc[-1]),
                        "score_pct": _rn(body.score_pct.iloc[-1]),
                        "zone": zone_label(float(body.score_pct.iloc[-1])),
                        "reason": ""}

    # series alineadas al indice del score (con maxor mapeado; NaN -> null)
    def col(s):
        v = s.reindex(body.index)
        return [_rn(x) for x in v]

    return {
        "generated_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "formula": "COCKPIT_SCORE = mean(PUT_pct, THETA_pct, SKTS_pct); MAXOR solo flag (APR 2026-07-07)",
        "n_days": int(len(body)),
        "thresholds": {"roja_max": ROJA_MAX, "verde_min": VERDE_MIN, "turbo_min": TURBO_MIN},
        "latest": {**latest_score, "dials": dials, "maxor_flag": flag,
                   "zone_display": latest_score["zone"]},
        "dates": body.index.tolist(),
        "score": [_rn(x) for x in body.score],
        "score_pct": [_rn(x) for x in body.score_pct],
        "put": col(put), "theta_pct": col(theta), "skts_pct": col(skts),
        "maxor": col(maxor),
    }


def _payload_data_changed(new_payload: dict) -> bool:
    if not DATA_JSON.exists():
        return True
    try:
        old = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    except Exception:
        return True
    for k in ("dates", "score_pct", "latest", "n_days"):
        if old.get(k) != new_payload.get(k):
            return True
    return False


def write_data_json(payload: dict) -> None:
    DATA_JSON.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
                         encoding="utf-8")


def _git(args: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(DASHBOARD_DIR), *args],
                          capture_output=True, text=True, check=False)


def push_to_github() -> int:
    if not (DASHBOARD_DIR / ".git").exists():
        print(f"[X] {DASHBOARD_DIR} no es un repo git propio (falta .git); push abortado")
        return 1
    _git(["config", "user.name", GH_USER_NAME])
    _git(["config", "user.email", GH_USER_EMAIL])
    _git(["add", "-A"])
    status = _git(["status", "--porcelain"])
    if not status.stdout.strip():
        print("[INFO] no changes to commit, nothing to push")
        return 0
    today = datetime.now(TZ).strftime("%Y-%m-%d %H:%M")
    commit = _git(["commit", "-m", f"daily update {today}"])
    if commit.returncode != 0:
        print(f"[X] commit failed: {commit.stderr.strip()}")
        return 1
    push = _git(["push", "origin", BRANCH])
    if push.returncode != 0:
        print(f"[X] push failed: {push.stderr.strip()}")
        return 1
    print("[OK] pushed to https://manumartinb.github.io/COCKPIT_BATMAN_LT/")
    return 0


def main() -> int:
    try:
        if not DASHBOARD_DIR.exists():
            print(f"[X] dashboard dir not found: {DASHBOARD_DIR}")
            return 1
        payload = build_data_payload()
        changed = _payload_data_changed(payload)
        write_data_json(payload)
        L = payload["latest"]
        print(f"[INFO] data.json {'updated' if changed else 'identical'} | "
              f"score={L['score']} pct={L['score_pct']} zone={L['zone']} @ {L['date']} | "
              f"dials: " + " ".join(f"{k}={v['pct']}@{v['date']}(+{v['stale_busdays']}bd)"
                                    for k, v in L["dials"].items())
              + f" | MAXOR flag={'ON' if L['maxor_flag']['on'] else 'OFF'}({L['maxor_flag']['value']}) | n_days={payload['n_days']}")
        if not changed:
            return 0
        return push_to_github()
    except Exception as exc:
        print(f"[X] update_dashboard failed: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
