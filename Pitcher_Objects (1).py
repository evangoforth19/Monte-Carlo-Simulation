# build_pitcher_packs.py
# Purpose: build full Pitcher objects (linked to hitters) and export sim packs
#          that contain pitcher_data_arch + models needed by Monte Carlo.

import os
import time
import types
from typing import Optional

import pandas as pd
import joblib

from baseball_utils import *  # if unused, you can remove this import
from Pitcher_Class import Pitcher
from Hitter_Class import Hitter                         # only needed if you build a fresh link-hitter
from General_Initialization import classify_archetype   # to annotate `batting`


# ---------- Core data loaders ----------
def load_core_frames():
    pitcher_data   = pd.read_pickle("pitcher_data.pkl")
    batter_data    = pd.read_pickle("batter_data.pkl")
    batting        = pd.read_pickle("batting_2025.pkl")
    encoder_data   = pd.read_pickle("encoder_data.pkl")
    bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")

    # ensure archetype column exists (your pipeline expects it)
    if "Hitter_Archetype" not in batting.columns:
        batting["Hitter_Archetype"] = batting.apply(classify_archetype, axis=1)

    return pitcher_data, batter_data, batting, encoder_data, bat_stats_2025


def make_team_woba_df():
    return pd.DataFrame({
        "Team": [
            "CHC","NYY","TOR","LAD","ARI","BOS","DET","NYM","MIL","SEA",
            "PHI","HOU","STL","ATH","ATL","SDP","TBR","BAL","MIN","MIA",
            "TEX","CIN","SFG","CLE","LAA","WSN","KCR","PIT","CHW","COL"
        ],
        "wOBA": [
            0.333,0.337,0.328,0.334,0.329,0.328,0.322,0.317,0.313,0.319,
            0.323,0.318,0.312,0.323,0.311,0.307,0.316,0.314,0.312,0.309,
            0.298,0.313,0.302,0.296,0.311,0.305,0.298,0.285,0.293,0.296
        ],
    })


# Path to your packs folder
PACKS_DIR = "packs"

# Map hitter names to their saved joblib filenames
HITTER_PACKS = {
    "judge": "hitter_judge.joblib",
    "mcneil": "hitter_mcneil.joblib",
    "olson": "hitter_olson.joblib",
    "semien": "hitter_semien.joblib",
    "alonso": "hitter_alonso.joblib",
    "castellanos": "hitter_castellanos.joblib",
    "ozuna": "hitter_ozuna.joblib",
    "witt_jr": "hitter_witt_jr.joblib",
    "soto": "hitter_soto.joblib",
    "freeman": "hitter_freeman.joblib",
    "riley": "hitter_riley.joblib",
    "adames": "hitter_adames.joblib",
}


def _wrap_pack_as_namespace(obj):
    """
    Hitter packs are saved as dicts; Pitcher_Class expects attribute access
    (e.g. hitter.cluster_encoder). Wrap dicts in SimpleNamespace.
    """
    if isinstance(obj, dict):
        return types.SimpleNamespace(**obj)
    return obj


# Load all hitter objects into a dict (wrapped as attribute namespaces)
loaded_hitters = {}
for name, filename in HITTER_PACKS.items():
    path = os.path.join(PACKS_DIR, filename)
    if not os.path.exists(path):
        print(f"[WARN] Hitter pack not found: {path}")
        continue
    obj = joblib.load(path)
    obj = _wrap_pack_as_namespace(obj)
    loaded_hitters[name] = obj

    enc = getattr(obj, "cluster_encoder", None)
    n_classes = len(getattr(enc, "classes_", [])) if enc is not None else 0
    print(f"[OK] Loaded hitter: {name} from {filename} | "
          f"enc={type(enc)} | n_classes={n_classes}")


# ---------- Exporter ----------
def export_pitcher_pack(pitcher: Pitcher, path: str, include_full_df: bool = True):
    """
    Save everything Monte Carlo needs so it doesn't import GI:
      - pitcher_data_arch with count_enc
      - fast handedness map
      - starter IP/BF models + ip_std
      - team and winning_pct_value
      - bf_per_out (bullpen distribution)
      - extra-innings distributions + probability
      - identity fields for AtBatSim logging
    """
    df = getattr(pitcher, "pitcher_data_arch", None)

    # Ensure count_enc exists (avoid GI.encode_count later)
    if df is not None and not df.empty and "count_enc" not in df.columns:
        # local encode_count to avoid GI import
        def _enc(b, s):
            table = {
                (0,0):0,(0,1):1,(0,2):2,(1,0):3,(1,1):4,(1,2):5,
                (2,0):6,(2,1):7,(2,2):8,(3,0):9,(3,1):10,(3,2):11
            }
            return table.get((int(b), int(s)), 0)
        df = df.copy()
        df["count_enc"] = df[["balls","strikes"]].apply(lambda r: _enc(r["balls"], r["strikes"]), axis=1)

    # Fast-handedness lookup
    stand_map = {}
    if df is not None and not df.empty:
        stand_map = (
            df[["batter_name", "stand"]]
            .dropna()
            .assign(batter_name_lower=lambda d: d["batter_name"].str.lower())
            .drop_duplicates("batter_name_lower")
            .set_index("batter_name_lower")["stand"]
            .to_dict()
        )

    # Bullpen BF per out
    try:
        bf_per_out = list(pitcher.bf_list())
    except Exception:
        bf_per_out = []

    # Pull extras distributions from GI *now* (build time), not at sim time
    try:
        from General_Initialization import home_IP_extras, away_IP_extras, prob_extra_innings
        home_extras = list(home_IP_extras)
        away_extras = list(away_IP_extras)
        p_extras    = float(prob_extra_innings)
    except Exception:
        home_extras, away_extras, p_extras = [], [], 0.09  # safe defaults

    pack = {
        "stand_by_batter_lower": stand_map,
        "IPLinReg": getattr(pitcher, "IPLinReg", None),
        "poisson_model": getattr(pitcher, "poisson_model", None),
        "ip_std": float(getattr(pitcher, "ip_std", 0.0) or 0.0),
        "team": getattr(pitcher, "team", None),
        "winning_pct_value": float(getattr(pitcher, "winning_pct_value", 0.5) or 0.5),

        # bullpen / extras
        "bf_per_out": bf_per_out,
        "home_IP_extras": home_extras,
        "away_IP_extras": away_extras,
        "prob_extra_innings": p_extras,

        # identity (for logs)
        "first_lower": getattr(pitcher, "first_lower", "max"),
        "last_lower":  getattr(pitcher, "last_lower",  "fried"),
        "full_lower":  getattr(pitcher, "full_lower",  "max fried"),
        "first_upper": getattr(pitcher, "first_upper", "Max"),
        "last_upper":  getattr(pitcher, "last_upper",  "Fried"),
        "full_upper":  getattr(pitcher, "full_upper",  "Max Fried"),
    }

    if include_full_df and df is not None:
        pack["pitcher_data_arch"] = df

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pack, path, compress=3)
    print(f"[OK] Saved pitcher pack → {path}  "
          f"(with{'out' if not include_full_df else ''} pitcher_data_arch)")


# ---------- Small helper: build linked Hitter (for cluster space) ----------
def create_link_hitter_for_pitcher(
    first: str, last: str, team_name: str, team_id: int,
    batter_data: pd.DataFrame, encoder_data: pd.DataFrame,
    bat_stats_2025: pd.DataFrame, batting: pd.DataFrame
) -> Hitter:
    """
    Pitcher_Class uses the hitter's GMMs/cluster encoder to assign local clusters.
    Build a full Hitter here (once) purely to link during Pitcher construction.
    """
    return Hitter(
        first_lower=first.lower(), first_upper=first.title(),
        last_lower=last.lower(),   last_upper=last.title(),
        team_name=team_name, team_id=team_id,
        batter_data=batter_data, encoder_data=encoder_data,
        bat_stats_2025=bat_stats_2025, batting=batting,
        player_id=None,  # OK: only need archetype/encoders/cluster space
    )


# ---------- Factory: Pitcher for storage ----------
def create_pitcher_for_storage(
    first: str,
    last: str,
    team_nickname: str,
    pitcher_data: pd.DataFrame,
    batting: pd.DataFrame,
    team_woba: pd.DataFrame,
    linked_hitter,              # Hitter instance OR wrapped pack (SimpleNamespace)
    player_id: Optional[int] = None,
) -> Pitcher:
    """
    Creates a full Pitcher object (runs its pipeline) with the provided linked hitter.
    If player_id is provided, the Pitcher class will skip lookup; otherwise it will search.
    """
    return Pitcher(
        first_lower=first.lower(),
        first_upper=first.title(),
        last_lower=last.lower(),
        last_upper=last.title(),
        team=team_nickname,
        pitcher_data=pitcher_data,
        batting=batting,
        team_woba=team_woba,
        hitter=linked_hitter,          # crucial for cluster encoder space
        player_id=player_id,
    )


# ---------- Main build/export ----------
if __name__ == "__main__":
    pitcher_data, batter_data, batting, encoder_data, bat_stats_2025 = load_core_frames()
    team_woba = make_team_woba_df()

    def require(name: str):
        """Fetch a hitter by key from loaded_hitters or raise with a clear message."""
        obj = loaded_hitters.get(name)
        if obj is None:
            raise RuntimeError(f"Missing hitter pack '{name}'. Cannot build linked pitcher.")
        return obj

    # Soriano ← Freeman
    soriano = create_pitcher_for_storage(
        first="Jose", last="Soriano", team_nickname="Angels",
        pitcher_data=pitcher_data, batting=batting, team_woba=team_woba,
        linked_hitter=require("freeman"),
        player_id=667755  # MLBAM id
    )
    export_pitcher_pack(soriano, os.path.join(PACKS_DIR, "pitcher_soriano.joblib"))
"""
    # Abbott ← Castellanos
    abbott = create_pitcher_for_storage(
        first="Andrew", last="Abbott", team_nickname="Reds",
        pitcher_data=pitcher_data, batting=batting, team_woba=team_woba,
        linked_hitter=require("castellanos")
    )
    export_pitcher_pack(abbott, os.path.join(PACKS_DIR, "pitcher_abbott.joblib"))

    # Heaney ← Adames
    heaney = create_pitcher_for_storage(
        first="Andrew", last="Heaney", team_nickname="Rangers",
        pitcher_data=pitcher_data, batting=batting, team_woba=team_woba,
        linked_hitter=require("adames")
    )
    export_pitcher_pack(heaney, os.path.join(PACKS_DIR, "pitcher_heaney.joblib"))

    # Cavalli ← Witt Jr (only if the Witt Jr pack exists)
    if "witt_jr" in loaded_hitters:
        cavalli = create_pitcher_for_storage(
            first="Cade", last="Cavalli", team_nickname="Nationals",
            pitcher_data=pitcher_data, batting=batting, team_woba=team_woba,
            linked_hitter=require("witt_jr"),
            player_id=680777  # MLBAM id
        )
        export_pitcher_pack(cavalli, os.path.join(PACKS_DIR, "pitcher_cavalli.joblib"))
    else:
        print("[SKIP] Witt Jr pack missing; skipped Cavalli.")"""
