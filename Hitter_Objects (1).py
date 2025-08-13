# Hitter_Objects.py
import os, traceback
import pandas as pd
import joblib
from pybaseball import playerid_lookup
from baseball_utils import *

from Hitter_Class import Hitter
from General_Initialization import classify_archetype  # used to annotate `batting`


# ---------- Load core data once ----------
def load_core_frames():
    batter_data    = pd.read_pickle("batter_data.pkl")
    pitcher_data   = pd.read_pickle("pitcher_data.pkl")  # not used here, but harmless
    batting        = pd.read_pickle("batting_2025.pkl")
    encoder_data   = pd.read_pickle("encoder_data.pkl")
    bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")

    # Precompute hitter archetypes on batting
    if "Hitter_Archetype" not in batting.columns:
        batting["Hitter_Archetype"] = batting.apply(classify_archetype, axis=1)

    return batter_data, pitcher_data, batting, encoder_data, bat_stats_2025


# ---------- Pack exporter ----------
def export_hitter_pack(hitter, path, ensure_metadata: bool = False):
    """
    Save the artifacts the sim (and Pitcher_Class) need — INCLUDING gmm_models.
    If ensure_metadata=True, try to fetch lineup spot (may do network calls).
    """
    # Optional metadata refresh
    if ensure_metadata:
        try:
            hitter._get_player_metadata()
        except Exception:
            pass

    # --- hard checks so we fail fast if something is off ---
    enc = getattr(hitter, "cluster_encoder", None)
    if enc is None or not hasattr(enc, "classes_") or len(getattr(enc, "classes_", [])) == 0:
        raise RuntimeError("cluster_encoder missing or unfitted on hitter.")

    gmm_models = getattr(hitter, "gmm_models", None)
    if not isinstance(gmm_models, dict) or len(gmm_models) == 0:
        raise RuntimeError("gmm_models missing on hitter (required by Pitcher_Class).")

    # Build pack
    pack = {
        # --- cluster space / models (required by Pitcher_Class) ---
        "gmm_models": hitter.gmm_models,
        "importances": hitter.importances,
        "cluster_encoder": hitter.cluster_encoder,

        # --- encoders & outcome/xBA artifacts ---
        "stand_encoder": hitter.stand_encoder,
        "arch_encoder": getattr(hitter, "arch_encoder", None),
        "outcome_encoder": hitter.outcome_encoder,
        "nb_outcome_model": hitter.nb_outcome_model,
        "outcome_lookup_table": hitter.outcome_lookup_table,
        "outcome_class_labels": hitter.outcome_class_labels,
        "xba_lookup_table": hitter.xba_lookup_table,
        "global_bip_xba": float(getattr(hitter, "global_bip_xba", 0.300) or 0.300),

        # --- identifiers / context ---
        "arch_enc": int(hitter.arch_enc),
        "full_lower": getattr(hitter, "full_lower", None),
        "full_upper": getattr(hitter, "full_upper", None),
        "team_name": getattr(hitter, "team_name", None),

        # --- simple fallbacks so MC can run without network (not modeling fallbacks) ---
        "xba": float(getattr(hitter, "xba", getattr(hitter, "global_bip_xba", 0.300)) or 0.300),
        "most_recent_spot": int(getattr(hitter, "most_recent_spot", 3) or 3),
        "winning_pct_value": float(getattr(hitter, "winning_pct_value", 0.5) or 0.5),
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pack, path, compress=3)

    # sanity log
    n_classes = len(getattr(enc, "classes_", []))
    print(f"[OK] Saved hitter pack → {path} | enc_classes={n_classes} | gmm_keys={len(gmm_models)}")


# ---------- Hitter factory (builds the full object once) ----------
def create_hitter_for_storage(
    first: str,
    last: str,
    team_name: str,
    team_id: int,
    batter_data: pd.DataFrame,
    encoder_data: pd.DataFrame,
    bat_stats_2025: pd.DataFrame,
    batting: pd.DataFrame,
    player_id: int | None = None,
) -> Hitter:
    """
    Creates a Hitter object with optional explicit player_id.
    If player_id is provided, bypasses pybaseball lookup.
    """
    if player_id is None:
        pid = playerid_lookup(last.lower(), first.lower())["key_mlbam"].values[0]
    else:
        pid = int(player_id)

    hitter = Hitter(
        first_lower=first.lower(),
        first_upper=first.title(),
        last_lower=last.lower(),
        last_upper=last.title(),
        team_name=team_name,
        team_id=team_id,
        batter_data=batter_data,
        encoder_data=encoder_data,
        bat_stats_2025=bat_stats_2025,
        batting=batting,
        player_id=pid,
    )
    return hitter


# ---------- Safe runner (skip on error) ----------
def safe_build_and_export(
    player_spec: dict,
    batter_data: pd.DataFrame,
    encoder_data: pd.DataFrame,
    bat_stats_2025: pd.DataFrame,
    batting: pd.DataFrame
) -> bool:
    """
    player_spec requires keys:
      first, last, team_name, team_id, player_id, pack
    """
    label = f"{player_spec.get('first','?')} {player_spec.get('last','?')}"
    try:
        hitter = create_hitter_for_storage(
            first=player_spec["first"],
            last=player_spec["last"],
            team_name=player_spec["team_name"],
            team_id=player_spec["team_id"],
            batter_data=batter_data,
            encoder_data=encoder_data,
            bat_stats_2025=bat_stats_2025,
            batting=batting,
            player_id=player_spec.get("player_id"),
        )
        export_hitter_pack(hitter, os.path.join("packs", player_spec["pack"]))
        return True
    except Exception as e:
        print(f"[SKIP] {label}: {e.__class__.__name__}: {e}")
        if os.environ.get("VERBOSE_ERRORS", ""):
            traceback.print_exc()
        return False


# ---------- Main (build and export) ----------
if __name__ == "__main__":
    batter_data, pitcher_data, batting, encoder_data, bat_stats_2025 = load_core_frames()

    players = [
        # keep the list small while testing
        {"first": "Freddie", "last": "Freeman", "team_name": "Dodgers", "team_id": 119, "player_id": 518692, "pack": "hitter_freeman.joblib"},
        {"first": "Jeff",    "last": "McNeil",  "team_name": "Mets",    "team_id": 121, "player_id": 643446, "pack": "hitter_mcneil.joblib"},
    ]

    successes, failures = 0, 0
    for p in players:
        ok = safe_build_and_export(
            player_spec=p,
            batter_data=batter_data,
            encoder_data=encoder_data,
            bat_stats_2025=bat_stats_2025,
            batting=batting,
        )
        successes += int(ok)
        failures  += int(not ok)

    print(f"\n[SUMMARY] Success: {successes}  |  Failed/Skipped: {failures}")

"""
        {"first":"Aaron","last":"Judge","team_name":"Yankees","team_id":147,"player_id":592450,"pack":"hitter_judge.joblib"},
        {"first":"Jeff","last":"McNeil","team_name":"Mets","team_id":121,"player_id":643446,"pack":"hitter_mcneil.joblib"},
        {"first":"Matt","last":"Olson","team_name":"Braves","team_id":144,"player_id":621566,"pack":"hitter_olson.joblib"},
        {"first":"Marcus","last":"Semien","team_name":"Rangers","team_id":140,"player_id":543760,"pack":"hitter_semien.joblib"},
        {"first":"Pete","last":"Alonso","team_name":"Mets","team_id":121,"player_id":624413,"pack":"hitter_alonso.joblib"},
        {"first":"Nick","last":"Castellanos","team_name":"Phillies","team_id":143,"player_id":592206,"pack":"hitter_castellanos.joblib"},
        {"first":"Marcell","last":"Ozuna","team_name":"Braves","team_id":144,"player_id":542303,"pack":"hitter_ozuna.joblib"},
        {"first":"Juan","last":"Soto","team_name":"Mets","team_id":121,"player_id":665742,"pack":"hitter_soto.joblib"},
        {"first":"Austin","last":"Riley","team_name":"Braves","team_id":144,"player_id":663586,"pack":"hitter_riley.joblib"},
        {"first":"Willy","last":"Adames","team_name":"Brewers","team_id":158,"player_id":642715,"pack":"hitter_adames.joblib"},
        {"first":"Mookie","last":"Betts","team_name":"Dodgers","team_id":119,"player_id":605141,"pack":"hitter_betts.joblib"},
        {"first":"Rafael","last":"Devers","team_name":"Red Sox","team_id":111,"player_id":646240,"pack":"hitter_devers.joblib"},
        {"first":"Bo","last":"Bichette","team_name":"Blue Jays","team_id":141,"player_id":666182,"pack":"hitter_bichette.joblib"},
        {"first":"Yordan","last":"Alvarez","team_name":"Astros","team_id":117,"player_id":670541,"pack":"hitter_alvarez.joblib"},
        {"first":"Francisco","last":"Lindor","team_name":"Mets","team_id":121,"player_id":596019,"pack":"hitter_lindor.joblib"},
        {"first":"Kyle","last":"Tucker","team_name":"Astros","team_id":117,"player_id":663656,"pack":"hitter_tucker.joblib"},
        {"first":"Trea","last":"Turner","team_name":"Phillies","team_id":143,"player_id":607208,"pack":"hitter_turner.joblib"},

        # Everyday types
        {"first":"Alec","last":"Bohm","team_name":"Phillies","team_id":143,"player_id":664761,"pack":"hitter_bohm.joblib"},
        {"first":"Nico","last":"Hoerner","team_name":"Cubs","team_id":112,"player_id":663538,"pack":"hitter_hoerner.joblib"},
        {"first":"Jake","last":"Cronenworth","team_name":"Padres","team_id":135,"player_id":630105,"pack":"hitter_cronenworth.joblib"},
        {"first":"Spencer","last":"Steer","team_name":"Reds","team_id":113,"player_id":668715,"pack":"hitter_steer.joblib"},
        {"first":"Ryan","last":"McMahon","team_name":"Yankees","team_id":147,"player_id":641857,"pack":"hitter_mcmahon.joblib"},
        {"first":"Lourdes","last":"Gurriel Jr.","team_name":"Diamondbacks","team_id":109,"player_id":666971,"pack":"hitter_gurriel.joblib"},
        {"first":"Christian","last":"Walker","team_name":"Astros","team_id":117,"player_id":572233,"pack":"hitter_walker.joblib"},
        {"first":"J.P.","last":"Crawford","team_name":"Mariners","team_id":136,"player_id":641487,"pack":"hitter_crawford.joblib"},
        {"first":"Jeremy","last":"Peña","team_name":"Astros","team_id":117,"player_id":665161,"pack":"hitter_pena.joblib"},
        {"first":"Brandon","last":"Nimmo","team_name":"Mets","team_id":121,"player_id":607043,"pack":"hitter_nimmo.joblib"},
"""
