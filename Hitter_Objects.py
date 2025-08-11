import os
import pandas as pd
import joblib
from pybaseball import playerid_lookup

# Import only what you truly need from your own modules
from Hitter_Class import Hitter
from General_Initialization import classify_archetype  # used to annotate `batting`

# ---------- Load core data once ----------
def load_core_frames():
    batter_data   = pd.read_pickle("batter_data.pkl")
    pitcher_data  = pd.read_pickle("pitcher_data.pkl")   # not used by Hitter, but harmless if you need later
    batting       = pd.read_pickle("batting_2025.pkl")
    encoder_data  = pd.read_pickle("encoder_data.pkl")
    bat_stats_2025= pd.read_pickle("bat_stats_2025.pkl")

    # Precompute hitter archetypes on batting (many parts of your pipeline expect this column to exist)
    if "Hitter_Archetype" not in batting.columns:
        batting["Hitter_Archetype"] = batting.apply(classify_archetype, axis=1)

    return batter_data, pitcher_data, batting, encoder_data, bat_stats_2025


# ---------- Pack exporter ----------
def export_hitter_pack(hitter, path, ensure_metadata=False):
    """
    Save only the attributes the simulation uses.
    If ensure_metadata=True, try to fetch lineup spot (may do network calls).
    """
    # Optional: try to populate lineup spot/win% if you want (can hit network)
    if ensure_metadata:
        try:
            hitter._get_player_metadata()
        except Exception:
            pass

    pack = {
        # core sim artifacts
        "cluster_encoder": hitter.cluster_encoder,
        "stand_encoder": hitter.stand_encoder,
        "arch_encoder": getattr(hitter, "arch_encoder", None),
        "outcome_encoder": hitter.outcome_encoder,
        "nb_outcome_model": hitter.nb_outcome_model,
        "outcome_lookup_table": hitter.outcome_lookup_table,
        "outcome_class_labels": hitter.outcome_class_labels,
        "xba_lookup_table": hitter.xba_lookup_table,
        "global_bip_xba": getattr(hitter, "global_bip_xba", 0.300),
        "arch_enc": int(hitter.arch_enc),

        # identifiers / context
        "full_lower": getattr(hitter, "full_lower", None),
        "full_upper": getattr(hitter, "full_upper", None),
        "team_name": getattr(hitter, "team_name", "Mets"),  # default to Mets for McNeil

        # simple fallbacks so MC can run without network
        "xba": float(getattr(hitter, "xba", getattr(hitter, "global_bip_xba", 0.300)) or 0.300),
        "most_recent_spot": int(getattr(hitter, "most_recent_spot", 3) or 3),
        "winning_pct_value": float(getattr(hitter, "winning_pct_value", 0.5) or 0.5),
    }

    import os, joblib
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pack, path, compress=3)
    print(f"[OK] Saved hitter pack → {path}")


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
) -> Hitter:
    # pybaseball expects last, first (lowercase is fine)
    pid = playerid_lookup(last.lower(), first.lower())["key_mlbam"].values[0]

    # Build the full Hitter; your Hitter.__init__ already builds all pipelines/encoders/lookups
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


# ---------- Main (build and export) ----------
if __name__ == "__main__":
    batter_data, pitcher_data, batting, encoder_data, bat_stats_2025 = load_core_frames()

    # Example: Aaron Judge
    judge = create_hitter_for_storage(
        first="Aaron", last="Judge", team_name="Yankees", team_id=147,
        batter_data=batter_data, encoder_data=encoder_data,
        bat_stats_2025=bat_stats_2025, batting=batting,
    )
    export_hitter_pack(judge, "packs/hitter_judge.joblib")

    # Example: Jeff McNeil
    mcneil = create_hitter_for_storage(
        first="Jeff", last="McNeil", team_name="Mets", team_id=121,
        batter_data=batter_data, encoder_data=encoder_data,
        bat_stats_2025=bat_stats_2025, batting=batting,
    )
    export_hitter_pack(mcneil, "packs/hitter_mcneil.joblib")

    print("[DONE] All hitter packs successfully built and saved to /packs.")