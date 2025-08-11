# AtBatSim.py — lean, stateless, reload-safe (no project-wide imports)

import numpy as np

# ---------- tiny helpers (no GI import) ----------

def _encode_count(balls, strikes):
    table = {
        (0,0):0,(0,1):1,(0,2):2,(1,0):3,(1,1):4,(1,2):5,
        (2,0):6,(2,1):7,(2,2):8,(3,0):9,(3,1):10,(3,2):11
    }
    return table.get((int(balls), int(strikes)), 0)

def _argmax_counter(counter_dict):
    # counter_dict: {class_label: count}
    return max(counter_dict.items(), key=lambda kv: (kv[1], kv[0]))[0]

def hybrid_predict_row(key, lookup_table, nb_model, proba_classes):
    """
    If key is in lookup_table use argmax counts; else NB sample by probs.
    `proba_classes` must align with nb_model.classes_.
    """
    if key in lookup_table and len(lookup_table[key]) > 0:
        return _argmax_counter(lookup_table[key])
    probs = nb_model.predict_proba([list(key)])[0]  # (n_classes,)
    return np.random.choice(proba_classes, p=probs)

def hybrid_pitch_predict(X, nb_model, lookup_table, class_labels):
    """
    X: list of [stand_enc, count_enc, arch_enc]
    Returns: list[int] of predicted pitch_cluster_enc (global ids)
    """
    preds = []
    for row in X:
        key = tuple(int(v) for v in row)
        pred = hybrid_predict_row(key, lookup_table, nb_model, class_labels)
        preds.append(int(pred))
    return preds  # just the list

def predict_zone_hybrid(X, zone_nb_model, zone_lookup_table, zone_class_labels):
    """
    X: list of [pitch_cluster_enc, count_enc, stand_enc]
    Returns: list[int] of predicted zone_enc
    """
    preds = []
    for row in X:
        key = tuple(int(v) for v in row)
        pred = hybrid_predict_row(key, zone_lookup_table, zone_nb_model, zone_class_labels)
        preds.append(int(pred))
    return preds

def predict_outcome(*, pitch_cluster_enc, zone_enc, count_enc, nb_model, lookup_table, class_labels):
    """
    Hierarchical lookup backoff, then NB sample.
    Keys may include: (c,z,k), (c,z), (c,k), (c,), (z,k), (z,)
    """
    c, z, k = int(pitch_cluster_enc), int(zone_enc), int(count_enc)
    for key in ((c, z, k), (c, z), (c, k), (c,), (z, k), (z,)):
        if key in lookup_table and len(lookup_table[key]) > 0:
            return _argmax_counter(lookup_table[key])
    probs = nb_model.predict_proba([[c, z, k]])[0]
    return int(np.random.choice(class_labels, p=probs))

def predict_xba(pitch_cluster_enc, zone_enc, count_enc, xba_lookup_table, global_fallback=0.300):
    """
    xBA lookup with hierarchical backoff → grand mean fallback → global_fallback.
    """
    c, z, k = int(pitch_cluster_enc), int(zone_enc), int(count_enc)
    for key in ((c, z, k), (c, z), (c, k), (c,), (z, k), (z,)):
        vals = xba_lookup_table.get(key)
        if vals:
            return float(np.mean(vals))
    # Final fallback (already precomputed per-hitter as global_bip_xba, else 0.300)
    return float(global_fallback)

# Optional: lightweight logger (no-op by default)
def print_at_bat_log(log):
    # Replace with any pretty-printer you want; left blank to keep module light.
    pass


# ---------- Main simulation ----------

def simulate_at_bat_between(
    hitter,
    pitcher,
    nb_pitch_model,
    pitch_lookup_table,
    pitch_class_labels,
    nb_zone_model,
    zone_lookup_table,
    zone_class_labels,
    verbose=True,
    verbose_audit=False
):
    """
    Works with either full class instances or 'pack' namespaces that expose the same attributes.
    Hot-path optimizations:
      - O(1) handedness lookup via pitcher.stand_by_batter_lower
      - No label inverse_transform unless verbose_audit=True
      - No GI import (local _encode_count)
    """

    # --- Resolve pitcher dataframe from object ---
    pitcher_df = getattr(pitcher, "pitcher_data_arch", None)
    if pitcher_df is None:
        pitcher_df = getattr(pitcher, "pitcher_data", None)
    if pitcher_df is None or len(pitcher_df) == 0:
        raise ValueError("Pitcher object has no non-empty `pitcher_data_arch` or `pitcher_data`.")

    # --- Pull encoders/models from hitter object ---
    cluster_encoder = hitter.cluster_encoder
    stand_encoder   = hitter.stand_encoder
    arch_encoder    = getattr(hitter, "arch_encoder", None)
    outcome_encoder = hitter.outcome_encoder

    nb_outcome_model      = hitter.nb_outcome_model
    outcome_lookup_table  = hitter.outcome_lookup_table
    outcome_class_labels  = hitter.outcome_class_labels
    xba_lookup_table      = hitter.xba_lookup_table
    global_bip_xba        = float(getattr(hitter, "global_bip_xba", 0.300) or 0.300)

    # --- Resolve names (for logging only) ---
    hitter_name_str  = getattr(hitter, "full_lower", None) or \
                       f"{getattr(hitter, 'first_lower', 'unknown')} {getattr(hitter, 'last_lower', 'unknown')}"
    pitcher_name_str = getattr(pitcher, "full_lower", None) or \
                       f"{getattr(pitcher, 'first_lower', 'unknown')} {getattr(pitcher, 'last_lower', 'unknown')}"

    # --- Handedness lookup: O(1) via prebuilt dict; fallback cached on pitcher ---
    hitter_hand = None
    stand_map = getattr(pitcher, "stand_by_batter_lower", None)
    if isinstance(stand_map, dict):
        hitter_hand = stand_map.get(hitter_name_str)

    if hitter_hand is None:
        # fallback scan ONCE per (pitcher, hitter) and cache result on pitcher
        cache = getattr(pitcher, "_hand_cache", None)
        if cache is None:
            cache = {}
            setattr(pitcher, "_hand_cache", cache)
        if hitter_name_str in cache:
            hitter_hand = cache[hitter_name_str]
        else:
            # substring match to preserve original behavior
            mask = pitcher_df["batter_name"].str.contains(hitter_name_str, case=False, na=False)
            if not mask.any():
                raise ValueError(f"No matchup data found between {pitcher_name_str} and {hitter_name_str}")
            first_idx = mask.idxmax()
            hitter_hand = str(pitcher_df.loc[first_idx, "stand"])
            cache[hitter_name_str] = hitter_hand

    stand_enc = stand_encoder.transform([hitter_hand])[0]

    # --- Archetype encoding (prefer precomputed arch_enc on packs) ---
    if hasattr(hitter, "arch_enc") and hitter.arch_enc is not None:
        arch_enc = int(hitter.arch_enc)
    else:
        # fallback: derive from hitter_df/batting using arch_encoder
        hitter_rows = getattr(hitter, "batter_data", None)
        if hitter_rows is None:
            raise ValueError("arch_enc missing and no batter_data available to derive it.")
        rows = hitter_rows[hitter_rows['batter_name'].str.contains(hitter_name_str, case=False, na=False)]
        if rows.empty:
            raise ValueError(f"No archetype data found for {hitter_name_str}")
        if 'arch_enc' in rows.columns:
            arch_enc = int(rows['arch_enc'].values[0])
        else:
            if arch_encoder is None:
                raise ValueError("arch_enc not found and hitter.arch_encoder missing.")
            label = rows['Hitter_Archetype'].values[0] if 'Hitter_Archetype' in rows.columns \
                    else getattr(hitter, "hitter_archetype", None)
            if label is None:
                raise ValueError("Cannot infer hitter archetype label to encode.")
            arch_enc = int(arch_encoder.transform([label])[0])

    balls, strikes = 0, 0
    pitch_num = 1
    make_logs = bool(verbose or verbose_audit)
    log = [] if make_logs else None

    while True:
        count_enc = _encode_count(balls, strikes)

        # --- PITCH PREDICTION ---
        preds = hybrid_pitch_predict(
            [[stand_enc, count_enc, arch_enc]],
            nb_pitch_model, pitch_lookup_table, pitch_class_labels
        )
        pitch_global_id = preds[0]

        # Only compute human-readable label when auditing
        if verbose_audit:
            pitch_cluster_label = cluster_encoder.inverse_transform([pitch_global_id])[0]

        # --- ZONE PREDICTION ---
        zone_enc = predict_zone_hybrid(
            [[pitch_global_id, count_enc, stand_enc]],
            zone_nb_model=nb_zone_model,
            zone_lookup_table=zone_lookup_table,
            zone_class_labels=zone_class_labels
        )[0]

        # --- OUTCOME PREDICTION ---
        outcome_enc = predict_outcome(
            pitch_cluster_enc=pitch_global_id,
            zone_enc=zone_enc,
            count_enc=count_enc,
            nb_model=nb_outcome_model,
            lookup_table=outcome_lookup_table,
            class_labels=outcome_class_labels
        )
        outcome_label = outcome_encoder.inverse_transform([outcome_enc])[0]

        # --- Terminal outcomes first (fast path w/o logging) ---
        if not make_logs:
            if outcome_label == "bip":
                xba = predict_xba(pitch_global_id, zone_enc, count_enc, xba_lookup_table, global_fallback=global_bip_xba)
                return ('HIT', None) if (np.random.rand() < xba) else ('OUT', None)
            if outcome_label == "hbp":
                return 'HBP', None

        # --- Logging path for terminal outcomes ---
        if make_logs and outcome_label in ("bip", "hbp"):
            entry = {"Pitch #": pitch_num, "Zone": zone_enc + 1, "Outcome": outcome_label}
            if verbose_audit:
                entry["Pitch Cluster"] = pitch_cluster_label
            if outcome_label == "bip":
                xba = predict_xba(pitch_global_id, zone_enc, count_enc, xba_lookup_table, global_fallback=global_bip_xba)
                is_hit = np.random.rand() < xba
                entry["xBA"] = round(xba, 3)
                entry["BIP Result"] = "Hit" if is_hit else "Out"
                log.append(entry)
                if verbose: print_at_bat_log(log)
                return ('HIT', log) if is_hit else ('OUT', log)
            else:  # hbp
                entry["Terminal"] = "HBP"
                log.append(entry)
                if verbose: print_at_bat_log(log)
                return 'HBP', log

        # --- Count update ---
        # map: outcome -> (add_balls, add_strikes, max_2_flag)
        add_balls, add_strikes, max2 = {
            "ball":   (1, 0, False),
            "strike": (0, 1, False),
            "foul":   (0, 1, True),
            "hbp":    (0, 0, False),  # handled earlier
            "bip":    (0, 0, False),  # handled earlier
        }[outcome_label]

        if add_balls:
            balls += add_balls
        if add_strikes:
            if max2 and strikes >= 2:
                pass
            else:
                strikes += add_strikes

        # --- Terminal counts ---
        if balls >= 4:
            if make_logs:
                log.append({"Pitch #": pitch_num, "Terminal": "WALK"})
                if verbose: print_at_bat_log(log)
                return 'WALK', log
            return 'WALK', None

        if strikes >= 3:
            if make_logs:
                log.append({"Pitch #": pitch_num, "Terminal": "K"})
                if verbose: print_at_bat_log(log)
                return 'K', log
            return 'K', None

        if make_logs:
            entry = {"Pitch #": pitch_num, "Zone": zone_enc + 1, "Outcome": outcome_label}
            if verbose_audit:
                entry["Pitch Cluster"] = pitch_cluster_label
            log.append(entry)

        pitch_num += 1
