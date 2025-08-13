# Hitter_Class.py
from __future__ import annotations

import os
import unicodedata
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import requests

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import CategoricalNB

from pybaseball import playerid_lookup, playerid_reverse_lookup

# Project helpers
from General_Initialization import encode_count, get_archetype


class Hitter:
    def __init__(
        self,
        first_lower: str,
        first_upper: str,
        last_lower: str,
        last_upper: str,
        team_name: str,
        team_id: int,
        batter_data: pd.DataFrame,
        encoder_data: pd.DataFrame,
        bat_stats_2025: pd.DataFrame,
        batting: pd.DataFrame,
        player_id: Optional[int] = None,
    ):
        # ----- identity/data -----
        self.first_lower = first_lower
        self.first_upper = first_upper
        self.last_lower  = last_lower
        self.last_upper  = last_upper
        self.team_name   = team_name
        self.team_id     = team_id
        self.full_lower  = f"{first_lower} {last_lower}"
        self.full_upper  = f"{first_upper} {last_upper}"

        # Dataframes passed in
        self.batter_data    = batter_data
        self.encoder_data   = encoder_data
        self.bat_stats_2025 = bat_stats_2025
        self.batting        = batting

        # ----- player_id priority: use provided override, else robust lookup -----
        if player_id is not None:
            self.player_id = int(player_id)
        else:
            self.player_id = self._lookup_player_id()

        # ---- load encoders (trained elsewhere) ----
        self.stand_encoder   = joblib.load("encoders/stand_encoder.joblib")
        self.arch_encoder    = joblib.load("encoders/arch_encoder.joblib")
        self.outcome_encoder = joblib.load("encoders/outcome_encoder.joblib")

        # ---- archetype + encoded archetype ----
        self.hitter_archetype = get_archetype(self.last_upper, self.batting)
        if self.hitter_archetype is None:
            raise ValueError(f"Could not resolve Hitter_Archetype for {self.full_upper}.")
        self.arch_enc = self.arch_encoder.transform([self.hitter_archetype])[0]

        # ---- to be set by pipeline ----
        self.importances: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        self.gmm_models: Dict[Tuple[str, str], Tuple[GaussianMixture, StandardScaler, np.ndarray]] = {}
        self.cluster_encoder: Optional[LabelEncoder] = None
        self.encoder_data_with_clusters: Optional[pd.DataFrame] = None
        self.hitter_df: pd.DataFrame = pd.DataFrame()
        self.most_recent_spot: Optional[int] = None
        self.most_recent_date: Optional[str] = None
        self.winning_pct_value: Optional[float] = None
        self.xba: Optional[float] = None

        # outcome/xBA artifacts
        self.nb_outcome_model: Optional[CategoricalNB] = None
        self.outcome_lookup_table = None
        self.outcome_class_labels = None
        self.xba_lookup_table = None
        self.global_bip_xba: Optional[float] = None

        # ===== Pipeline =====
        self._fit_feature_importances()            # 1) RF importances per (R/L, group)
        self._fit_gmm_models()                     # 2) 12 hitter-weighted GMMs
        self._init_cluster_encoder_and_clusters()  # 3) LabelEncoder from GMM label space
        self._prepare_hitter_data()                # 4) assign clusters + encodings on hitter data
        self._build_xba_lookup()                   # 5) robust, no-crash
        self._train_outcome_model()                # 6) hybrid outcome artifacts

        # Metadata (optional)
        self._get_player_metadata()
        self.winning_pct_value = self._winning_pct()

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def _strip_accents(text: str) -> str:
        if not isinstance(text, str):
            return text
        return ''.join(
            c for c in unicodedata.normalize('NFD', text)
            if unicodedata.category(c) != 'Mn'
        )

    def _lookup_player_id(self) -> int:
        first = self._strip_accents(self.first_upper).strip()
        last  = self._strip_accents(self.last_upper).strip()

        df = playerid_lookup(last, first)
        if df.empty:
            df = playerid_lookup(last.upper(), first.upper())
        if df.empty:
            df = playerid_lookup(first, last)
        if df.empty:
            try:
                rev = playerid_reverse_lookup([f"{first} {last}"])
                if not rev.empty and "key_mlbam" in rev:
                    rev = rev[rev["key_mlbam"].notna()]
                    if not rev.empty:
                        return int(rev.iloc[0]["key_mlbam"])
            except Exception:
                pass

        if not df.empty and "key_mlbam" in df.columns and df["key_mlbam"].notna().any():
            return int(df["key_mlbam"].dropna().astype(int).iloc[0])

        raise ValueError(
            f"MLBAM id not found for hitter '{first} {last}'. "
            "Tried multiple name variants. Double-check spelling (accents, Jr./Sr.) "
            "or pass an explicit player_id override."
        )
    @staticmethod
    def _ensure_pitch_group(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a 'pitch_group' column exists based on 'pitch_type'.
        Drops rows with missing pitch_type, but raises an error if >10% are dropped.
        """
        if 'pitch_group' in df.columns:
            return df
    
        if 'pitch_type' not in df.columns:
            raise ValueError("Missing 'pitch_type' column; cannot derive     'pitch_group'.")
    
        out = df.copy()
    
        # Drop rows with missing pitch_type
        total_rows = len(out)
        out = out.dropna(subset=['pitch_type'])
        dropped = total_rows - len(out)
    
        if total_rows > 0 and (dropped / total_rows) > 0.10:
            raise ValueError(
                f"Over 10% of rows ({dropped}/{total_rows}) missing 'pitch_type'. "
                "Likely systemic data issue."
            )
    
        pitch_group_map = {
            'FF': '4F', 'FA': '4F', 'FC': 'CF', 'SI': '2F', 'FT': '2F',
            'SL': 'S',  'ST': 'S',  'CU': 'C',  'KC': 'C',  'CS': 'C',
            'CH': 'CH', 'FS': 'CH'
        }
        out['pitch_group'] = out['pitch_type'].map(pitch_group_map)
    
        return out


    # ---------- Feature importance ----------
    def _filter_and_fit_importance(self, hand: str, group: str):
        bdf = self._ensure_pitch_group(self.batter_data)
        if 'batter_name' not in bdf.columns:
            return 0.3333, 0.3333, 0.3333

        df = bdf[
            (bdf['batter_name'].str.lower() == self.full_lower) &
            (bdf['p_throws'] == hand) &
            (bdf['pitch_group'] == group)
        ].dropna(subset=['release_speed', 'pfx_x', 'pfx_z', 'estimated_ba_using_speedangle'])

        df = df[df['estimated_ba_using_speedangle'] > 0]
        if len(df) < 30:
            return 0.3333, 0.3333, 0.3333

        X = df[['release_speed', 'pfx_x', 'pfx_z']]
        y = df['estimated_ba_using_speedangle']
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        return float(imp['release_speed']), float(imp['pfx_x']), float(imp['pfx_z'])

    def _fit_feature_importances(self):
        for hand in ['R', 'L']:
            for group in ['4F', '2F', 'CF', 'S', 'C', 'CH']:
                self.importances[(hand, group)] = self._filter_and_fit_importance(hand, group)

    # ---------- 12 GMMs ----------
    def _fit_gmm_models(self):
        pitch_group_map = {
            'FF': '4F', 'FA': '4F', 'FC': 'CF', 'SI': '2F', 'FT': '2F',
            'SL': 'S',  'ST': 'S',  'CU': 'C',  'KC': 'C',  'CS': 'C',
            'CH': 'CH', 'FS': 'CH'
        }
        group_map = {
            '4F': ['FF', 'FA'], '2F': ['SI', 'FT'], 'CF': ['FC'],
            'S':  ['SL', 'ST'], 'C':  ['CU', 'KC', 'CS'], 'CH': ['CH', 'FS']
        }
        p = pd.read_pickle("pitcher_data.pkl")
        p = p[p['pitch_type'].isin(pitch_group_map.keys())].copy()
        p['pitch_group'] = p['pitch_type'].map(pitch_group_map)
        base = p[['pitch_type', 'pitch_group', 'p_throws', 'release_speed', 'pfx_x', 'pfx_z']].dropna()

        for hand in ['R', 'L']:
            for group in ['4F', '2F', 'CF', 'S', 'C', 'CH']:
                df = base[
                    (base['p_throws'] == hand) &
                    (base['pitch_type'].isin(group_map[group]))
                ][['release_speed', 'pfx_x', 'pfx_z']].dropna()

                if df.empty:
                    continue

                scaler = StandardScaler()
                X = scaler.fit_transform(df)
                weights = np.array(self.importances.get((hand, group), (1.0, 1.0, 1.0)), dtype=float)
                Xw = X * weights
                n_components = 5 if group == 'S' else 4 if group == '4F' else 2 if group in ['CF', 'CH', 'C'] else 3
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(Xw)
                self.gmm_models[(hand, group)] = (gmm, scaler, weights)

        for (h, g), triple in self.gmm_models.items():
            setattr(self, f"gmm_{h}_{g}", triple)

    # ---------- NEW: Build deterministic label vocabulary from GMMs ----------
    def _build_cluster_label_vocab(self) -> list[str]:
        """
        Build the complete label vocabulary directly from the fitted GMMs.
        Guarantees the encoder knows every label _assign_row_cluster can emit.
        """
        if not self.gmm_models:
            raise RuntimeError(
                "No GMMs trained; cannot build cluster label vocabulary. "
                "Check pitcher_data.pkl and _fit_gmm_models() inputs."
            )
        labels = []
        for (hand, group), (gmm, _, _) in self.gmm_models.items():
            n = int(getattr(gmm, "n_components", 0) or 0)
            if n <= 0:
                raise RuntimeError(f"GMM for {(hand, group)} has zero components.")
            labels.extend([f"{hand}{group}{i+1}" for i in range(n)])
        return sorted(set(labels))

    # ---------- Canonical cluster encoder (now from GMM label space) ----------
    def _init_cluster_encoder_and_clusters(self):
        """
        Fit the hitter's cluster_encoder directly from the hitter's GMM label space.
        This guarantees the encoder is always fitted with the full set of local cluster labels,
        independent of how many rows exist in encoder_data.
        """
        # Build expected labels straight from the fitted GMMs
        if not getattr(self, "gmm_models", None):
            raise RuntimeError(f"No GMMs present for {self.full_upper}; cannot build label space.")
    
        labels = []
        for (hand, group), (gmm, _, _) in self.gmm_models.items():
            n = int(getattr(gmm, "n_components", 0) or 0)
            if n <= 0:
                raise RuntimeError(f"GMM for {(hand, group)} has no components.")
            labels.extend([f"{hand}{group}{i+1}" for i in range(n)])
        classes = sorted(set(labels))
        if not classes:
            raise RuntimeError(f"Empty label space synthesized for {self.full_upper}.")
    
        # Fit encoder on the full local label space
        enc = LabelEncoder()
        enc.fit(classes)
        self.cluster_encoder = enc
        self.allowed_cluster_labels = np.array(enc.classes_)  # optional: for debugging
    
        # Keep a convenience view of encoder_data with assigned clusters (not used to fit)
        edf = self._ensure_pitch_group(self.encoder_data.copy())
        edf['pitch_cluster'] = edf.apply(self._assign_row_cluster, axis=1)
        self.encoder_data_with_clusters = edf
    
        os.makedirs("encoders", exist_ok=True)
        joblib.dump(self.cluster_encoder, f"encoders/cluster_encoder__{self.last_lower}.joblib")


    # ---------- Prepare hitter rows (assign + encode) ----------
    def _prepare_hitter_data(self):
        # Player rows (may be empty)
        if 'batter_name' in self.batter_data.columns:
            df = self.batter_data[self.batter_data['batter_name'].str.lower() == self.full_lower].copy()
        else:
            df = self.batter_data.iloc[0:0].copy()

        df = self._ensure_pitch_group(df)

        # Assign clusters (may all be NaN if GMMs missing features)
        df['pitch_cluster'] = df.apply(self._assign_row_cluster, axis=1)

        # Ensure required columns exist
        must_have = [
            'pitch_cluster', 'zone', 'balls', 'strikes', 'stand', 'description',
            'estimated_ba_using_speedangle', 'Hitter_Archetype'
        ]
        for c in must_have:
            if c not in df.columns:
                df[c] = np.nan

        # Map description → simple tokens
        def _map_desc(desc):
            if desc in ['ball', 'blocked_ball', 'pitchout']: return 'ball'
            if desc in ['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'missed_bunt']: return 'strike'
            if desc in ['foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip']: return 'foul'
            if desc == 'hit_by_pitch': return 'hbp'
            if desc == 'hit_into_play': return 'bip'
            return 'unknown'

        df['description'] = df['description'].apply(_map_desc)

        # Archetype tag (constant for this hitter)
        df['Hitter_Archetype'] = self.hitter_archetype

        # Cluster encode (no unseen labels now)
        # Cluster encode (no fitting here — just transform)
        if getattr(self, "cluster_encoder", None) is None or not hasattr(self.cluster_encoder, "classes_"):
            raise RuntimeError(f"cluster_encoder missing for {self.full_upper} (should have been created from GMMs).")
        
        valid = set(self.cluster_encoder.classes_)
        mask = df['pitch_cluster'].isin(valid) & df['pitch_cluster'].notna()
        df.loc[mask, 'pitch_cluster_enc'] = self.cluster_encoder.transform(df.loc[mask, 'pitch_cluster'])
        df.loc[~mask, 'pitch_cluster_enc'] = np.nan

        # Ensure numeric basics
        df['balls']   = pd.to_numeric(df['balls'], errors='coerce').fillna(0).astype(int)
        df['strikes'] = pd.to_numeric(df['strikes'], errors='coerce').fillna(0).astype(int)
        df['zone_enc'] = (pd.to_numeric(df['zone'], errors='coerce') - 1).round().astype('Int64')

        # ---- count_enc: guaranteed scalar ----
        def _safe_encode_count_scalar(b, s):
            try:
                out = encode_count(int(b), int(s))
            except Exception:
                return np.nan
            if isinstance(out, dict):
                if not out:
                    return np.nan
                k_max = max(out.items(), key=lambda kv: float(kv[1]))[0]
                try: return int(k_max)
                except Exception: return np.nan
            if isinstance(out, (list, tuple, np.ndarray, pd.Series)):
                arr = np.asarray(out, dtype=float).ravel()
                if arr.size == 0 or np.isnan(arr).all(): return np.nan
                return int(np.nanargmax(arr))
            try:
                return int(out)
            except Exception:
                return np.nan

        df['count_enc'] = pd.Series(
            (_safe_encode_count_scalar(b, s) for b, s in zip(df['balls'], df['strikes'])),
            index=df.index
        ).astype('Int64')

        # stand_enc
        if df['stand'].notna().any():
            mask = df['stand'].notna()
            df.loc[mask, 'stand_enc'] = self.stand_encoder.transform(df.loc[mask, 'stand'])
            df.loc[~mask, 'stand_enc'] = np.nan
        else:
            df['stand_enc'] = np.nan

        # outcome_enc
        if df['description'].notna().any():
            mask = df['description'].notna()
            df.loc[mask, 'outcome_enc'] = self.outcome_encoder.transform(df.loc[mask, 'description'])
            df.loc[~mask, 'outcome_enc'] = np.nan
        else:
            df['outcome_enc'] = np.nan

        # arch_enc (constant)
        df['arch_enc'] = self.arch_encoder.transform(df['Hitter_Archetype'].fillna('Unknown'))

        # Final set (allow empty)
        keep_cols = [
            'pitch_cluster_enc', 'zone_enc', 'count_enc', 'outcome_enc', 'arch_enc',
            'estimated_ba_using_speedangle', 'description'
        ]
        extra_cols = ['pitch_cluster', 'zone', 'balls', 'strikes', 'stand', 'Hitter_Archetype']
        self.hitter_df = df[keep_cols + extra_cols].copy()

        # Persist (even if empty)
        self._persist_hitter_df()

        # Seasonal xBA fallback
        row = self.bat_stats_2025[self.bat_stats_2025['Name'].str.contains(self.last_upper, case=False, na=False)]
        self.xba = float(row.iloc[0]['xBA']) if not row.empty else 0.300

    # ---------- assign cluster using hitter GMMs ----------
    def _assign_row_cluster(self, row: pd.Series) -> str | float:
        hand  = row.get('p_throws', None)
        group = row.get('pitch_group', None)
        if hand not in ['R', 'L'] or group not in ['4F', '2F', 'CF', 'S', 'C', 'CH']:
            return np.nan
        key = (hand, group)
        if key not in self.gmm_models:
            return np.nan
        if pd.isna(row.get('release_speed')) or pd.isna(row.get('pfx_x')) or pd.isna(row.get('pfx_z')):
            return np.nan
        gmm, scaler, weights = self.gmm_models[key]
        x_df = pd.DataFrame(
            [[row['release_speed'], row['pfx_x'], row['pfx_z']]],
            columns=['release_speed', 'pfx_x', 'pfx_z']
        )
        xw = scaler.transform(x_df) * weights
        comp = int(gmm.predict(xw)[0])
        return f"{hand}{group}{comp+1}"

    # ---------- build per-hitter xBA lookup (robust) ----------
    def _build_xba_lookup(self):
        os.makedirs("models", exist_ok=True)

        if not hasattr(self, "hitter_df") or self.hitter_df is None or self.hitter_df.empty:
            self.global_bip_xba = float(getattr(self, "xba", 0.300) or 0.300)
            self.xba_lookup_table = {
                "L3": {}, "L2": {}, "L1": {},
                "G": {('__GLOBAL__',): {"sum": self.global_bip_xba, "n": 1}},
                "min_support": (20, 30, 40),
                "prior_equiv": (20, 40, 80),
                "mode": "mean",
                "schema": "hierarchical_sum_n_v1"
            }
            joblib.dump(self.xba_lookup_table, "models/XBA_Lookup_Hierarchical.joblib")
            return

        needed = ['estimated_ba_using_speedangle', 'pitch_cluster_enc', 'zone_enc', 'count_enc', 'description']
        for c in needed:
            if c not in self.hitter_df.columns:
                self.hitter_df[c] = np.nan

        df = self.hitter_df.dropna(subset=['pitch_cluster_enc', 'zone_enc', 'count_enc']).copy()
        bip = df[(df['description'] == 'bip') & df['estimated_ba_using_speedangle'].notna()].copy()

        if bip.empty:
            self.global_bip_xba = float(getattr(self, "xba", 0.300) or 0.300)
            self.xba_lookup_table = {
                "L3": {}, "L2": {}, "L1": {},
                "G": {('__GLOBAL__',): {"sum": self.global_bip_xba, "n": 1}},
                "min_support": (20, 30, 40),
                "prior_equiv": (20, 40, 80),
                "mode": "mean",
                "schema": "hierarchical_sum_n_v1"
            }
            joblib.dump(self.xba_lookup_table, "models/XBA_Lookup_Hierarchical.joblib")
            return

        self.global_bip_xba = float(
            bip['estimated_ba_using_speedangle'].astype(float).clip(0.0, 1.0).mean()
        )

        L3, L2, L1 = {}, {}, {}
        G  = {('__GLOBAL__',): {"sum": 0.0, "n": 0}}

        def _add(tbl, key, x):
            if key not in tbl:
                tbl[key] = {"sum": float(x), "n": 1}
            else:
                tbl[key]["sum"] += float(x)
                tbl[key]["n"]   += 1

        for row in bip.itertuples(index=False):
            p = int(getattr(row, 'pitch_cluster_enc'))
            z = int(getattr(row, 'zone_enc'))
            k = int(getattr(row, 'count_enc'))
            x = float(getattr(row, 'estimated_ba_using_speedangle'))
            if 0.0 <= x <= 1.0:
                _add(L3, (p, z, k), x)
                _add(L2, (p, z), x)
                _add(L1, (p,), x)
                _add(G, ('__GLOBAL__',), x)

        self.xba_lookup_table = {
            "L3": L3, "L2": L2, "L1": L1, "G": G,
            "min_support": (20, 30, 40),
            "prior_equiv": (20, 40, 80),
            "mode": "mean",
            "schema": "hierarchical_sum_n_v1"
        }
        joblib.dump(self.xba_lookup_table, "models/XBA_Lookup_Hierarchical.joblib")

    # ---------- build per-hitter outcome hybrid (lookup + NB) ----------
    def _train_outcome_model(self, min_lookup_count: int = 5):
        os.makedirs("models", exist_ok=True)

        required = ['pitch_cluster_enc', 'zone_enc', 'count_enc', 'outcome_enc']
        if not hasattr(self, 'hitter_df') or any(c not in self.hitter_df.columns for c in required):
            self.outcome_lookup_table = defaultdict(Counter)
            nb = CategoricalNB()
            X_dummy = np.array([[0, 0, 0], [0, 0, 0]])
            y_dummy = np.array([0, 1])
            nb.fit(X_dummy, y_dummy)
            self.nb_outcome_model = nb
            self.outcome_class_labels = nb.classes_
            joblib.dump(self.nb_outcome_model,     "models/hybrid_outcome_nb_model.joblib")
            joblib.dump(self.outcome_lookup_table, "models/hybrid_outcome_lookup_table.joblib")
            joblib.dump(self.outcome_class_labels, "models/hybrid_outcome_class_labels.joblib")
            return

        df = self.hitter_df.dropna(subset=required).copy()

        if df.empty:
            self.outcome_lookup_table = defaultdict(Counter)
            nb = CategoricalNB()
            X_dummy = np.array([[0, 0, 0], [0, 0, 0]])
            y_dummy = np.array([0, 1])
            nb.fit(X_dummy, y_dummy)
            self.nb_outcome_model = nb
            self.outcome_class_labels = nb.classes_
        else:
            X = df[['pitch_cluster_enc', 'zone_enc', 'count_enc']].astype(int).values
            y = df['outcome_enc'].astype(int).values

            lookup_table = defaultdict(Counter)
            for feats, target in zip(X, y):
                lookup_table[tuple(feats)][int(target)] += 1

            nb_model = CategoricalNB()
            nb_model.fit(X, y)

            self.outcome_lookup_table = lookup_table
            self.nb_outcome_model = nb_model
            self.outcome_class_labels = nb_model.classes_

        joblib.dump(self.nb_outcome_model,     "models/hybrid_outcome_nb_model.joblib")
        joblib.dump(self.outcome_lookup_table, "models/hybrid_outcome_lookup_table.joblib")
        joblib.dump(self.outcome_class_labels, "models/hybrid_outcome_class_labels.joblib")

    # ---------- persistence ----------
    def _persist_hitter_df(self):
        os.makedirs("data", exist_ok=True)
        self.hitter_df.to_pickle(f"data/{self.last_lower}_data.pkl")

    # ---------- metadata ----------
    def _get_player_metadata(self):
        def get_recent_game_ids(team_id, lookback_days=14):
            today = datetime.utcnow().date()
            start, end = (today - timedelta(days=lookback_days)).isoformat(), today.isoformat()
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}&startDate={start}&endDate={end}"
            try:
                resp = requests.get(url, timeout=10)
                dates = resp.json().get('dates', [])
                return [
                    {'gamePk': g['gamePk'], 'date': g['gameDate'][:10]}
                    for d in dates for g in d.get('games', [])
                    if g.get('status', {}).get('detailedState') in [
                        "Final", "Completed Early", "In Progress", "In Progress - Delay"
                    ]
                ]
            except Exception:
                return []

        def get_batting_order_spot(game_id, player_id):
            try:
                url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
                team = requests.get(url, timeout=10).json()['liveData']['boxscore']['teams']
                for side in ['home', 'away']:
                    for p in team[side]['players'].values():
                        if str(p['person']['id']) == str(player_id):
                            return int(str(p.get('battingOrder', '0'))[0]) if 'battingOrder' in p else None
            except Exception:
                pass
            return None

        if self.player_id is not None and self.team_id is not None:
            games = sorted(get_recent_game_ids(self.team_id), key=lambda x: x['date'], reverse=True)
            for g in games:
                spot = get_batting_order_spot(g['gamePk'], self.player_id)
                if spot:
                    self.most_recent_spot = spot
                    self.most_recent_date = g['date']
                    break

    def _winning_pct(self) -> Optional[float]:
        nickname_to_full_name = {
            "Yankees": "New York Yankees", "Red Sox": "Boston Red Sox",
            "Blue Jays": "Toronto Blue Jays", "Orioles": "Baltimore Orioles",
            "Rays": "Tampa Bay Rays", "White Sox": "Chicago White Sox",
            "Guardians": "Cleveland Guardians", "Tigers": "Detroit Tigers",
            "Royals": "Kansas City Royals", "Twins": "Minnesota Twins",
            "Astros": "Houston Astros", "Mariners": "Seattle Mariners",
            "Athletics": "Oakland Athletics", "Rangers": "Texas Rangers",
            "Angels": "Los Angeles Angels", "Braves": "Atlanta Braves",
            "Phillies": "Philadelphia Phillies", "Marlins": "Miami Marlins",
            "Nationals": "Washington Nationals", "Mets": "New York Mets",
            "Cubs": "Chicago Cubs", "Cardinals": "St. Louis Cardinals",
            "Pirates": "Pittsburgh Pirates", "Reds": "Cincinnati Reds",
            "Brewers": "Milwaukee Brewers", "Dodgers": "Los Angeles Dodgers",
            "Giants": "San Francisco Giants", "Padres": "San Diego Padres",
            "Diamondbacks": "Arizona Diamondbacks", "Rockies": "Colorado Rockies"
        }
        if self.team_name not in nickname_to_full_name:
            return None
        full_team_name = nickname_to_full_name[self.team_name]
        url = "https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season=2025"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            for record in data.get('records', []):
                for team_record in record.get('teamRecords', []):
                    if team_record['team']['name'] == full_team_name:
                        return float(team_record['winningPercentage'])
        except Exception:
            pass
        return None
