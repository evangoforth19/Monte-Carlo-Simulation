from baseball_utils import *
batter_data = pd.read_pickle("batter_data.pkl")
pitcher_data = pd.read_pickle("pitcher_data.pkl")
batting = pd.read_pickle("batting_2025.pkl")
encoder_data = pd.read_pickle("encoder_data.pkl")
bat_stats_2025 = pd.read_pickle("bat_stats_2025.pkl")

from General_Initialization import (
    encode_count, pitch_group_map, features, elite_woba_threshold,
    power_barrel_threshold, patient_chase_threshold, get_archetype, classify_archetype,
    assign_pitch_cluster, map_description_to_simple, prob_extra_innings,
    OUT_EVENTS, MULTI_OUTS, outs_from_event, pitcher_team, outs_df,
    convert_outs_to_ip, sample_extra_innings, home_IP_extras, away_IP_extras 
)

stand_encoder     = joblib.load("encoders/stand_encoder.joblib")
arch_encoder      = joblib.load("encoders/arch_encoder.joblib")
pitch_encoder     = joblib.load("encoders/pitch_encoder.joblib")
outcome_encoder   = joblib.load("encoders/outcome_encoder.joblib")

hitter_arch = batting.apply(classify_archetype, axis = 1)
batting['Hitter_Archetype'] = hitter_arch

class Hitter:
    def __init__(self, first_lower, first_upper, last_lower, last_upper,
                 team_name, team_id, batter_data, encoder_data,
                 bat_stats_2025, batting, player_id):
        # ----- identity/data -----
        self.first_lower = first_lower
        self.first_upper = first_upper
        self.last_lower = last_lower
        self.last_upper = last_upper
        self.team_name = team_name
        self.team_id = team_id
        self.full_lower = f"{first_lower} {last_lower}"
        self.full_upper = f"{first_upper} {last_upper}"
        self.batter_data = batter_data
        self.encoder_data = encoder_data
        self.bat_stats_2025 = bat_stats_2025
        self.batting = batting
        self.player_id = player_id

        # ---- encoders already trained elsewhere ----
        self.stand_encoder   = joblib.load("encoders/stand_encoder.joblib")
        self.arch_encoder    = joblib.load("encoders/arch_encoder.joblib")
        self.outcome_encoder = joblib.load("encoders/outcome_encoder.joblib")

        # ---- NEW: set hitter_archetype and arch_enc robustly ----
        self.hitter_archetype = None
        self.arch_enc = None

        # find a usable name column in self.batting
        name_col = None
        if isinstance(self.batting, pd.DataFrame):
            for cand in ("Name", "player_name", "batter_name"):
                if cand in self.batting.columns:
                    name_col = cand
                    break

        # try to pull the precomputed archetype from batting
        if name_col and "Hitter_Archetype" in self.batting.columns:
            _rows = self.batting[self.batting[name_col].str.lower() == self.full_lower]
            if not _rows.empty:
                self.hitter_archetype = _rows["Hitter_Archetype"].iloc[0]

        # fallback: compute via classify_archetype if not found
        if self.hitter_archetype is None:
            try:
                if name_col:
                    _row = self.batting[self.batting[name_col].str.lower() == self.full_lower]
                    if not _row.empty:
                        self.hitter_archetype = classify_archetype(_row.iloc[0])
            except Exception:
                pass

        # last resort: General_Initialization.get_archetype()
        if self.hitter_archetype is None:
            try:
                self.hitter_archetype = get_archetype(self.last_upper, self.batting)
            except Exception:
                pass

        # encode archetype → arch_enc
        if self.hitter_archetype is None:
            raise ValueError(f"Could not resolve Hitter_Archetype for {self.full_upper}.")
        if self.arch_encoder is None:
            raise ValueError("arch_encoder must be loaded before encoding archetype.")
        self.arch_enc = self.arch_encoder.transform([self.hitter_archetype])[0]

        # ---- will be set by our pipeline ----
        self.importances = {}        # (hand, group) -> (w_speed, w_pfx_x, w_pfx_z)
        self.gmm_models = {}         # (hand, group) -> (gmm, scaler, weights)
        self.cluster_encoder = None  # LabelEncoder fit on encoder_data clusters
        self.encoder_data_with_clusters = None
        self.hitter_df = pd.DataFrame()
        self.most_recent_spot = None
        self.most_recent_date = None
        self.winning_pct_value = None
        self.xba = None

        # outcome/xBA artifacts (built later in this __init__)
        self.nb_outcome_model = None
        self.outcome_lookup_table = None
        self.outcome_class_labels = None
        self.xba_lookup_table = None
        self.global_bip_xba = None

        # ===== Pipeline =====
        self._fit_feature_importances()           # 1) RF importances per (R/L, group)
        self._fit_gmm_models()                    # 2) 12 hitter-weighted GMMs
        self._init_cluster_encoder_and_clusters() # 3) fit LabelEncoder on encoder_data clusters
        self._prepare_hitter_data()               # 4) assign clusters + encodings on hitter data

        # 5) Per-hitter models built AFTER hitter_df is ready
        self._build_xba_lookup()
        self._train_outcome_model()

        # Metadata (not required for models)
        self._get_player_metadata()
        self.winning_pct_value = self._winning_pct()

    # ---------- Feature importance ----------
    def _filter_and_fit_importance(self, hand, group):
        bdf = self._ensure_pitch_group(self.batter_data)
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
                weights = np.array(self.importances[(hand, group)], dtype=float)
                Xw = X * weights
                n_components = 5 if group in ['S'] else 4 if group in ['4F'] else 2 if group in ['CF', 'CH', 'C'] else 3
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(Xw)
                self.gmm_models[(hand, group)] = (gmm, scaler, weights)

        # expose individually too
        for (h, g), triple in self.gmm_models.items():
            setattr(self, f"gmm_{h}_{g}", triple)

    # ---------- Canonical cluster encoder on encoder_data ----------
    def _init_cluster_encoder_and_clusters(self):
        edf = self._ensure_pitch_group(self.encoder_data.copy())

        edf['pitch_cluster'] = edf.apply(self._assign_row_cluster, axis=1)
        edf = edf.dropna(subset=['pitch_cluster'])

        self.cluster_encoder = LabelEncoder()
        if edf.empty:
            self.cluster_encoder.fit(np.array([], dtype=str))
        else:
            self.cluster_encoder.fit(edf['pitch_cluster'])

        self.encoder_data_with_clusters = edf.copy()

        os.makedirs("encoders", exist_ok=True)
        joblib.dump(self.cluster_encoder, f"encoders/cluster_encoder__{self.last_lower}.joblib")

    # ---------- Prepare hitter rows (assign + encode) ----------
    def _prepare_hitter_data(self):
        df = self.batter_data[self.batter_data['batter_name'].str.lower() == self.full_lower].copy()
        df = self._ensure_pitch_group(df)

        df['pitch_cluster'] = df.apply(self._assign_row_cluster, axis=1)
        df = df.dropna(subset=['pitch_cluster'])

        valid = set(getattr(self.cluster_encoder, 'classes_', []))
        if len(valid) > 0:
            df = df[df['pitch_cluster'].isin(valid)]
            if df.empty:
                self.hitter_df = pd.DataFrame()
                self._persist_hitter_df()
                return
            df['pitch_cluster_enc'] = self.cluster_encoder.transform(df['pitch_cluster'])
        else:
            self.hitter_df = pd.DataFrame()
            self._persist_hitter_df()
            return

        hitter_arch = get_archetype(self.last_upper, self.batting)
        df['Hitter_Archetype'] = hitter_arch

        def map_description_to_simple(desc):
            if desc in ['ball', 'blocked_ball', 'pitchout']: return 'ball'
            elif desc in ['called_strike', 'swinging_strike', 'swinging_strike_blocked', 'missed_bunt']: return 'strike'
            elif desc in ['foul', 'foul_tip', 'foul_bunt', 'bunt_foul_tip']: return 'foul'
            elif desc == 'hit_by_pitch': return 'hbp'
            elif desc == 'hit_into_play': return 'bip'
            else: return 'unknown'

        df['simple_description'] = df['description'].apply(map_description_to_simple)
        df = df[df['simple_description'].notna() & (df['simple_description'] != 'unknown')]
        df['description'] = df['simple_description']
        df.drop(columns='simple_description', inplace=True)

        df = df.dropna(subset=['zone', 'balls', 'strikes', 'stand', 'description', 'Hitter_Archetype', 'pitch_cluster'])
        df['zone_enc']   = df['zone'].astype(int) - 1
        df['count_enc']  = df.apply(lambda r: encode_count(r['balls'], r['strikes']), axis=1)
        df['stand_enc']  = self.stand_encoder.transform(df['stand'])
        df['outcome_enc'] = self.outcome_encoder.transform(df['description'])
        df['arch_enc']   = self.arch_encoder.transform(df['Hitter_Archetype'].fillna('Unknown'))

        self.hitter_df = df
        self._persist_hitter_df()

        # seasonal xBA grab (for xBA fallback)
        row = self.bat_stats_2025[self.bat_stats_2025['Name'].str.contains(self.last_upper, case=False)]
        self.xba = float(row.iloc[0]['xBA']) if not row.empty else 0.300

    # ---------- assign cluster using hitter GMMs ----------
    def _assign_row_cluster(self, row):
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
        x_df = pd.DataFrame([[row['release_speed'], row['pfx_x'], row['pfx_z']]],columns=['release_speed', 'pfx_x', 'pfx_z'])
        xw = scaler.transform(x_df) * weights
        comp = int(gmm.predict(xw)[0])
        return f"{hand}{group}{comp+1}"

    # ---------- build per-hitter xBA lookup ----------
    def _build_xba_lookup(self):
        os.makedirs("models", exist_ok=True)

        needed = ['estimated_ba_using_speedangle', 'pitch_cluster_enc', 'zone_enc', 'count_enc', 'description']
        if not hasattr(self, 'hitter_df') or any(c not in self.hitter_df.columns for c in needed):
            raise ValueError("hitter_df not prepared or missing required columns for xBA lookup.")

        df = self.hitter_df.dropna(subset=['pitch_cluster_enc', 'zone_enc', 'count_enc']).copy()
        bip = df[(df['description'] == 'bip') & df['estimated_ba_using_speedangle'].notna()].copy()

        xba_lookup_table = defaultdict(list)

        if not bip.empty:
            for _, row in bip.iterrows():
                c = int(row['pitch_cluster_enc'])
                z = int(row['zone_enc'])
                k = int(row['count_enc'])
                x = float(row['estimated_ba_using_speedangle'])

                xba_lookup_table[(c, z, k)].append(x)
                xba_lookup_table[(c, z)].append(x)
                xba_lookup_table[(c, k)].append(x)
                xba_lookup_table[(c,)].append(x)
                xba_lookup_table[(z, k)].append(x)
                xba_lookup_table[(z,)].append(x)

            self.global_bip_xba = float(bip['estimated_ba_using_speedangle'].mean())
        else:
            self.global_bip_xba = float(self.xba) if self.xba is not None else 0.300

        self.xba_lookup_table = xba_lookup_table
        joblib.dump(self.xba_lookup_table, "models/XBA_Lookup_Hierarchical.joblib")

    # ---------- build per-hitter outcome hybrid (lookup + NB) ----------
    def _train_outcome_model(self, min_lookup_count: int = 5):
        """
        Builds per-hitter hybrid outcome artifacts and attaches:
          self.nb_outcome_model, self.outcome_lookup_table, self.outcome_class_labels
        Also saves to disk for debugging.
        """
        os.makedirs("models", exist_ok=True)

        required = ['pitch_cluster_enc', 'zone_enc', 'count_enc', 'outcome_enc']
        if not hasattr(self, 'hitter_df') or any(c not in self.hitter_df.columns for c in required):
            raise ValueError("hitter_df not prepared or missing required columns for outcome model.")

        df = self.hitter_df.dropna(subset=required).copy()

        if df.empty:
            # safe empties
            self.outcome_lookup_table = defaultdict(Counter)
            nb = CategoricalNB()
            X_dummy = np.array([[0,0,0],[0,0,0]])
            y_dummy = np.array([0,1])
            nb.fit(X_dummy, y_dummy)
            self.nb_outcome_model = nb
            self.outcome_class_labels = nb.classes_
        else:
            X = df[['pitch_cluster_enc', 'zone_enc', 'count_enc']].astype(int).values
            y = df['outcome_enc'].astype(int).values

            # lookup table on full hitter rows
            lookup_table = defaultdict(Counter)
            for feats, target in zip(X, y):
                lookup_table[tuple(feats)][int(target)] += 1

            nb_model = CategoricalNB()
            nb_model.fit(X, y)

            self.outcome_lookup_table = lookup_table
            self.nb_outcome_model = nb_model
            self.outcome_class_labels = nb_model.classes_

        # save (optional)
        joblib.dump(self.nb_outcome_model,     "models/hybrid_outcome_nb_model.joblib")
        joblib.dump(self.outcome_lookup_table, "models/hybrid_outcome_lookup_table.joblib")
        joblib.dump(self.outcome_class_labels, "models/hybrid_outcome_class_labels.joblib")

    # ---------- helpers ----------
    @staticmethod
    def _ensure_pitch_group(df):
        if 'pitch_group' in df.columns:
            return df
        pitch_group_map = {
            'FF': '4F', 'FA': '4F', 'FC': 'CF', 'SI': '2F', 'FT': '2F',
            'SL': 'S',  'ST': 'S',  'CU': 'C',  'KC': 'C',  'CS': 'C',
            'CH': 'CH', 'FS': 'CH'
        }
        out = df.copy()
        if 'pitch_type' in out.columns:
            out['pitch_group'] = out['pitch_type'].map(pitch_group_map)
        else:
            out['pitch_group'] = np.nan
        return out

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
                    if g.get('status', {}).get('detailedState') in ["Final", "Completed Early", "In Progress", "In Progress - Delay"]
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

    def _winning_pct(self):
        nickname_to_full_name = {
            "Yankees": "New York Yankees", "Mets": "New York Mets", "Red Sox": "Boston Red Sox",
            "Blue Jays": "Toronto Blue Jays", "Orioles": "Baltimore Orioles", "Rays": "Tampa Bay Rays",
            "White Sox": "Chicago White Sox", "Guardians": "Cleveland Guardians", "Tigers": "Detroit Tigers",
            "Royals": "Kansas City Royals", "Twins": "Minnesota Twins", "Astros": "Houston Astros",
            "Mariners": "Seattle Mariners", "Athletics": "Oakland Athletics", "Rangers": "Texas Rangers",
            "Braves": "Atlanta Braves", "Phillies": "Philadelphia Phillies", "Marlins": "Miami Marlins",
            "Nationals": "Washington Nationals", "Cubs": "Chicago Cubs", "Cardinals": "St. Louis Cardinals",
            "Pirates": "Pittsburgh Pirates", "Reds": "Cincinnati Reds", "Brewers": "Milwaukee Brewers",
            "Dodgers": "Los Angeles Dodgers", "Giants": "San Francisco Giants", "Padres": "San Diego Padres",
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

