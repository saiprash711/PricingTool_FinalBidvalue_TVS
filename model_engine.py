"""
model_engine.py
Dual-Engine Ensemble Valuation Platform for TVS Certified Tamil Nadu Auction Network.
Combines:
1. Econometric Hierarchical Mixed-Effects Model (MixedLM) with empirical Bayes shrinkage (BLUPs)
2. Gradient Boosted Decision Trees (CatBoost) capturing high-order non-linear interactions
Provides:
- Point estimates, prediction intervals (±1σ and 80% CI)
- Adaptive shrinkage weighting (CatBoost non-linear + MixedLM Bayes prior)
- Sub-model transparency (CatBoost vs MixedLM agreement)
- Multi-driver valuation waterfall with non-linear interaction synergy
- Empirical sensitivity curves across odometer and age
"""

import os
import io
import joblib
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

MODEL_FILE = "price_model.joblib"
CATBOOST_MODEL_FILE = "catboost_model.cbm"
DATA_FILE = "cleaned_tn_vehicle_data.csv"
FALLBACK_DATA_FILE = "cleaned_vehicle_data.csv"

def train_and_save_model(data_path=DATA_FILE, model_path=MODEL_FILE, cbm_path=CATBOOST_MODEL_FILE):
    """
    Trains the Dual-Engine Ensemble Valuation Model on Tamil Nadu auction records.
    Fits both Hierarchical MixedLM and CatBoostRegressor, performs honest 5-fold CV,
    and saves serialized payloads.
    """
    if not os.path.exists(data_path):
        if os.path.exists("masterdata_TN.xlsx"):
            from data_cleaner import clean_tn_master_data
            df = clean_tn_master_data("masterdata_TN.xlsx", data_path)
        elif os.path.exists(FALLBACK_DATA_FILE):
            df = pd.read_csv(FALLBACK_DATA_FILE, low_memory=False)
        else:
            from data_cleaner import clean_vehicle_data
            df = clean_vehicle_data()
    else:
        df = pd.read_csv(data_path, low_memory=False)

    # Filter to records with valid regression variables
    train_df = df.dropna(subset=['age', 'log_odo', 'FINAL BID VALUE']).copy()
    train_df = train_df[(train_df['FINAL BID VALUE'] > 5000) & (train_df['YEAR'] >= 1995)].copy()

    # Engineer indicator and categorical variables
    train_df['is_diesel'] = (train_df['FUEL TYPE'].astype(str).str.strip().str.title() == 'Diesel').astype(float)
    train_df['is_auto'] = (train_df['TRANSMISSION'].astype(str).str.strip().str.title() == 'Automatic').astype(float)
    train_df['is_owner2'] = (train_df['OWNER'].astype(str).str.startswith('2')).astype(float)
    train_df['is_owner3plus'] = (train_df['OWNER'].astype(str).str.startswith('3')).astype(float)
    
    valid_zones = ['Chennai Metro', 'Coimbatore Hub', 'Madurai / South TN', 'Rest of TN']
    zone_categories = [z for z in valid_zones if z in train_df['RTO_ZONE'].values]
    train_df['zone_cat'] = pd.Categorical(train_df['RTO_ZONE'].fillna('Chennai Metro'), categories=zone_categories)

    # Ensure clean categorical strings for CatBoost
    cat_features = ['MAKE', 'MODEL', 'VARIANT', 'make_model', 'FUEL TYPE', 'TRANSMISSION', 'OWNER', 'RTO_ZONE']
    if 'VARIANT' not in train_df.columns:
        train_df['VARIANT'] = 'Standard / Base'
    for col in cat_features:
        train_df[col] = train_df[col].fillna('Standard / Base' if col == 'VARIANT' else 'Unknown').astype(str).str.strip()

    cb_features = ['age', 'ODO METER', 'log_odo', 'MAKE', 'MODEL', 'VARIANT', 'make_model', 'FUEL TYPE', 'TRANSMISSION', 'OWNER', 'RTO_ZONE']
    cb_cat_cols = ['MAKE', 'MODEL', 'VARIANT', 'make_model', 'FUEL TYPE', 'TRANSMISSION', 'OWNER', 'RTO_ZONE']

    print(f"Training Dual-Engine Ensemble on {len(train_df)} verified records across {train_df['make_model'].nunique()} vehicle models...")

    # --- 1. FIT MIXEDLM ---
    formula = "log_price ~ age + log_odo + is_diesel + is_auto + is_owner2 + is_owner3plus + C(zone_cat)"
    md = smf.mixedlm(formula, train_df, groups=train_df["make_model"])
    mdf = md.fit()

    fixed_params = {
        'Intercept': float(mdf.params['Intercept']),
        'age': float(mdf.params['age']),
        'log_odo': float(mdf.params['log_odo']),
        'is_diesel': float(mdf.params.get('is_diesel', 0.059)),
        'is_auto': float(mdf.params.get('is_auto', 0.092)),
        'is_owner2': float(mdf.params.get('is_owner2', -0.072)),
        'is_owner3plus': float(mdf.params.get('is_owner3plus', -0.375))
    }

    # Extract Zone adjustments
    zone_effects = {'Chennai Metro': 0.0, 'Outside TN': -0.15}
    for z in ['Coimbatore Hub', 'Madurai / South TN', 'Rest of TN']:
        term = f"C(zone_cat)[T.{z}]"
        if term in mdf.params:
            zone_effects[z] = float(mdf.params[term])
        else:
            zone_effects[z] = 0.0

    state_effects = {'TN': 0.224, 'KL': 0.156, 'AP/TS': 0.0}
    overall_state_effect = 0.224

    sigma_eps = float(np.sqrt(mdf.scale))
    sigma_group = float(np.sqrt(mdf.cov_re.iloc[0, 0]))

    # Extract BLUPs (random effects) for each make_model group
    random_effects = {}
    for grp, effect_series in mdf.random_effects.items():
        random_effects[grp] = float(effect_series.iloc[0])

    # Brand-level fallback random effects
    brand_effects = {}
    brand_counts = train_df.groupby('MAKE').size().to_dict()
    model_counts = train_df.groupby('make_model').size().to_dict()

    for brand in train_df['MAKE'].unique():
        brand_models = train_df[train_df['MAKE'] == brand]['make_model'].unique()
        effects = [random_effects[m] for m in brand_models if m in random_effects]
        brand_effects[brand] = float(np.mean(effects)) if len(effects) > 0 else 0.0

    # Medians for baselines
    model_medians = {k: float(v) for k, v in df.groupby('make_model')['FINAL BID VALUE'].median().items()}
    brand_medians = {k: float(v) for k, v in df.groupby('MAKE')['FINAL BID VALUE'].median().items()}
    overall_median = float(df['FINAL BID VALUE'].median())

    # --- 2. FIT CATBOOST REGRESSOR ---
    print("Fitting CatBoost Regressor with categorical tree structures...")
    cb_full_pool = Pool(train_df[cb_features], train_df['log_price'], cat_features=cb_cat_cols)
    cb_model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.08,
        depth=6,
        loss_function='RMSE',
        verbose=0,
        random_seed=42
    )
    cb_model.fit(cb_full_pool)
    cb_model.save_model(cbm_path)

    # In-sample predictions
    actual_prices = train_df['FINAL BID VALUE'].values
    mixed_in_preds = np.exp(mdf.fittedvalues.values)
    cb_in_preds = np.exp(cb_model.predict(cb_full_pool))
    ens_in_preds = 0.65 * cb_in_preds + 0.35 * mixed_in_preds

    in_mape_mixed = float(np.mean(np.abs(actual_prices - mixed_in_preds) / actual_prices) * 100)
    in_mape_cb = float(np.mean(np.abs(actual_prices - cb_in_preds) / actual_prices) * 100)
    in_mape_ens = float(np.mean(np.abs(actual_prices - ens_in_preds) / actual_prices) * 100)

    # --- 3. 5-FOLD HONEST CROSS-VALIDATION ---
    print("Performing 5-fold cross-validation on MixedLM, CatBoost, and Ensemble...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_mixed_preds = np.zeros(len(train_df))
    cv_cb_preds = np.zeros(len(train_df))
    cv_ens_preds = np.zeros(len(train_df))

    for fold, (train_idx, test_idx) in enumerate(kf.split(train_df)):
        cv_tr = train_df.iloc[train_idx]
        cv_te = train_df.iloc[test_idx]

        # Fold MixedLM
        cv_md = smf.mixedlm(formula, cv_tr, groups=cv_tr["make_model"])
        cv_res = cv_md.fit(disp=False)
        cv_fe = cv_res.params
        cv_re = {g: float(s.iloc[0]) for g, s in cv_res.random_effects.items()}

        cv_brand_re = {}
        for b in cv_tr['MAKE'].unique():
            b_models = cv_tr[cv_tr['MAKE'] == b]['make_model'].unique()
            b_effs = [cv_re[m] for m in b_models if m in cv_re]
            cv_brand_re[b] = float(np.mean(b_effs)) if b_effs else 0.0

        fe_pred = (
            cv_fe['Intercept'] + 
            cv_fe['age'] * cv_te['age'] + 
            cv_fe['log_odo'] * cv_te['log_odo'] +
            cv_fe.get('is_diesel', 0.0) * cv_te['is_diesel'] +
            cv_fe.get('is_auto', 0.0) * cv_te['is_auto'] +
            cv_fe.get('is_owner2', 0.0) * cv_te['is_owner2'] +
            cv_fe.get('is_owner3plus', 0.0) * cv_te['is_owner3plus']
        )
        for z in ['Coimbatore Hub', 'Madurai / South TN', 'Rest of TN']:
            term = f"C(zone_cat)[T.{z}]"
            if term in cv_fe:
                fe_pred += (cv_te['zone_cat'] == z) * cv_fe[term]

        re_pred = []
        for _, r in cv_te.iterrows():
            mm = r['make_model']
            if mm in cv_re:
                re_pred.append(cv_re[mm])
            elif r['MAKE'] in cv_brand_re:
                re_pred.append(cv_brand_re[r['MAKE']])
            else:
                re_pred.append(0.0)

        fold_mixed_log = fe_pred.values + np.array(re_pred)
        fold_mixed_p = np.exp(fold_mixed_log)
        cv_mixed_preds[test_idx] = fold_mixed_p

        # Fold CatBoost
        fold_cb_train_pool = Pool(cv_tr[cb_features], cv_tr['log_price'], cat_features=cb_cat_cols)
        fold_cb_val_pool = Pool(cv_te[cb_features], cat_features=cb_cat_cols)

        fold_cb_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.08,
            depth=6,
            loss_function='RMSE',
            verbose=0,
            random_seed=42
        )
        fold_cb_model.fit(fold_cb_train_pool)
        fold_cb_log = fold_cb_model.predict(fold_cb_val_pool)
        fold_cb_p = np.exp(fold_cb_log)
        cv_cb_preds[test_idx] = fold_cb_p

        # Adaptive Ensemble Blend
        cv_ens_preds[test_idx] = 0.65 * fold_cb_p + 0.35 * fold_mixed_p

    # Compute Metrics
    mixed_mape = float(mean_absolute_percentage_error(actual_prices, cv_mixed_preds) * 100)
    mixed_mae = float(mean_absolute_error(actual_prices, cv_mixed_preds))
    mixed_r2 = float(r2_score(actual_prices, cv_mixed_preds))

    cb_mape = float(mean_absolute_percentage_error(actual_prices, cv_cb_preds) * 100)
    cb_mae = float(mean_absolute_error(actual_prices, cv_cb_preds))
    cb_r2 = float(r2_score(actual_prices, cv_cb_preds))

    ens_mape = float(mean_absolute_percentage_error(actual_prices, cv_ens_preds) * 100)
    ens_mae = float(mean_absolute_error(actual_prices, cv_ens_preds))
    ens_r2 = float(r2_score(actual_prices, cv_ens_preds))

    ens_log_resid = np.log(actual_prices) - np.log(cv_ens_preds)
    sigma_ensemble = float(np.std(ens_log_resid))

    metrics = {
        # Primary Ensemble CV Metrics
        'cv_mape_mean': ens_mape,
        'cv_mae_mean': ens_mae,
        'cv_r2_mean': ens_r2,
        'in_sample_mape': in_mape_ens,
        'sigma_eps': sigma_eps,
        'sigma_group': sigma_group,
        'sigma_ensemble': sigma_ensemble,
        
        # Sub-Model Comparison Metrics
        'mixedlm_cv_mape': mixed_mape,
        'mixedlm_cv_mae': mixed_mae,
        'mixedlm_cv_r2': mixed_r2,
        'catboost_cv_mape': cb_mape,
        'catboost_cv_mae': cb_mae,
        'catboost_cv_r2': cb_r2,
        'ensemble_cv_mape': ens_mape,
        'ensemble_cv_mae': ens_mae,
        'ensemble_cv_r2': ens_r2,
        'ensemble_mae_saving_vs_mixedlm': mixed_mae - ens_mae,
        'ensemble_mape_gain_vs_mixedlm': mixed_mape - ens_mape,

        'total_records': len(df),
        'trained_records': len(train_df),
        'unique_models': len(random_effects),
        'unique_makes': len(brand_effects),
        'model_architecture': 'Dual-Engine Ensemble (Hierarchical MixedLM + CatBoost Regressor)'
    }

    # Package model payload
    model_payload = {
        'fixed_params': fixed_params,
        'zone_effects': zone_effects,
        'state_effects': state_effects,
        'overall_state_effect': overall_state_effect,
        'random_effects': random_effects,
        'brand_effects': brand_effects,
        'model_counts': model_counts,
        'brand_counts': brand_counts,
        'model_medians': model_medians,
        'brand_medians': brand_medians,
        'overall_median': overall_median,
        'sigma_eps': sigma_eps,
        'sigma_group': sigma_group,
        'sigma_ensemble': sigma_ensemble,
        'metrics': metrics,
        'cb_model_path': cbm_path
    }

    joblib.dump(model_payload, model_path)
    print("Dual-Engine Ensemble Model saved successfully.")
    print(f"Metrics: MixedLM CV MAPE: {mixed_mape:.2f}% | CatBoost CV MAPE: {cb_mape:.2f}% | Ensemble CV MAPE: {ens_mape:.2f}%")
    print(f"Ensemble CV MAE: Rs. {ens_mae:,.0f} (Saved Rs. {mixed_mae - ens_mae:,.0f} vs MixedLM alone)")
    return model_payload

class VehiclePricePredictor:
    def __init__(self, model_path=MODEL_FILE, data_path=DATA_FILE, cbm_path=CATBOOST_MODEL_FILE):
        self.model_path = model_path
        self.data_path = data_path
        self.cbm_path = cbm_path

        if not os.path.exists(model_path):
            self.payload = train_and_save_model(data_path=data_path, model_path=model_path, cbm_path=cbm_path)
        else:
            self.payload = joblib.load(model_path)

        self.fixed_params = self.payload['fixed_params']
        self.zone_effects = self.payload.get('zone_effects', {
            'Chennai Metro': 0.0, 'Coimbatore Hub': 0.052, 'Madurai / South TN': -0.058, 'Rest of TN': -0.023, 'Outside TN': -0.278
        })
        self.state_effects = self.payload.get('state_effects', {'AP/TS': 0.0, 'KL': 0.156, 'TN': 0.224})
        self.overall_state_effect = self.payload.get('overall_state_effect', 0.224)
        self.random_effects = self.payload['random_effects']
        self.brand_effects = self.payload['brand_effects']
        self.model_counts = self.payload['model_counts']
        self.brand_counts = self.payload['brand_counts']
        self.model_medians = self.payload.get('model_medians', {})
        self.brand_medians = self.payload.get('brand_medians', {})
        self.overall_median = self.payload.get('overall_median', 320000.0)
        self.sigma_eps = self.payload['sigma_eps']
        self.sigma_group = self.payload['sigma_group']
        self.sigma_ensemble = self.payload.get('sigma_ensemble', self.sigma_eps * 0.92)
        self.metrics = self.payload['metrics']

        # Load CatBoost Model
        self.cb_model = None
        target_cbm = self.payload.get('cb_model_path', cbm_path)
        if not os.path.exists(target_cbm):
            target_cbm = os.path.join(os.path.dirname(model_path), CATBOOST_MODEL_FILE)
            
        if os.path.exists(target_cbm):
            try:
                self.cb_model = CatBoostRegressor()
                self.cb_model.load_model(target_cbm)
            except Exception as e:
                print(f"Warning: Could not load CatBoost model from {target_cbm}: {e}")
                self.cb_model = None

        # Load dataset for comps and market baseline lookup
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path, low_memory=False)
        elif os.path.exists(FALLBACK_DATA_FILE):
            self.df = pd.read_csv(FALLBACK_DATA_FILE, low_memory=False)
        else:
            self.df = None

    def update_dataset(self, new_df):
        """Updates the active dataset with new or uploaded data and refreshes baseline metrics."""
        if new_df is not None and len(new_df) > 0:
            self.df = new_df.copy()
            self.model_counts = self.df.groupby('make_model').size().to_dict()
            self.brand_counts = self.df.groupby('MAKE').size().to_dict()
            self.model_medians = {k: float(v) for k, v in self.df.groupby('make_model')['FINAL BID VALUE'].median().items()}
            self.brand_medians = {k: float(v) for k, v in self.df.groupby('MAKE')['FINAL BID VALUE'].median().items()}
            self.overall_median = float(self.df['FINAL BID VALUE'].median())

    def get_catalog(self, branch=None):
        """Returns hierarchy of available makes and models in the dataset, optionally filtered by branch or zone."""
        if self.df is not None and len(self.df) > 0:
            df_target = self.df
            if branch and branch not in ['All', 'All Branches', 'All Branches (Combined)', 'All Tamil Nadu']:
                branch_clean = str(branch).strip()
                cond = False
                if 'RTO_ZONE' in df_target.columns:
                    cond = cond | (df_target['RTO_ZONE'] == branch_clean)
                if 'BRANCH' in df_target.columns:
                    cond = cond | (df_target['BRANCH'] == branch_clean)
                if 'STATE' in df_target.columns:
                    cond = cond | (df_target['STATE'] == branch_clean)
                branch_matches = df_target[cond]
                if len(branch_matches) > 0:
                    df_target = branch_matches

            catalog = {}
            for make, group in df_target.groupby('MAKE'):
                make_str = str(make).strip()
                if make_str and make_str.lower() != 'nan':
                    models = sorted([str(m).strip() for m in group['MODEL'].dropna().unique() if str(m).strip() and str(m).strip().lower() != 'nan'])
                    if models:
                        catalog[make_str] = models
            return catalog
        return {}

    def get_variants(self, make=None, model=None):
        """Returns sorted list of distinct variants/trims for a given make & model."""
        if self.df is None or len(self.df) == 0 or 'VARIANT' not in self.df.columns:
            return []
        
        df_target = self.df
        if make and str(make).strip().lower() not in ['all', 'all makes', 'none', '']:
            df_target = df_target[df_target['MAKE'].str.strip().str.lower() == str(make).strip().lower()]
        if model and str(model).strip().lower() not in ['all', 'all models', 'none', '']:
            df_target = df_target[df_target['MODEL'].str.strip().str.lower() == str(model).strip().lower()]
        
        if len(df_target) == 0:
            return []

        counts = df_target['VARIANT'].value_counts()
        variants = [
            str(v).strip() for v in counts.index 
            if str(v).strip() and str(v).strip().lower() not in ['nan', 'none', 'unknown', '']
        ]
        return variants

    def predict(
        self, 
        make, 
        model, 
        year, 
        odometer, 
        fuel='Petrol', 
        transmission='Manual', 
        owner='1st Owner', 
        ownership=None,
        rto_zone='Chennai Metro', 
        reference_year=2026, 
        branch=None,
        variant=None,
        **kwargs
    ):
        """
        Predicts vehicle auction final bid value using the Dual-Engine Ensemble
        (MixedLM Empirical Bayes BLUPs + CatBoost Non-Linear Regressor).
        """
        if ownership is not None:
            owner = ownership
        age = max(0.5, reference_year - int(year))
        odo = max(500.0, float(odometer))
        log_odo = np.log(odo)

        make_clean = str(make).strip()
        model_clean = str(model).strip()
        mm_key = f"{make_clean} | {model_clean}"

        # Standardize variant
        variant_clean = str(variant).strip() if variant is not None else ""
        if not variant_clean or variant_clean.lower() in ['all', 'all variants', 'all / any variant', 'none', 'nan']:
            available = self.get_variants(make_clean, model_clean)
            variant_clean = available[0] if available else 'Standard / Base'

        # Standardize categorical strings
        fuel_str = str(fuel).strip().title()
        if 'Diesel' in fuel_str:
            fuel_clean = 'Diesel'
        elif 'Cng' in fuel_str or 'Lpg' in fuel_str:
            fuel_clean = 'CNG/LPG'
        elif 'Hybrid' in fuel_str or 'Ev' in fuel_str:
            fuel_clean = 'Hybrid/EV'
        else:
            fuel_clean = 'Petrol'

        trans_str = str(transmission).strip().title()
        trans_clean = 'Automatic' if 'Auto' in trans_str else 'Manual'

        owner_str = str(owner).strip()
        if owner_str.startswith('2'):
            owner_clean = '2nd Owner'
        elif owner_str.startswith('3'):
            owner_clean = '3+ Owners'
        else:
            owner_clean = '1st Owner'

        # Regional zone calibration
        zone_clean = str(rto_zone).strip()
        state_delta = 0.0
        if branch and branch not in ['All', 'All Branches', 'All Branches (Combined)', 'All Tamil Nadu']:
            b_str = str(branch).strip()
            if b_str in self.zone_effects:
                zone_clean = b_str
            elif b_str == 'TN':
                zone_clean = 'Chennai Metro'
                state_delta = 0.0
            elif b_str in self.state_effects:
                state_delta = self.state_effects[b_str] - self.overall_state_effect
        zone_delta = self.zone_effects.get(zone_clean, 0.0)

        # --- SUB-MODEL 1: MIXED-EFFECTS LINEAR MODEL ---
        b0 = self.fixed_params['Intercept']
        b_age = self.fixed_params['age']
        b_odo = self.fixed_params['log_odo']
        
        is_diesel = 1.0 if fuel_clean == 'Diesel' else 0.0
        is_auto = 1.0 if trans_clean == 'Automatic' else 0.0
        is_owner2 = 1.0 if owner_clean == '2nd Owner' else 0.0
        is_owner3plus = 1.0 if owner_clean == '3+ Owners' else 0.0

        fe_log = (
            b0 + 
            b_age * age + 
            b_odo * log_odo +
            self.fixed_params.get('is_diesel', 0.059) * is_diesel +
            self.fixed_params.get('is_auto', 0.092) * is_auto +
            self.fixed_params.get('is_owner2', -0.072) * is_owner2 +
            self.fixed_params.get('is_owner3plus', -0.375) * is_owner3plus
        )

        is_known_model = mm_key in self.random_effects
        is_known_brand = make_clean in self.brand_effects

        # Sample counts & regional scope
        total_samples = 0
        zone_samples = 0
        is_all_branches = (
            not branch or 
            str(branch).strip() in ['All', 'All Branches', 'All Branches (Combined)', 'All Tamil Nadu']
        )
        is_specific_variant = (
            variant is not None and 
            str(variant).strip().lower() not in ['all', 'all variants', 'all / any variant', 'none', 'nan', '']
        )

        model_total_samples = 0
        if self.df is not None and len(self.df) > 0:
            exact_matches = self.df[(self.df['MAKE'].str.strip().str.lower() == make_clean.lower()) & 
                                    (self.df['MODEL'].str.strip().str.lower() == model_clean.lower())]
            model_total_samples = len(exact_matches)

            # If user selected a specific variant, filter to that variant
            if is_specific_variant and 'VARIANT' in exact_matches.columns:
                v_target = str(variant).strip().upper()
                v_matches = exact_matches[exact_matches['VARIANT'].str.strip().str.upper() == v_target]
                if len(v_matches) > 0:
                    matches_for_count = v_matches
                else:
                    matches_for_count = exact_matches
            else:
                matches_for_count = exact_matches

            total_samples = len(matches_for_count)
            if 'RTO_ZONE' in matches_for_count.columns:
                zone_matches = matches_for_count[matches_for_count['RTO_ZONE'] == zone_clean]
                zone_samples = len(zone_matches)
            else:
                zone_samples = total_samples
        else:
            total_samples = self.model_counts.get(mm_key, 0)
            zone_samples = total_samples
            model_total_samples = total_samples

        if is_known_model:
            re = self.random_effects[mm_key]
            if is_all_branches:
                n_samples = total_samples
            else:
                n_samples = zone_samples if zone_samples > 0 else total_samples

            confidence_level = "High" if n_samples >= 10 else ("Medium" if n_samples >= 3 else ("Low" if n_samples > 0 else "Low (Prior)"))
            
            trim_desc = f" {variant_clean}" if is_specific_variant else ""
            zone_desc = f"in {zone_clean}" if not is_all_branches else "across Tamil Nadu"
            source_info = f"Ensemble calibrated from {total_samples} auction records for {make_clean} {model_clean}{trim_desc} ({n_samples} {zone_desc})."
        elif is_known_brand:
            re = self.brand_effects[make_clean]
            n_samples = total_samples
            confidence_level = "Brand Prior"
            source_info = f"Unseen model under known make '{make_clean}'. Shrunk using empirical brand prior based on {self.brand_counts.get(make_clean, 0)} records."
        else:
            re = 0.0
            n_samples = 0
            confidence_level = "Market Prior"
            source_info = "Unseen brand and model. Evaluated using overall market baseline."

        pred_log_mixed = fe_log + re + zone_delta + state_delta
        mixed_expected_price = float(np.exp(pred_log_mixed))

        # --- SUB-MODEL 2: CATBOOST REGRESSOR ---
        if self.cb_model is not None:
            cb_input = pd.DataFrame([{
                'age': float(age),
                'ODO METER': float(odo),
                'log_odo': float(log_odo),
                'MAKE': str(make_clean),
                'MODEL': str(model_clean),
                'VARIANT': str(variant_clean),
                'make_model': str(mm_key),
                'FUEL TYPE': str(fuel_clean),
                'TRANSMISSION': str(trans_clean),
                'OWNER': str(owner_clean),
                'RTO_ZONE': str(zone_clean)
            }])
            cb_log_pred = float(self.cb_model.predict(cb_input)[0])
            cb_expected_price = float(np.exp(cb_log_pred + state_delta))

            # --- ADAPTIVE ENSEMBLE WEIGHTING ---
            # High sample models leverage CatBoost's rich interaction modeling
            # Low sample or unseen models adaptively rely on MixedLM's empirical Bayes shrinkage
            if is_known_model:
                if n_samples >= 10:
                    w_cb = 0.65
                elif n_samples >= 3:
                    w_cb = 0.55
                else:
                    w_cb = 0.45
            elif is_known_brand:
                w_cb = 0.30
            else:
                w_cb = 0.15

            w_mixed = 1.0 - w_cb
            pred_log_ensemble = w_cb * (cb_log_pred + state_delta) + w_mixed * pred_log_mixed
            expected_price = float(np.exp(pred_log_ensemble))
        else:
            cb_expected_price = mixed_expected_price
            w_cb = 0.0
            w_mixed = 1.0
            pred_log_ensemble = pred_log_mixed
            expected_price = mixed_expected_price

        # Estimated seller net payout and seller reserve
        est_net_payout = max(10000.0, expected_price - 10000.0)
        est_seller_reserve = expected_price * 1.15

        # Compute total prediction uncertainty
        base_sigma = self.sigma_ensemble if self.cb_model is not None else self.sigma_eps
        if is_known_model:
            shrinkage_weight = 1.0 - (n_samples / (n_samples + 3.0))
            sigma_pred = np.sqrt(base_sigma**2 + (self.sigma_group * shrinkage_weight)**2)
        elif is_known_brand:
            sigma_pred = np.sqrt(base_sigma**2 + (self.sigma_group * 0.7)**2)
        else:
            sigma_pred = np.sqrt(base_sigma**2 + self.sigma_group**2)

        range_low_1sigma = float(np.exp(pred_log_ensemble - sigma_pred))
        range_high_1sigma = float(np.exp(pred_log_ensemble + sigma_pred))
        ci_80_low = float(np.exp(pred_log_ensemble - 1.282 * sigma_pred))
        ci_80_high = float(np.exp(pred_log_ensemble + 1.282 * sigma_pred))

        # Driver Multipliers
        base_market_price = float(np.exp(b0))
        age_multiplier = float(np.exp(b_age * age))
        annual_dep_pct = float((1.0 - np.exp(b_age)) * 100)
        model_multiplier = float(np.exp(re))
        
        fuel_delta = self.fixed_params.get('is_diesel', 0.059) * is_diesel
        trans_delta = self.fixed_params.get('is_auto', 0.092) * is_auto
        owner_delta = self.fixed_params.get('is_owner2', -0.072) * is_owner2 + self.fixed_params.get('is_owner3plus', -0.375) * is_owner3plus

        trans_multiplier = float(np.exp(trans_delta))
        owner_multiplier = float(np.exp(owner_delta))
        zone_multiplier = float(np.exp(zone_delta + state_delta))
        fuel_multiplier = float(np.exp(fuel_delta))

        ref_odo = 60000.0
        odo_vs_ref = float(np.exp(b_odo * (log_odo - np.log(ref_odo))))
        mileage_adj_pct = float((odo_vs_ref - 1.0) * 100)

        # Comparable Market Baseline
        if is_specific_variant and self.df is not None and len(self.df) > 0:
            v_target = str(variant).strip().upper()
            exact_v = self.df[(self.df['MAKE'].str.strip().str.lower() == make_clean.lower()) & 
                              (self.df['MODEL'].str.strip().str.lower() == model_clean.lower()) & 
                              (self.df['VARIANT'].str.strip().str.upper() == v_target)]
            if len(exact_v) >= 2 and 'FINAL BID VALUE' in exact_v.columns:
                comp_price = float(exact_v['FINAL BID VALUE'].median())
                comp_label = f"Historical median for {make_clean} {model_clean} {v_target}"
            else:
                comp_price = self.model_medians.get(mm_key, self.brand_medians.get(make_clean, self.overall_median))
                comp_label = f"Historical median for {make_clean} {model_clean}"
        else:
            comp_price = self.model_medians.get(mm_key, self.brand_medians.get(make_clean, self.overall_median))
            comp_label = f"Historical median for {make_clean} {model_clean}"

        age_dep_pct = float((1.0 - age_multiplier) * 100)
        model_prem_pct = float((model_multiplier - 1.0) * 100)
        zone_prem_pct = float((zone_multiplier - 1.0) * 100)
        trans_prem_pct = float((trans_multiplier - 1.0) * 100)
        owner_pen_pct = float((owner_multiplier - 1.0) * 100)

        # Model agreement / divergence
        model_spread_pct = float(abs(cb_expected_price - mixed_expected_price) / max(1.0, mixed_expected_price) * 100)
        agreement_level = "High Consensus" if model_spread_pct < 8.0 else ("Moderate Spread" if model_spread_pct < 18.0 else "Non-Linear Divergence")

        return {
            'expected_price': expected_price,
            'est_net_payout': est_net_payout,
            'seller_net_payout': est_net_payout,
            'est_seller_reserve': est_seller_reserve,
            'seller_reserve_target': est_seller_reserve,
            'range_low': range_low_1sigma,
            'range_high': range_high_1sigma,
            'range_low_1sigma': range_low_1sigma,
            'range_high_1sigma': range_high_1sigma,
            'sigma_pred': float(sigma_pred),
            'ci_80_low': ci_80_low,
            'ci_80_high': ci_80_high,
            'confidence_level': confidence_level,
            'n_samples': n_samples,
            'total_samples': total_samples,
            'zone_samples': zone_samples,
            'source_info': source_info,
            'age': age,
            'odometer': odo,
            'fuel': fuel_clean,
            'transmission': trans_clean,
            'owner': owner_clean,
            'rto_zone': zone_clean,
            'variant': variant_clean,
            
            # Dual-engine ensemble diagnostics
            'catboost_price': cb_expected_price,
            'mixedlm_price': mixed_expected_price,
            'catboost_weight': w_cb,
            'mixedlm_weight': w_mixed,
            'model_spread_pct': model_spread_pct,
            'agreement_level': agreement_level,
            'model_architecture': 'MixedLM + CatBoost Dual-Engine Ensemble',

            'breakdown': {
                'base_market_price': base_market_price,
                'age_multiplier': age_multiplier,
                'annual_dep_pct': annual_dep_pct,
                'annual_depreciation_pct': annual_dep_pct,
                'age_depreciation_pct': age_dep_pct,
                'odo_multiplier': odo_vs_ref,
                'mileage_adj_pct': mileage_adj_pct,
                'mileage_ref_km': int(ref_odo),
                'model_multiplier': model_multiplier,
                'model_premium_pct': model_prem_pct,
                'trans_multiplier': trans_multiplier,
                'trans_premium_pct': trans_prem_pct,
                'transmission_adj_pct': trans_prem_pct,
                'owner_multiplier': owner_multiplier,
                'owner_penalty_pct': owner_pen_pct,
                'ownership_adj_pct': owner_pen_pct,
                'zone_multiplier': zone_multiplier,
                'zone_premium_pct': zone_prem_pct,
                'zone_adj_pct': zone_prem_pct,
                'regional_calibration_pct': zone_prem_pct,
                'regional_calibration_label': f"RTO Zone ({zone_clean})",
                'fuel_multiplier': fuel_multiplier,
                'comparable_median': comp_price,
                'comparable_baseline_price': comp_price,
                'comparable_label': comp_label,
                'comparable_baseline_label': comp_label,
                'catboost_price': cb_expected_price,
                'mixedlm_price': mixed_expected_price,
                'catboost_weight': w_cb,
                'mixedlm_weight': w_mixed,
                'ensemble_synergy_pct': float(((expected_price / max(1.0, mixed_expected_price)) - 1.0) * 100)
            }
        }

    def get_valuation_waterfall(self, prediction):
        """Builds structured data for waterfall decomposition chart."""
        bd = prediction['breakdown']
        p = prediction['expected_price']
        
        base_val = bd['comparable_median']
        age_effect = base_val * (bd['age_multiplier'] - 1.0)
        odo_effect = base_val * (bd['odo_multiplier'] - 1.0)
        model_effect = base_val * (bd['model_multiplier'] - 1.0)
        trans_effect = base_val * (bd['trans_multiplier'] - 1.0)
        owner_effect = base_val * (bd['owner_multiplier'] - 1.0)
        zone_effect = base_val * (bd['zone_multiplier'] - 1.0)

        linear_sum = base_val + age_effect + odo_effect + model_effect + trans_effect + owner_effect + zone_effect
        ensemble_synergy = p - linear_sum

        steps = [
            {"name": "Market Baseline", "value": base_val, "type": "baseline"},
            {"name": f"Age ({prediction['age']:.1f}y)", "value": age_effect, "type": "relative"},
            {"name": f"Mileage ({prediction['odometer']:,}km)", "value": odo_effect, "type": "relative"},
            {"name": "Model Effect (BLUP)", "value": model_effect, "type": "relative"},
            {"name": f"Trans ({prediction['transmission']})", "value": trans_effect, "type": "relative"},
            {"name": f"Ownership ({prediction['owner']})", "value": owner_effect, "type": "relative"},
            {"name": f"Zone ({prediction['rto_zone']})", "value": zone_effect, "type": "relative"}
        ]
        
        if abs(ensemble_synergy) >= 500:
            steps.append({"name": "Ensemble Synergy", "value": ensemble_synergy, "type": "relative"})

        steps.append({"name": "Final Ensemble Bid", "value": p, "type": "total"})
        return steps

    def get_sensitivity_curve(
        self, 
        make, 
        model, 
        year, 
        odometer=60000, 
        curve_type=None, 
        fuel='Petrol', 
        transmission='Manual', 
        owner='1st Owner', 
        ownership=None, 
        rto_zone='Chennai Metro', 
        variant=None,
        **kwargs
    ):
        """Generates age and mileage depreciation sensitivity curves using the ensemble."""
        if ownership is not None:
            owner = ownership
        odo_val = 60000.0 if str(odometer).lower() in ['odometer', 'odo', 'km', 'age', 'year'] else float(odometer)
        odo_points = np.linspace(10000, 180000, 25)
        odo_predictions = [
            self.predict(make, model, year, o, fuel=fuel, transmission=transmission, owner=owner, rto_zone=rto_zone, variant=variant)['expected_price']
            for o in odo_points
        ]

        current_year = 2026
        years = list(range(2012, current_year + 1))
        age_predictions = [
            self.predict(make, model, y, odo_val, fuel=fuel, transmission=transmission, owner=owner, rto_zone=rto_zone, variant=variant)['expected_price']
            for y in years
        ]

        res = {
            'odometer_curve': {'odometers': odo_points.tolist(), 'prices': odo_predictions},
            'age_curve': {'years': years, 'prices': age_predictions}
        }
        if str(odometer).lower() in ['odometer', 'odo', 'km'] or str(curve_type).lower() in ['odometer', 'odo', 'km']:
            return res['odometer_curve']
        if str(odometer).lower() in ['age', 'year'] or str(curve_type).lower() in ['age', 'year']:
            return res['age_curve']
        return res

    def get_comps(self, make, model, rto_zone=None, branch=None, variant=None, limit=None, **kwargs):
        """Returns comparable historical auction records for a given make, model, and optional variant/branch."""
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()

        cond = (self.df['MAKE'].str.strip().str.lower() == str(make).strip().lower()) & \
               (self.df['MODEL'].str.strip().str.lower() == str(model).strip().lower())

        matches = self.df[cond].copy()

        # Filter by variant if provided
        if variant and str(variant).strip().lower() not in ['all', 'all variants', 'all / any variant', 'none', 'nan', '']:
            v_clean = str(variant).strip().upper()
            if 'VARIANT' in matches.columns:
                exact_variant_matches = matches[matches['VARIANT'].str.strip().str.upper() == v_clean]
                if len(exact_variant_matches) > 0:
                    matches = exact_variant_matches

        target_filter = rto_zone or branch
        if target_filter and target_filter not in ['All', 'All Tamil Nadu', 'All Branches', 'All Branches (Combined)']:
            if 'RTO_ZONE' in matches.columns and target_filter in matches['RTO_ZONE'].values:
                matches = matches[matches['RTO_ZONE'] == target_filter]
            elif 'BRANCH' in matches.columns and target_filter in matches['BRANCH'].values:
                matches = matches[matches['BRANCH'] == target_filter]
            elif 'STATE' in matches.columns and target_filter in matches['STATE'].values:
                matches = matches[matches['STATE'] == target_filter]

        if 'EXPECTED PRICE' in matches.columns and 'SELLER RESERVE' not in matches.columns:
            matches['SELLER RESERVE'] = matches['EXPECTED PRICE']
        if 'SELLER RESERVE' in matches.columns and 'EXPECTED PRICE' not in matches.columns:
            matches['EXPECTED PRICE'] = matches['SELLER RESERVE']

        if 'BRANCH' not in matches.columns:
            if 'RTO_ZONE' in matches.columns:
                matches['BRANCH'] = matches['RTO_ZONE']
            elif 'STATE' in matches.columns:
                matches['BRANCH'] = matches['STATE']
            else:
                matches['BRANCH'] = 'TN'

        sort_col = 'YEAR' if 'YEAR' in matches.columns else matches.columns[0]
        sorted_matches = matches.sort_values(by=sort_col, ascending=False)
        if limit is not None:
            return sorted_matches.head(limit)
        return sorted_matches

    get_historical_comps = get_comps

if __name__ == "__main__":
    print("Training and validating upgraded Dual-Engine Ensemble Engine (MixedLM + CatBoost)...")
    payload = train_and_save_model()
    predictor = VehiclePricePredictor()
    
    # Test sample predictions
    sample_tests = [
        {"make": "Maruti", "model": "Swift", "year": 2019, "odo": 45000, "trans": "Manual", "owner": "1st Owner", "zone": "Chennai Metro"},
        {"make": "Maruti", "model": "Swift", "year": 2019, "odo": 45000, "trans": "Automatic", "owner": "1st Owner", "zone": "Chennai Metro"},
        {"make": "Maruti", "model": "Swift", "year": 2019, "odo": 45000, "trans": "Automatic", "owner": "3+ Owners", "zone": "Chennai Metro"},
        {"make": "Maruti", "model": "Swift", "year": 2019, "odo": 45000, "trans": "Automatic", "owner": "1st Owner", "zone": "Coimbatore Hub"},
        {"make": "Toyota", "model": "Innova", "year": 2018, "odo": 95000, "trans": "Manual", "owner": "1st Owner", "zone": "Chennai Metro"},
    ]
    
    print("\n--- SAMPLE ENSEMBLE PREDICTIONS ---")
    for st in sample_tests:
        res = predictor.predict(
            st['make'], st['model'], st['year'], st['odo'], 
            transmission=st['trans'], owner=st['owner'], rto_zone=st['zone']
        )
        print(f"{st['make']} {st['model']} ({st['year']}, {st['odo']}km, {st['trans']}, {st['owner']}, {st['zone']}):")
        print(f"  Ensemble Forecast: Rs. {res['expected_price']:,.0f} | 1-Sigma: Rs. {res['range_low']:,.0f} - {res['range_high']:,.0f}")
        print(f"  CatBoost: Rs. {res['catboost_price']:,.0f} ({res['catboost_weight']*100:.0f}%) | MixedLM: Rs. {res['mixedlm_price']:,.0f} ({res['mixedlm_weight']*100:.0f}%) | Consensus: {res['agreement_level']}")
