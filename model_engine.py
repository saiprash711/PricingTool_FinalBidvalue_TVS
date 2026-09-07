"""
model_engine.py
Hierarchical Mixed-Effects Model for Vehicle Price Forecasting in AP/Telangana.
Implements empirical Bayes shrinkage on log(price) with age, log(odometer), and make/model random effects.
Provides point estimates, prediction intervals, and valuation factor decomposition.
"""

import os
import joblib
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

MODEL_FILE = "price_model.joblib"
DATA_FILE = "cleaned_vehicle_data.csv"

def train_and_save_model(data_path=DATA_FILE, model_path=MODEL_FILE):
    if not os.path.exists(data_path):
        from data_cleaner import clean_vehicle_data
        df = clean_vehicle_data()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    print(f"Training model on {len(df)} records across {df['make_model'].nunique()} vehicle models...")
    
    # Fit MixedLM: log_price ~ age + log_odo with random intercept per make_model
    md = smf.mixedlm("log_price ~ age + log_odo", df, groups=df["make_model"])
    mdf = md.fit()
    
    fixed_params = {
        'Intercept': float(mdf.params['Intercept']),
        'age': float(mdf.params['age']),
        'log_odo': float(mdf.params['log_odo'])
    }
    
    sigma_eps = float(np.sqrt(mdf.scale))
    sigma_group = float(np.sqrt(mdf.cov_re.iloc[0, 0]))
    
    # Extract BLUPs (random effects) for each make_model group
    random_effects = {}
    for grp, effect_series in mdf.random_effects.items():
        random_effects[grp] = float(effect_series.iloc[0])
        
    # Compute brand-level fallback random effects
    brand_effects = {}
    brand_counts = df.groupby('MAKE').size().to_dict()
    model_counts = df.groupby('make_model').size().to_dict()
    
    for brand in df['MAKE'].unique():
        brand_models = df[df['MAKE'] == brand]['make_model'].unique()
        effects = [random_effects[m] for m in brand_models if m in random_effects]
        brand_effects[brand] = float(np.mean(effects)) if len(effects) > 0 else 0.0

    # Precompute medians for intuitive comparable baselines
    model_medians = {k: float(v) for k, v in df.groupby('make_model')['FINAL BID VALUE'].median().items()}
    brand_medians = {k: float(v) for k, v in df.groupby('MAKE')['FINAL BID VALUE'].median().items()}
    overall_median = float(df['FINAL BID VALUE'].median())

    # Calculate in-sample metrics
    fitted_prices = []
    actual_prices = df['FINAL BID VALUE'].values
    for _, row in df.iterrows():
        fe = fixed_params['Intercept'] + fixed_params['age'] * row['age'] + fixed_params['log_odo'] * row['log_odo']
        re = random_effects.get(row['make_model'], 0.0)
        fitted_prices.append(np.exp(fe + re))
        
    fitted_prices = np.array(fitted_prices)
    in_mape = float(np.mean(np.abs(actual_prices - fitted_prices) / actual_prices) * 100)
    in_mae = float(np.mean(np.abs(actual_prices - fitted_prices)))
    in_r2 = float(r2_score(actual_prices, fitted_prices))

    # 5-fold honest Cross-Validation for out-of-sample metrics
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mapes, cv_maes, cv_r2s = [], [], []
    
    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        
        cv_md = smf.mixedlm("log_price ~ age + log_odo", train_df, groups=train_df["make_model"])
        cv_res = cv_md.fit()
        cv_fe = cv_res.params
        cv_re = {g: float(s.iloc[0]) for g, s in cv_res.random_effects.items()}
        
        cv_brand_re = {}
        for b in train_df['MAKE'].unique():
            b_models = train_df[train_df['MAKE'] == b]['make_model'].unique()
            b_effs = [cv_re[m] for m in b_models if m in cv_re]
            cv_brand_re[b] = float(np.mean(b_effs)) if b_effs else 0.0
            
        fold_preds = []
        for _, r in test_df.iterrows():
            f_val = cv_fe['Intercept'] + cv_fe['age'] * r['age'] + cv_fe['log_odo'] * r['log_odo']
            mm = r['make_model']
            if mm in cv_re:
                r_val = cv_re[mm]
            elif r['MAKE'] in cv_brand_re:
                r_val = cv_brand_re[r['MAKE']]
            else:
                r_val = 0.0
            fold_preds.append(np.exp(f_val + r_val))
            
        fold_preds = np.array(fold_preds)
        y_test = test_df['FINAL BID VALUE'].values
        cv_mapes.append(mean_absolute_percentage_error(y_test, fold_preds) * 100)
        cv_maes.append(mean_absolute_error(y_test, fold_preds))
        cv_r2s.append(r2_score(y_test, fold_preds))
        
    metrics = {
        'in_sample_mape': in_mape,
        'in_sample_mae': in_mae,
        'in_sample_r2': in_r2,
        'cv_mape_mean': float(np.mean(cv_mapes)),
        'cv_mape_std': float(np.std(cv_mapes)),
        'cv_mae_mean': float(np.mean(cv_maes)),
        'cv_r2_mean': float(np.mean(cv_r2s)),
        'sigma_eps': sigma_eps,
        'sigma_group': sigma_group,
        'total_records': len(df),
        'unique_models': len(random_effects),
        'unique_makes': len(brand_effects)
    }

    # Package model payload
    model_payload = {
        'fixed_params': fixed_params,
        'random_effects': random_effects,
        'brand_effects': brand_effects,
        'model_counts': model_counts,
        'brand_counts': brand_counts,
        'model_medians': model_medians,
        'brand_medians': brand_medians,
        'overall_median': overall_median,
        'sigma_eps': sigma_eps,
        'sigma_group': sigma_group,
        'metrics': metrics
    }
    
    joblib.dump(model_payload, model_path)
    print("Model saved successfully.")
    print(f"Metrics: In-sample MAPE: {in_mape:.2f}% | CV MAPE: {metrics['cv_mape_mean']:.2f}% | CV MAE: Rs. {metrics['cv_mae_mean']:,.0f}")
    return model_payload

class VehiclePricePredictor:
    def __init__(self, model_path=MODEL_FILE, data_path=DATA_FILE):
        if not os.path.exists(model_path):
            self.payload = train_and_save_model(data_path=data_path, model_path=model_path)
        else:
            self.payload = joblib.load(model_path)
            
        self.fixed_params = self.payload['fixed_params']
        self.random_effects = self.payload['random_effects']
        self.brand_effects = self.payload['brand_effects']
        self.model_counts = self.payload['model_counts']
        self.brand_counts = self.payload['brand_counts']
        self.model_medians = self.payload.get('model_medians', {})
        self.brand_medians = self.payload.get('brand_medians', {})
        self.overall_median = self.payload.get('overall_median', 187000.0)
        self.sigma_eps = self.payload['sigma_eps']
        self.sigma_group = self.payload['sigma_group']
        self.metrics = self.payload['metrics']
        
        # Load dataset for comps
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
        else:
            self.df = None

    def get_catalog(self):
        """Returns hierarchy of available makes and models in the training set."""
        if self.df is not None:
            catalog = {}
            for make, group in self.df.groupby('MAKE'):
                catalog[make] = sorted(group['MODEL'].unique().tolist())
            return catalog
        return {}

    def predict(self, make, model, year, odometer, reference_year=2026):
        """
        Predict vehicle final bid value with uncertainty bands and valuation breakdown.
        """
        age = max(0.5, reference_year - int(year))
        odo = max(500.0, float(odometer))
        log_odo = np.log(odo)
        
        make_clean = str(make).strip()
        model_clean = str(model).strip()
        mm_key = f"{make_clean} | {model_clean}"
        
        # Fixed component
        b0 = self.fixed_params['Intercept']
        b_age = self.fixed_params['age']
        b_odo = self.fixed_params['log_odo']
        fe_log = b0 + b_age * age + b_odo * log_odo
        
        # Random component
        is_known_model = mm_key in self.random_effects
        is_known_brand = make_clean in self.brand_effects
        
        if is_known_model:
            re = self.random_effects[mm_key]
            n_samples = self.model_counts.get(mm_key, 0)
            confidence_level = "High" if n_samples >= 10 else ("Medium" if n_samples >= 3 else "Low")
            source_info = f"Model-specific random effect calibrated from {n_samples} historical auction record(s)."
        elif is_known_brand:
            re = self.brand_effects[make_clean]
            n_samples = self.brand_counts.get(make_clean, 0)
            confidence_level = "Brand Prior"
            source_info = f"Unseen model under known make '{make_clean}'. Shrunk using brand prior based on {n_samples} brand records."
        else:
            re = 0.0
            n_samples = 0
            confidence_level = "Market Prior"
            source_info = "Unseen brand and model. Evaluated against overall market depreciation baseline."
            
        pred_log = fe_log + re
        expected_price = np.exp(pred_log)
        
        # Compute total prediction uncertainty
        # For known models with many samples, sigma_eps dominates.
        # For unseen/rare models, we must also include sigma_group (random effect variance).
        if is_known_model:
            # Shrinkage absorbs most of sigma_group for well-observed models
            shrinkage_weight = 1.0 - (n_samples / (n_samples + 3.0))  # approx remaining uncertainty
            sigma_pred = np.sqrt(self.sigma_eps**2 + (self.sigma_group * shrinkage_weight)**2)
        elif is_known_brand:
            # Brand prior: partial sigma_group uncertainty remains
            sigma_pred = np.sqrt(self.sigma_eps**2 + (self.sigma_group * 0.7)**2)
        else:
            # Completely unseen: full sigma_group uncertainty
            sigma_pred = np.sqrt(self.sigma_eps**2 + self.sigma_group**2)
        
        range_low_1sigma = np.exp(pred_log - sigma_pred)
        range_high_1sigma = np.exp(pred_log + sigma_pred)
        ci_80_low = np.exp(pred_log - 1.282 * sigma_pred)
        ci_80_high = np.exp(pred_log + 1.282 * sigma_pred)
        
        # Value factor breakdown
        base_market_price = np.exp(b0)
        age_multiplier = np.exp(b_age * age)
        annual_dep_pct = (1.0 - np.exp(b_age)) * 100  # true annual depreciation rate
        model_multiplier = np.exp(re)
        
        # Mileage discount relative to 50,000 km reference (a meaningful baseline)
        ref_odo = 50000.0
        odo_vs_ref = np.exp(b_odo * (log_odo - np.log(ref_odo)))
        mileage_adj_pct = (odo_vs_ref - 1.0) * 100  # positive = more km = lower price
        
        # Comparable Market Baseline (Median of similar auction sales)
        if self.df is not None and len(self.df) > 0:
            exact_matches = self.df[(self.df['MAKE'].str.strip().str.lower() == make_clean.lower()) & 
                                    (self.df['MODEL'].str.strip().str.lower() == model_clean.lower())]
            if len(exact_matches) > 0:
                comp_price = float(exact_matches['FINAL BID VALUE'].median())
                comp_label = f"Median of similar {model_clean} sales"
            else:
                brand_matches = self.df[self.df['MAKE'].str.strip().str.lower() == make_clean.lower()]
                if len(brand_matches) > 0:
                    comp_price = float(brand_matches['FINAL BID VALUE'].median())
                    comp_label = f"Median of similar {make_clean} sales"
                else:
                    comp_price = float(self.df['FINAL BID VALUE'].median())
                    comp_label = "Median across all auction sales"
        else:
            if mm_key in self.model_medians:
                comp_price = float(self.model_medians[mm_key])
                comp_label = f"Median of similar {model_clean} sales"
            elif make_clean in self.brand_medians:
                comp_price = float(self.brand_medians[make_clean])
                comp_label = f"Median of similar {make_clean} sales"
            else:
                comp_price = float(self.overall_median)
                comp_label = "Median across all auction sales"

        # Rounding for clean display
        return {
            'expected_price': round(float(expected_price)),
            'range_low_1sigma': round(float(range_low_1sigma)),
            'range_high_1sigma': round(float(range_high_1sigma)),
            'ci_80_low': round(float(ci_80_low)),
            'ci_80_high': round(float(ci_80_high)),
            'age': age,
            'odometer': odo,
            'make': make_clean,
            'model': model_clean,
            'make_model': mm_key,
            'n_samples': n_samples,
            'confidence_level': confidence_level,
            'source_info': source_info,
            'random_effect': float(re),
            'sigma_pred': float(sigma_pred),
            'sigma_eps': self.sigma_eps,
            'breakdown': {
                'comparable_baseline_price': int(round(float(comp_price))),
                'comparable_baseline_label': comp_label,
                'base_market_price': int(round(float(base_market_price))),
                'age_depreciation_pct': float(round((1.0 - age_multiplier) * 100, 1)),
                'annual_depreciation_pct': float(round(annual_dep_pct, 1)),
                'age_coef': float(b_age),
                'mileage_adj_pct': float(round(mileage_adj_pct, 1)),
                'mileage_ref_km': int(ref_odo),
                'model_premium_pct': float(round((model_multiplier - 1.0) * 100, 1)),
            }
        }

    def get_historical_comps(self, make, model):
        """Returns matching historical records from the AP/TS dataset."""
        if self.df is None:
            return pd.DataFrame()
        
        make_clean = str(make).strip()
        model_clean = str(model).strip()
        
        matches = self.df[(self.df['MAKE'] == make_clean) & (self.df['MODEL'] == model_clean)]
        if len(matches) > 0:
            cols = ['VEH NO', 'DATE', 'YEAR', 'ODO METER', 'FINAL BID VALUE', 'TOTAL INCOME', 'BUYER NAME']
            return matches[cols].sort_values(by=['YEAR', 'FINAL BID VALUE'], ascending=[False, False])
        return pd.DataFrame()

if __name__ == "__main__":
    predictor = VehiclePricePredictor()
    sample = predictor.predict("Honda", "City", 2015, 80000)
    print("\nTest Prediction for Honda City 2015, 80k km:")
    for k, v in sample.items():
        print(f"  {k}: {v}")
