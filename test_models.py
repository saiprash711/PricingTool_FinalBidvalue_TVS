import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

# 1. Load data
excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
df = pd.concat([pd.read_excel(excel_path, sheet_name=s) for s in sheets], ignore_index=True)

# 2. Clean data
# Fix mislabeled makes
df.loc[df['VEH NO'] == 'AP37DE4995', 'MAKE'] = 'Volkswagen' # Vento
df.loc[df['VEH NO'] == 'AP16BU7806', 'MAKE'] = 'Maruti'     # Wagon-R
df.loc[df['VEH NO'] == 'AP16BQ7777', 'MAKE'] = 'Hyundai'    # i10
df.loc[df['VEH NO'] == 'TS07FP7214', 'MAKE'] = 'Nissan'     # Datson
df.loc[df['VEH NO'] == 'TS07FP7214', 'MODEL'] = 'Datsun'
df.loc[df['VEH NO'] == 'AP16CG0018', 'MODEL'] = 'Ritz'      # Riaz -> Ritz

# Model standardizations
model_map = {
    'Eco Sport': 'EcoSport',
    'Ecosport': 'EcoSport',
    'Wr-v': 'WR-V',
    'Wr-V': 'WR-V',
    'Cr-V': 'CR-V',
    'Br-V': 'BR-V',
    'Xuv500': 'XUV500',
    'Xuv300': 'XUV300',
    'Octovia': 'Octavia',
    'Datson': 'Datsun',
    'V Brezza': 'Vitara Brezza',
    'Brezza': 'Vitara Brezza',
    'Swift Dzire': 'Dzire',
    'i20 Asta': 'i20',
    'Getz Prime': 'Getz',
    'Celerio-X': 'Celerio',
}
df['MODEL'] = df['MODEL'].replace(model_map)
df['MAKE'] = df['MAKE'].str.strip()
df['MODEL'] = df['MODEL'].str.strip()

# Features
df['age'] = 2026 - df['YEAR']
df['log_price'] = np.log(df['FINAL BID VALUE'])
df['log_odo'] = np.log(df['ODO METER'])
df['make_model'] = df['MAKE'] + ' | ' + df['MODEL']

print(f"Cleaned dataset: {len(df)} rows, {df['make_model'].nunique()} unique make_models, {df['MAKE'].nunique()} makes.")

# Evaluate MixedLM
try:
    md = smf.mixedlm("log_price ~ age + log_odo", df, groups=df["make_model"])
    mdf = md.fit()
    print("\n--- MixedLM Summary ---")
    print(mdf.summary().tables[0])
    print(mdf.summary().tables[1])
    in_sample_preds = np.exp(mdf.predict(df))
    # Note: predict() on mixedlm only predicts fixed effects by default!
    # Let's check random effects
    re = mdf.random_effects
    fitted_vals = []
    for idx, row in df.iterrows():
        fe = mdf.params['Intercept'] + mdf.params['age']*row['age'] + mdf.params['log_odo']*row['log_odo']
        group_re = re.get(row['make_model'], [0.0])[0] if row['make_model'] in re else 0.0
        fitted_vals.append(np.exp(fe + group_re))
    fitted_vals = np.array(fitted_vals)
    actual = df['FINAL BID VALUE'].values
    in_mape = np.mean(np.abs(actual - fitted_vals) / actual) * 100
    in_mae = np.mean(np.abs(actual - fitted_vals))
    print(f"MixedLM in-sample MAPE (with RE): {in_mape:.2f}% | MAE: Rs. {in_mae:,.0f}")
except Exception as e:
    print("MixedLM error:", e)

# Cross validation test across models
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Test 1: Hierarchical / Empirical Bayes Shrinkage Linear Model
# Test 2: Target Encoded Ridge
# Test 3: Target Encoded Gradient Boosting
# Test 4: Target Encoded Random Forest

print("\n--- Running 5-Fold Cross Validation ---")
for model_type in ['hierarchical', 'ridge', 'gbr', 'rf']:
    mapes = []
    maes = []
    r2s = []
    for train_idx, test_idx in kf.split(df):
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        
        if model_type == 'hierarchical':
            # Global baseline fit
            # log_price = b0 + b1*age + b2*log_odo + delta_make_model
            # Fit overall fixed effects:
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(train[['age', 'log_odo']], train['log_price'])
            train['resid'] = train['log_price'] - lr.predict(train[['age', 'log_odo']])
            
            # Empirical Bayes shrinkage:
            # global prior mean = 0, variance = var(resid)
            # group mean = y_bar_g, group variance = sigma^2 / n_g
            # shrunk effect = (n_g / (n_g + lambda)) * y_bar_g
            # Also brand shrinkage for rare models
            grand_var = train['resid'].var()
            group_stats = train.groupby('make_model')['resid'].agg(['mean', 'count'])
            brand_stats = train.groupby('MAKE')['resid'].agg(['mean', 'count'])
            
            # Optimal lambda = residual_var / group_variance
            # Typically lambda around 2 to 5 works well for small samples
            shrinkage_k = 3.0
            brand_k = 2.0
            
            # Predict test
            preds = []
            for _, r in test.iterrows():
                base_pred = lr.predict([[r['age'], r['log_odo']]])[0]
                brand = r['MAKE']
                mm = r['make_model']
                
                brand_adj = 0.0
                if brand in brand_stats.index:
                    b_mean = brand_stats.loc[brand, 'mean']
                    b_cnt = brand_stats.loc[brand, 'count']
                    brand_adj = (b_cnt / (b_cnt + brand_k)) * b_mean
                
                mm_adj = brand_adj
                if mm in group_stats.index:
                    m_mean = group_stats.loc[mm, 'mean']
                    m_cnt = group_stats.loc[mm, 'count']
                    # Shrink towards brand adjustment
                    weight = m_cnt / (m_cnt + shrinkage_k)
                    mm_adj = weight * m_mean + (1 - weight) * brand_adj
                
                pred_log = base_pred + mm_adj
                preds.append(np.exp(pred_log))
            preds = np.array(preds)
            
        elif model_type in ['ridge', 'gbr', 'rf']:
            # Target encode make_model and MAKE
            # Out-of-fold target encoding for train
            global_mean = train['log_price'].mean()
            mm_means = train.groupby('make_model')['log_price'].agg(['mean', 'count'])
            mm_smooth = (mm_means['mean'] * mm_means['count'] + global_mean * 5) / (mm_means['count'] + 5)
            
            train_enc = train['make_model'].map(mm_smooth).fillna(global_mean)
            test_enc = test['make_model'].map(mm_smooth).fillna(global_mean)
            
            X_train = pd.DataFrame({'age': train['age'], 'log_odo': train['log_odo'], 'mm_enc': train_enc})
            X_test = pd.DataFrame({'age': test['age'], 'log_odo': test['log_odo'], 'mm_enc': test_enc})
            
            if model_type == 'ridge':
                m = Ridge(alpha=1.0)
            elif model_type == 'gbr':
                m = GradientBoostingRegressor(max_depth=2, n_estimators=60, learning_rate=0.08, random_state=42)
            elif model_type == 'rf':
                m = RandomForestRegressor(max_depth=4, n_estimators=100, min_samples_leaf=2, random_state=42)
                
            m.fit(X_train, train['log_price'])
            preds = np.exp(m.predict(X_test))
            
        actual_test = test['FINAL BID VALUE'].values
        mapes.append(mean_absolute_percentage_error(actual_test, preds) * 100)
        maes.append(mean_absolute_error(actual_test, preds))
        r2s.append(r2_score(actual_test, preds))
        
    print(f"{model_type:<14} | CV MAPE: {np.mean(mapes):.2f}% (±{np.std(mapes):.2f}%) | CV MAE: Rs. {np.mean(maes):,.0f} | CV R2: {np.mean(r2s):.3f}")
