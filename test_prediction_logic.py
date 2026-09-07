import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
df = pd.concat([pd.read_excel(excel_path, sheet_name=s) for s in sheets], ignore_index=True)

# Clean
df.loc[df['VEH NO'] == 'AP37DE4995', 'MAKE'] = 'Volkswagen'
df.loc[df['VEH NO'] == 'AP16BU7806', 'MAKE'] = 'Maruti'
df.loc[df['VEH NO'] == 'AP16BQ7777', 'MAKE'] = 'Hyundai'
df.loc[df['VEH NO'] == 'TS07FP7214', 'MAKE'] = 'Nissan'
df.loc[df['VEH NO'] == 'TS07FP7214', 'MODEL'] = 'Datsun'
df.loc[df['VEH NO'] == 'AP16CG0018', 'MODEL'] = 'Ritz'

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

df['age'] = 2026 - df['YEAR']
df['log_price'] = np.log(df['FINAL BID VALUE'])
df['log_odo'] = np.log(df['ODO METER'])
df['make_model'] = df['MAKE'] + ' | ' + df['MODEL']

# Fit MixedLM
md = smf.mixedlm("log_price ~ age + log_odo", df, groups=df["make_model"])
mdf = md.fit()

# Residual analysis and variance components
sigma_eps = np.sqrt(mdf.scale)
sigma_group = np.sqrt(mdf.cov_re.iloc[0, 0])
print(f"Residual std dev (sigma_eps): {sigma_eps:.4f}")
print(f"Group random effects std dev (sigma_group): {sigma_group:.4f}")

# Check predictions for specific cases:
# E.g. Maruti Dzire 2018 with 60,000 km
# Honda City 2015 with 80,000 km
# Rare model: Citroen C3 2022 with 30,000 km
# Unseen model: Maruti Jimny 2023 with 20,000 km

def predict_price(make, model, year, odo):
    age = max(0.5, 2026 - year)
    log_odo = np.log(max(odo, 500))
    mm = f"{make} | {model}"
    
    # Fixed effect prediction
    fe = mdf.params['Intercept'] + mdf.params['age'] * age + mdf.params['log_odo'] * log_odo
    
    # Group random effect
    if mm in mdf.random_effects:
        re = mdf.random_effects[mm].iloc[0]
        n_obs = len(df[df['make_model'] == mm])
    else:
        # Fallback to make-level average random effect if model not seen
        brand_models = df[df['MAKE'] == make]['make_model'].unique()
        brand_res = [mdf.random_effects[m].iloc[0] for m in brand_models if m in mdf.random_effects]
        re = np.mean(brand_res) if len(brand_res) > 0 else 0.0
        n_obs = 0
        
    pred_log = fe + re
    pred_price = np.exp(pred_log)
    
    # 80% and 90% prediction intervals
    # std error in log space ~ sigma_eps
    low_80 = np.exp(pred_log - 1.282 * sigma_eps)
    high_80 = np.exp(pred_log + 1.282 * sigma_eps)
    low_1sigma = np.exp(pred_log - sigma_eps)
    high_1sigma = np.exp(pred_log + sigma_eps)
    
    return {
        'pred_price': pred_price,
        'low_1sigma': low_1sigma,
        'high_1sigma': high_1sigma,
        'low_80': low_80,
        'high_80': high_80,
        're': re,
        'n_obs': n_obs
    }

for test_case in [
    ('Honda', 'City', 2015, 80000),
    ('Maruti', 'Dzire', 2018, 60000),
    ('Citroen', 'C3', 2022, 30000),
    ('Toyota', 'Innova', 2010, 150000),
    ('Maruti', 'Alto', 2014, 70000),
]:
    res = predict_price(*test_case)
    print(f"\n{test_case}:")
    print(f"  Pred: Rs. {res['pred_price']:,.0f}")
    print(f"  +-1sigma range: Rs. {res['low_1sigma']:,.0f} - Rs. {res['high_1sigma']:,.0f}")
    print(f"  80% range: Rs. {res['low_80']:,.0f} - Rs. {res['high_80']:,.0f}")
    print(f"  RE: {res['re']:.3f} (n={res['n_obs']})")
