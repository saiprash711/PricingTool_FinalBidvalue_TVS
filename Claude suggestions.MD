Method options, ranked for your situation

Linear regression on log(price) with age + log(odometer) + a shrinkage-adjusted make/model effect (hierarchical/mixed-effects style). This is what I'd recommend to start: it's interpretable, handles the sparse-category problem gracefully (rare models borrow strength from the overall trend instead of getting a wild estimate from 1 data point), and is easy to explain to stakeholders. On your data this gets ~27% in-sample MAPE, ~30% honest out-of-sample MAPE.
Gradient-boosted trees (XGBoost/LightGBM/CatBoost). Can capture non-linear interactions (e.g. mileage matters less for already-cheap old cars) and CatBoost handles categoricals natively. Worth trying as a second pass, but with only 444 rows you need to be careful about overfitting — cross-validate, keep trees shallow.
Random Forest. Similar tradeoffs to boosting, slightly more forgiving of small data, usually a bit less accurate than boosting when tuned well.

I'd skip plain OLS with one-hot-encoded Make+Model dummies (97 columns on 444 rows — that's overfitting by construction) and skip anything deep-learning (nowhere near enough data).

What do you want to build the tool in?
Python (scikit-learn/statsmodels + Streamlit UI)

Here's your build path, step by step.

1. Set up environment

bash
pip install pandas numpy scikit-learn statsmodels streamlit joblib openpyxl

2. Clean the data first
Load all 8 monthly sheets, concatenate, then fix the inconsistencies before anything else:

Standardize casing: df['MODEL'] = df['MODEL'].str.strip().str.title() won't fully fix "WR-V"/"Wr-V" — build a small manual mapping dict for known variants ({'Wr-V':'WR-V','Wr-v':'WR-V','Cr-V':'CR-V','Eco Sport':'EcoSport','Ecosport':'EcoSport','Octovia':'Octavia', ...}) and .replace() it in.
Fix the mislabeled rows I found (Vento tagged as Hyundai, i10 tagged as Maruti, etc.) — these are one-off data entry errors, correct the MAKE value directly for those VEH NOs.
Decide on "V Brezza"/"Vitara Brezza"/"Brezza" — likely the same model at different trims; I'd merge unless you specifically want trim-level granularity.

3. Feature engineering

python
df['age'] = 2026 - df['YEAR']
df['log_price'] = np.log(df['FINAL BID VALUE'])
df['log_odo'] = np.log(df['ODO METER'])
df['make_model'] = df['MAKE'] + ' | ' + df['MODEL']

Working in log(price) matters — it turns multiplicative depreciation into something linear regression can actually fit.

4. Start with the interpretable model (baseline)
Use statsmodels mixed linear model — this handles your sparse-category problem (47 models with 1 sample) properly, by letting rare models "borrow strength" from the overall trend instead of getting a wild fitted coefficient from one row:

python
import statsmodels.formula.api as smf
model = smf.mixedlm("log_price ~ age + log_odo", df, groups=df["make_model"])
result = model.fit()

This gives you a global age/mileage effect plus a shrunk per-model adjustment. Check result.summary() and look at result.random_effects for the per-model intercepts.

5. Try a tree-based model as a second pass

python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from category_encoders import TargetEncoder  # pip install category_encoders

X = df[['age','log_odo','make_model']]
X_encoded = TargetEncoder().fit_transform(X[['make_model']], df['log_price'])
X_final = pd.concat([df[['age','log_odo']], X_encoded], axis=1)

gbr = GradientBoostingRegressor(max_depth=3, n_estimators=100, learning_rate=0.05)
scores = cross_val_score(gbr, X_final, df['log_price'], cv=5, scoring='neg_mean_absolute_error')

Keep max_depth shallow (2–3) — with 444 rows a deep model will memorize noise.

6. Validate honestly
Use 5-fold cross-validation (not just training-set fit) and report MAPE on the original price scale, not log scale:

python
pred_price = np.exp(pred_log_price)
mape = np.mean(np.abs(actual - pred_price) / actual) * 100

Compare both models' CV MAPE and pick the winner — don't just trust in-sample R².

7. Save the model

python
import joblib
joblib.dump(result, 'price_model.pkl')  # or the sklearn pipeline

8. Build the Streamlit UI

python
import streamlit as st

st.title("AP/TS Vehicle Price Estimator")
make = st.selectbox("Make", sorted(df['MAKE'].unique()))
model_options = sorted(df[df['MAKE']==make]['MODEL'].unique())
model_name = st.selectbox("Model", model_options)
year = st.number_input("Year", min_value=1998, max_value=2026, value=2018)
odometer = st.number_input("Odometer (km)", min_value=0, value=80000)

if st.button("Predict price"):
    # rebuild features, run through your saved model, np.exp() the result
    st.metric("Estimated bid value", f"₹{predicted:,.0f}")
    st.caption(f"Based on {n} historical records for this model")

Run with streamlit run app.py.

One thing to flag as you build: with only 444 rows and 97 model combos, don't expect tight predictions — expect ~25–30% error bands. Show a range (e.g. ±1 residual std dev in log space) alongside the point estimate rather than a single confident number, or the tool will look more precise than it is.