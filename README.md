# TVS Mobility — AP/Telangana Vehicle Auction Price Forecasting Engine
### Executive Valuation Tool & Statistical Intelligence Platform

> **Target Audience:** Machine Learning Engineers, Quantitative Modelers, Senior Business Analysts, and TVS Mobility Used Vehicle & Auction Remarketing Operations.

---

## Executive Summary

The **TVS Mobility Vehicle Price Forecasting Tool** is an end-to-end predictive valuation and risk-quantification system built for used passenger vehicle remarketing and auction operations in Andhra Pradesh & Telangana. 

Operating on historical transaction data across 8 auction cycles (**January 2026 – August 2026**), the platform replaces subjective manual price estimations with a **Hierarchical Mixed-Effects Model (MixedLM) powered by Empirical Bayes Shrinkage**. It addresses the core operational dilemma of vehicle remarketing: **pricing high enough to protect gross liquidation margins while bidding accurately enough to win quality inventory without holding-cost write-downs**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           TVS MOBILITY VALUATION PIPELINE                       │
└─────────────────────────────────────────────────────────────────────────────────┘
 [8-Month Auction Excel]  ──>  [Data Cleaner & Audit]  ──>  [Feature Engineering]
 (JAN-AUG 2026: 444 rows)     (5 vehicle mislabels fixed)  (log_price, age, log_odo)
                               (16 make / 81 model norm)
                                          │
                                          ▼
                         [Hierarchical Mixed-Effects Model]
                         (log(Price) = β0 + β1*Age + β2*log(Odo) + u_group + ε)
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       [Empirical Bayes Engine]                      [Risk & Variance Engine]
    • High-volume models: Full BLUP               • ±1σ Realistic Auction Band (~28%)
    • Low-volume models: Shrunk effect            • 80% Prediction Interval
    • Unseen models: Brand/Market Prior           • Depreciation & Mileage Breakdown
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          ▼
                       [Streamlit Executive Intelligence UI]
                         (Pricing • Comps • Curves • Audit)
```

---

## 1. Business Analyst Perspective

### 1.1 The Business Problem & Remarketing Context
In wholesale vehicle auctions and trade-in exchanges (such as TVS Mobility's AP/TS operations):
1. **The Cost of Over-Valuation (Winner's Curse):** Bidding or acquiring vehicles above true liquidation value results in trapped working capital, extended days-to-sell (aging inventory), compounding lot depreciation, and painful terminal losses.
2. **The Cost of Under-Valuation (Lost Volume):** Conservative heuristics lead to rejected bids, inventory starvation, lower turnover, and lost commission/fee income.
3. **The "Sparse Category" Dilemma:** In a regional sample of 444 vehicles across 8 months, market liquidity is concentrated in popular models (e.g., Honda City with 65 units, Maruti Swift with 21 units), while 47+ models appear fewer than 3 times (e.g., Citroen C3, Volvo S60, Nissan Kicks). Traditional rule-of-thumb pricing fails on rare models, while unregularized averages overreact to outlier bids.

### 1.2 Core Business Metrics & Valuation Drivers
The model decomposes every vehicle's fair market bid into four actionable, transparent components:

| Valuation Component | Model Calibration | Operational Interpretation |
| :--- | :--- | :--- |
| **Market Baseline Price** | $\exp(\beta_0) \approx \text{₹ } 48.3 \text{ Lakhs}$ | Theoretical baseline intercept for a hypothetical 0-year, 0-km asset prior to age/mileage scaling. |
| **Annual Depreciation** | $-11.9\% \text{ per year}$ | Compound annual depreciation rate across the market portfolio ($\exp(-0.1269) - 1$). |
| **Mileage Elasticity** | $-0.166 \text{ elasticity}$ | Each $10\%$ increase in odometer reading discounts the bid price by $\approx 1.66\%$ relative to standard benchmark usage ($50,000 \text{ km}$). |
| **Make/Model Premium** | $\exp(u_{\text{make\_model}}) - 1$ | Specific brand equity and liquidity premium/discount relative to the market baseline. |

### 1.3 Bid Risk Boundaries (Floor & Ceiling Execution)
A major pitfall of traditional analytics is providing a single static number (e.g., "₹ 3,50,000"), which gives a false impression of certainty. Auction environments are volatile and driven by bidder attendance, mechanical condition, and localized demand.

The platform provides **probabilistic decision bands**:
- **Fair Market Expected Price:** The central tendency for a vehicle in median auction condition.
- **Realistic Auction Range ($\pm 1\sigma$ Band $\approx 27.7\%$):** 68% probability envelope reflecting standard auction variability.
  - *Floor Bid (Conservative):* Protects liquidation margins for rapid asset turnover.
  - *Ceiling Bid (Aggressive):* Maximum acceptable threshold when inventory demand is high or vehicle condition is pristine.
- **80% Prediction Interval:** Operational safety bounds for risk committee audits and bulk portfolio underwriting.

### 1.4 Operational Workflow for Auction Managers
```
1. Input Vehicle   ──>  Select Brand, Model, Registration Year, and Odometer Reading.
2. Review Forecast ──>  Examine Fair Market Value vs. ±1σ Auction Range.
3. Verify Comps    ──>  Inspect historical AP/TS branch winning bids (date, buyer, realized price).
4. Evaluate Risk   ──>  Check Confidence Badge (High: ≥10 comps | Medium: 3–9 | Brand Prior: 0–2).
5. Execute Bid     ──>  Set auction reserve and ceiling bid within the recommended spread.
```

---

## 2. Machine Learning Engineering Perspective

### 2.1 Problem Formulation & Why Standard Approaches Fail
Given a dataset of size $N = 444$ containing $K = 81$ unique make-model combinations across $B = 16$ makes:

1. **Why Plain Ordinary Least Squares (OLS) Fails:**
   One-hot encoding $K=81$ categorical levels on 444 observations leaves fewer than 5.5 degrees of freedom per parameter. Over 50% of the categories have $n_k \le 2$. Ordinary least squares suffers from extreme variance inflation ($\text{VIF} \to \infty$), rank deficiency, and memorization of noisy transaction quirks.
2. **Why Gradient Boosted Trees (GBDT / XGBoost / LightGBM) Fall Behind:**
   With small sample sizes ($N=444$), decision trees split greedily on high-cardinality categorical features. Even with regularization (max depth 2–3, target encoding), tree models achieve an honest 5-fold CV MAPE of **34.1%** because they fail to capture smooth logarithmic multiplicative depreciation curves across age and odometer.
3. **Why Random Forest Falls Behind:**
   Random Forests with target encoding achieve **34.8% CV MAPE**, failing to extrapolate cleanly outside dense leaf partitions.

### 2.2 Mathematical Architecture: Hierarchical Linear Mixed Model
The core engine implements a **Mixed-Effects Log-Linear Formulation**:

$$\ln(\text{Price}_i) = \beta_0 + \beta_{\text{age}} \cdot \text{Age}_i + \beta_{\text{odo}} \cdot \ln(\text{Odometer}_i) + u_{j[i]} + \epsilon_i$$

Where:
- $\text{Price}_i$ is the actual winning auction bid (`FINAL BID VALUE`) in ₹.
- $\text{Age}_i = 2026 - \text{Year}_i$ (anchored to calendar year 2026; clipped at $\ge 0.5$ years).
- $\text{Odometer}_i$ is the verified odometer reading in km (clipped at $\ge 500\text{ km}$).
- $\boldsymbol{\beta} = [\beta_0, \beta_{\text{age}}, \beta_{\text{odo}}]^T$ are the **fixed effects** capturing market-wide baseline price, age depreciation, and mileage degradation.
- $u_j \sim \mathcal{N}(0, \sigma^2_{\text{group}})$ is the **random intercept** (Best Linear Unbiased Predictor / BLUP) for make-model combination $j \in \{1, \dots, 81\}$.
- $\epsilon_i \sim \mathcal{N}(0, \sigma^2_{\epsilon})$ is the independent and identically distributed residual error representing idiosyncratic condition variation (scratches, tyre wear, service history, auction crowd dynamics).

### 2.3 Empirical Bayes Shrinkage Mechanism
To prevent sparse categories ($n_j = 1$) from exhibiting extreme, noisy coefficients, the model applies **Empirical Bayes (James-Stein style) Shrinkage**:

$$\hat{u}_j = \frac{n_j}{n_j + \lambda} \cdot \bar{r}_j + \left(1 - \frac{n_j}{n_j + \lambda}\right) \cdot u_{\text{brand}}$$

Where:
- $n_j$ is the number of historical auction transactions observed for model $j$.
- $\bar{r}_j = \frac{1}{n_j} \sum_{i \in \text{group } j} (\ln(y_i) - \hat{y}_{i, \text{fixed}})$ is the raw group mean residual.
- $\lambda = \frac{\sigma^2_{\epsilon}}{\sigma^2_{\text{group}}}$ is the variance ratio regularization penalty.
- $u_{\text{brand}}$ is the hierarchical brand-level prior:
  $$u_{\text{brand}} = \frac{1}{|M_{\text{brand}}|} \sum_{m \in M_{\text{brand}}} \hat{u}_m$$
  - **High-sample models (e.g. Honda City, $n=65$):** The shrinkage weight $\frac{n_j}{n_j + \lambda} \to 1.0$. The estimate relies almost entirely on its own empirical data.
  - **Low-sample models (e.g. Citroen C3, $n=1$):** The shrinkage weight $\to 0$. The estimate is pulled toward the brand/market expectation, preventing spurious valuation spikes.
  - **Cold-Start / Unseen Models:** The engine falls back hierarchically:
    1. Known Make & Model $\to$ Full BLUP + Shrinkage
    2. Known Make & Unseen Model $\to$ Brand Prior $u_{\text{brand}}$
    3. Unseen Make & Model $\to$ Global Market Baseline ($u = 0.0$)

### 2.4 Uncertainty Quantification & Prediction Intervals
Prediction variance combines observation noise with parameter/random effect estimation uncertainty:

$$\sigma_{\text{pred}}^2 = \sigma_{\epsilon}^2 + \left[\sigma_{\text{group}} \cdot \left(1 - \frac{n_j}{n_j + 3}\right)\right]^2$$

Back-transforming to original currency scale ₹:
- **Expected Value:** $\hat{Y} = \exp(\hat{\mu}_{\log})$
- **Realistic Auction Range ($\pm 1\sigma$):** $[\exp(\hat{\mu}_{\log} - \sigma_{\text{pred}}), \; \exp(\hat{\mu}_{\log} + \sigma_{\text{pred}})]$
- **80% Confidence Interval ($z = 1.282$):** $[\exp(\hat{\mu}_{\log} - 1.282\sigma_{\text{pred}}), \; \exp(\hat{\mu}_{\log} + 1.282\sigma_{\text{pred}})]$

---

## 3. Model Benchmark & Empirical Results

All evaluations were conducted using **honest 5-fold cross-validation on the original ₹ price scale** (not log-space artifacts):

| Model Architecture | In-Sample MAPE | 5-Fold Honest CV MAPE | 5-Fold Honest CV MAE | CV $R^2$ Score | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Hierarchical MixedLM (Shrinkage)** | **20.67%** | **27.91% (±3.68%)** | **₹ 51,895** | **0.656** | **Selected Engine** |
| Target-Encoded Gradient Boosting | 18.42% | 34.12% (±4.10%) | ₹ 63,400 | 0.528 | Overfits small sample |
| Target-Encoded Random Forest | 22.15% | 34.80% (±3.85%) | ₹ 64,950 | 0.512 | Discontinuous step predictions |
| Target-Encoded Ridge Regression | 29.80% | 35.15% (±4.50%) | ₹ 66,200 | 0.495 | Inflexible linear regularizer |
| Unregularized OLS (One-Hot) | 16.20% | 48.90% (±12.4%) | ₹ 98,400 | -0.120 | Severe rank-deficiency failure |

### Fitted Model Coefficients
- **$\beta_0$ (Intercept):** $+15.3909$
- **$\beta_{\text{age}}$ (Age Slope):** $-0.1269$ ($p < 0.001$, $z = -14.82$)
- **$\beta_{\text{odo}}$ ($\ln(\text{Odo})$ Slope):** $-0.1665$ ($p < 0.001$, $z = -4.91$)
- **Residual Standard Deviation ($\sigma_{\epsilon}$):** $0.2771$ (~$27.7\%$ log error)
- **Group Effect Standard Deviation ($\sigma_{\text{group}}$):** $0.4167$

---

## 4. Data Engineering & Integrity Audit

The raw dataset spans 8 monthly auction logs from `JAN26_AUG26_REPORT.xlsx` containing 444 total rows. Prior to training, `data_cleaner.py` resolves critical real-world data collection issues:

### 4.1 Rectified Data Entry Mislabels
A manual vehicle-level audit identified and corrected several high-impact mislabelings:
1. `AP37DE4995`: **Volkswagen Vento** entered under `MAKE = Hyundai`.
2. `AP16BU7806`: **Maruti Wagon-R** entered under `MAKE = Hyundai`.
3. `AP16BQ7777`: **Hyundai i10** entered under `MAKE = Maruti`.
4. `TS07FP7214`: **Nissan Datsun** entered as `MAKE = Renault`, `MODEL = Datson`.
5. `AP16CG0018`: **Maruti Ritz** (2013, 196,300 km) entered as typo `MODEL = Riaz`.

### 4.2 Casing & Spelling Standardizations
Automated normalization dict mappings:
- `Eco Sport`, `Ecosport` $\to$ `EcoSport`
- `Wr-v`, `Wr-V` $\to$ `WR-V`
- `Cr-V` $\to$ `CR-V`
- `Br-V` $\to$ `BR-V`
- `Xuv500`, `Xuv300` $\to$ `XUV500`, `XUV300`
- `Octovia` $\to$ `Octavia`

### 4.3 Model & Trim Consolidation
To prevent fragmentation of statistical degrees of freedom:
- `V Brezza`, `Brezza`, `Vitara Brezza` $\to$ unified as `Vitara Brezza`
- `Swift Dzire`, `Dzire` $\to$ unified as `Dzire`
- `i20 Asta` $\to$ merged into `i20`
- `Getz Prime` $\to$ merged into `Getz`
- `Celerio-X` $\to$ merged into `Celerio`

Result: Reduced noisy categories from 97 down to **81 clean, robust make-model groupings across 16 manufacturers**.

---

## 5. Repository File Structure & Architecture

```
.
├── JAN26_AUG26_REPORT.xlsx    # Raw multi-sheet auction ledger (8 monthly sheets)
├── cleaned_vehicle_data.csv   # Normalized, audited master dataset (444 records)
├── data_cleaner.py            # Automated ETL, data cleaning, and typo correction script
├── model_engine.py            # Hierarchical MixedLM, empirical Bayes shrinkage & inference engine
├── price_model.joblib         # Serialized production model payload (fixed params, REs, variances)
├── app.py                     # Streamlit Executive Dashboard (UI, interactive Plotly curves, comps)
├── run_app.bat                # 1-click Windows execution launcher for business operators
├── test_models.py             # 5-fold cross-validation benchmarking script (MixedLM vs GBDT vs RF)
├── verify_all.py              # Automated test suite covering 13 edge cases (rare models, cold start)
├── deep_inspect.py            # Data exploratory audit script (zero/negative checks, distributions)
└── README.md                  # Detailed technical & business documentation (this file)
```

---

## 6. Installation & Execution Guide

### Prerequisites
- Windows 10/11 or Linux
- Python 3.10 to 3.14
- Standard Python data science libraries

### 6.1 Environment Setup
Install required dependencies:
```bash
pip install pandas numpy scikit-learn statsmodels streamlit plotly joblib openpyxl
```

### 6.2 Retraining the Model / Running the ETL
To clean the raw Excel data and retrain the MixedLM model from scratch:
```bash
# Step 1: Clean raw multi-sheet Excel file
python data_cleaner.py

# Step 2: Train Hierarchical MixedLM and save price_model.joblib
python model_engine.py

# Step 3: Run comprehensive validation suite across edge cases
python verify_all.py
```

### 6.3 Launching the Executive Dashboard
Double-click `run_app.bat` or execute in terminal:
```bash
streamlit run app.py
```
The browser interface will launch at `http://localhost:8501`.

---

## 7. Interactive Dashboard Features

The Streamlit executive interface (`app.py`) provides:
1. **Interactive Vehicle Configurator (Sidebar):** Select Make, Model (or input custom unlisted model), Registration Year (2000–2026), and Odometer with preset buttons (40k, 80k, 120k km).
2. **Hero Valuation Display:** Instant Fair Market Expected Bid in formatted Indian Rupees (`₹`), with Confidence Badge and historical transaction counts.
3. **$\pm 1\sigma$ and 80% Prediction Bands:** Clear display of realistic auction margins.
4. **Driver Decomposition Pill Boxes:** Transparent quantification of Base Price, Age Depreciation %, Mileage Adjustment %, and Make/Model Premium %.
5. **Interactive Plotly Sensitivity Curves:**
   - *Price vs. Odometer Curve:* Visualizes logarithmic mileage decay against actual historical sales comps.
   - *Price vs. Year Curve:* Visualizes annual depreciation slope with uncertainty envelope.
6. **Historical Auction Comps Tab:** Instant lookup of every matching historical vehicle sold in AP/TS, with Vehicle Number, Sale Date, Odometer, Winning Bid, and Buyer Name.
7. **Caveat Resolution & Audit Matrix:** Transparent accountability verifying that all data inconsistencies and statistical requirements are fully addressed.
8. **AP/TS Market Data Explorer:** Filterable raw auction data grid with one-click CSV export.

---

## 8. Business Takeaways & Future Roadmap

### Key Business Takeaways
1. **Depreciation is Non-Linear:** Vehicles lose value rapidly in their initial 3–5 years ($\approx 11.9\%$ compound annual rate), flattening out in older brackets. Logarithmic modeling accurately captures this dynamic.
2. **Mileage Matters, But Slower Than Age:** The $-0.166$ elasticity confirms that mechanical age dominates pricing, but high-mileage penalties are pronounced for vehicles above $100,000\text{ km}$.
3. **Shrinkage Eliminates Pricing Disasters:** Rare inventory (e.g., luxury imports or discontinued hatchbacks) is safely priced toward brand averages rather than hallucinating based on 1 old sale.

### Recommended Next Iterations
- **Fuel Type & Transmission Features:** Incorporate Petrol vs. Diesel vs. CNG and Manual vs. Automatic once operational teams begin recording variant details systematically.
- **RTO / District Sub-Clustering:** Add regional coefficients (e.g., Vijayawada vs. Hyderabad vs. Visakhapatnam) to account for hyper-local resale premiums.
- **Automated Monthly Ingestion:** Link the script directly to cloud Google Sheets or SQL warehouse to auto-retrain weights each month.

---
*Developed for TVS Mobility Private Limited — AP/Telangana Branch Pricing Intelligence Engine.*
