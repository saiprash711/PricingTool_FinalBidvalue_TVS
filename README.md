# TVS Certified — Multi-Branch Vehicle Auction Price Forecasting Engine
### Executive Valuation Tool & Statistical Intelligence Platform (TN, KL, AP/TS)

> **Target Audience:** Machine Learning Engineers, Quantitative Modelers, Senior Business Analysts, and TVS Certified Used Vehicle & Auction Remarketing Operations across Tamil Nadu, Kerala, and Andhra Pradesh/Telangana.

---

## Executive Summary

The **TVS Certified Vehicle Price Forecasting Tool** is an end-to-end predictive valuation and risk-quantification system built for used passenger vehicle remarketing and auction operations across **Tamil Nadu (TN), Kerala (KL), and Andhra Pradesh/Telangana (AP/TS)**.

Operating on a master combined dataset of **3,328 historical auction transactions** spanning **January 2026 – August 2026**, the platform replaces subjective manual price estimations with a **Hierarchical Mixed-Effects Model (MixedLM) powered by Empirical Bayes Shrinkage and Regional Branch Calibration**. It addresses the core operational dilemma of vehicle remarketing: **pricing high enough to protect gross liquidation margins while bidding accurately enough to win quality inventory without holding-cost write-downs**.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      TVS CERTIFIED MULTI-BRANCH VALUATION PIPELINE              │
└─────────────────────────────────────────────────────────────────────────────────┘
 [Jan-Jul Conversion Data]   [8-Month Auction Report]
 (TN, KL, AP/TS: 3,272 rows)  (AP/TS Aug: 60 rows)
            │                           │
            └─────────────┬─────────────┘
                          ▼
            [Master Dataset Combination & Audit]
            (3,328 Total Rows • 23 Makes • 280 Models)
                          │
                          ▼
     [Hierarchical Mixed-Effects Model + Regional Calibration]
     log(Price) = β0 + β_age*Age + β_odo*log(Odo) + C(STATE) + u_model + ε
                          │
         ┌────────────────┴────────────────┐
         ▼                                 ▼
 [Empirical Bayes Engine]         [Regional Branch Calibration]
 • High-volume: Full BLUP         • TN Premium: +25.2% (p < 0.0001)
 • Low-volume: Shrunk effect      • KL Premium: +17.0% (p < 0.0001)
 • Unseen models: Brand Prior     • AP/TS: Calibrated Baseline
         │                                 │
         └────────────────┬────────────────┘
                          ▼
            [Streamlit Multi-Branch Executive UI]
   (Multi-Branch Pricing • Regional Comps • Sensitivity Curves • Master Excel Export)
```

---

## 1. Regional Scope & Master Dataset Integration

### 1.1 Multi-Branch Data Synthesis (Jan–Aug 2026)
The system unifies two core data streams into a single master repository:
1. **Jan to July 2026 Conversion Data.xlsx (3,272 rows):** Contains retail conversion transactions across Tamil Nadu (`TN`: 1,896 records), Kerala (`KL`: 988 records), and Andhra Pradesh/Telangana (`AP/TS`: 384 records).
2. **JAN26_AUG26_REPORT.xlsx (444 rows):** Multi-sheet auction report contributing 60 August 2026 AP/TS records not present in the conversion data.
3. **Combined Master Dataset:** Deduplicated on vehicle identity and auction month to produce **3,328 unique auction transactions** exported to both `cleaned_vehicle_data.csv` and `COMBINED_JAN26_AUG26_ALL_BRANCHES.xlsx`.

| Branch / Region | Active Units | Volume Share | Average Realized Bid | Top Seller Segments |
| :--- | :--- | :--- | :--- | :--- |
| **Tamil Nadu (TN)** | 1,896 | 57.0% | ₹ 3,68,410 | Exchange ROTN, Khivraj Motor, Exchange Chennai |
| **Kerala (KL)** | 988 | 29.7% | ₹ 3,24,180 | Popular Maruti, Incheon Kia, EVM Citroen |
| **Andhra Pradesh & Telangana (AP/TS)** | 444 | 13.3% | ₹ 2,95,380 | TVS Certified AP/TS, Exchange Hyderabad, Vijayawada, Vizag |
| **Total / Combined** | **3,328** | **100.0%** | **₹ 3,46,890** | **Multi-Branch TVS Remarketing Network** |

---

## 2. Quantitative Model Architecture & Calibration

### 2.1 Mathematical Formulation
The regression engine implements a **Mixed-Effects Log-Linear Model** with fixed effects for vehicle depreciation, mileage elasticity, and regional state premiums, combined with empirical Bayes random intercepts per vehicle model:

$$\ln(\text{Price}_i) = \beta_0 + \beta_{\text{age}} \cdot \text{Age}_i + \beta_{\text{odo}} \cdot \ln(\text{Odometer}_i) + \gamma_{\text{state}[i]} + u_{j[i]} + \epsilon_i$$

Where:
- $\text{Price}_i$ is the actual winning auction bid (`FINAL BID VALUE`) in ₹.
- $\text{Age}_i = 2026 - \text{Year}_i$ (anchored to calendar year 2026; clipped at $\ge 0.5$ years).
- $\text{Odometer}_i$ is the verified odometer reading in km (clipped at $\ge 500\text{ km}$).
- $\gamma_{\text{state}[i]}$ represents regional state fixed effects:
  - $\gamma_{\text{AP/TS}} = 0.0$ (Baseline)
  - $\gamma_{\text{TN}} = +0.224 \implies \mathbf{+25.2\%}$ regional price premium ($z = 8.77, p < 0.0001$)
  - $\gamma_{\text{KL}} = +0.156 \implies \mathbf{+17.0\%}$ regional price premium ($z = 4.78, p < 0.0001$)
- $u_{j[i]} \sim \mathcal{N}(0, \sigma_{\text{model}}^2)$ is the random intercept for vehicle make/model $j$.
- $\epsilon_i \sim \mathcal{N}(0, \sigma_\epsilon^2)$ is the residual error.

### 2.2 Handling Missing Registration Years in Kerala Records
In `Jan to july 2026 Conversion Data.xlsx`, 602 records from Kerala (KL) lacked registration years.
- **Regression Training Rigor:** Rather than imputing synthetic or guessed years (which would distort the empirical age depreciation coefficient $\beta_{\text{age}}$), the MixedLM regression model is trained strictly on the **2,693 verified records** with ground-truth years.
- **Inventory Comps Retention:** All 3,328 records remain available in the historical comps search, inventory lookup, and market data explorer tabs so transaction visibility is 100% preserved.

### 2.3 Validated Out-of-Sample Performance (Tamil Nadu Master Dataset)
Evaluated via rigorous 5-fold cross-validation on 5,300 verified Tamil Nadu auction records on the original ₹ scale:
- **1. MixedLM Econometric Alone:** 20.78% CV MAPE | ₹ 68,193 CV MAE | 0.673 $R^2$
- **2. CatBoost Trees Alone:** 18.90% CV MAPE | ₹ 66,856 CV MAE | 0.642 $R^2$
- **3. Dual-Engine Ensemble (Blended):** **18.63% CV MAPE** | **₹ 63,533 CV MAE** | **0.686 $R^2$**
- **Economic Value:** The Dual-Engine Ensemble reduces mean absolute error by **₹ 4,661 per car** over MixedLM alone (-2.15% MAPE gain), providing highest accuracy for high-sample trims and robust empirical Bayes BLUP shrinkage for rare trims.

---

## 3. Project File Structure

```
TVS C SAI/Aug Month data/
├── app.py                             # Executive Streamlit Web Application (Dual-Engine Ensemble)
├── model_engine.py                    # MixedLM + CatBoost Dual-Engine Ensemble, adaptive shrinkage
├── data_cleaner.py                    # Multi-branch ETL, resilient safe Excel loader, normalizer
├── price_model.joblib                 # Serialized ensemble payload with BLUPs and benchmark metrics
├── catboost_model.cbm                 # Serialized CatBoost categorical gradient boosting model
├── cleaned_tn_vehicle_data.csv        # Cleaned Tamil Nadu master dataset (5,300 auction records)
├── masterdata_TN.xlsx                 # Source Tamil Nadu auction records (July 2024 - Sept 2026)
├── test_tn_pipeline.py                # Automated Tamil Nadu pricing pipeline test suite
├── test_upload_and_branch.py          # Automated multi-branch test suite
├── verify_all.py                      # Edge-case validation script across vehicle categories
├── run_app.bat                        # One-click Windows launcher for the Streamlit dashboard
└── README.md                          # Technical and business documentation (this file)
```

---

## 4. Execution & Retraining Instructions

### 4.1 Retraining the Model & Combining Datasets
To rebuild the master dataset and retrain model weights:
```powershell
# Step 1: Combine conversion and auction reports into master dataset
python data_cleaner.py

# Step 2: Train Hierarchical MixedLM with regional calibration and save price_model.joblib
python model_engine.py

# Step 3: Run multi-branch automated test suite
python test_upload_and_branch.py

# Step 4: Run edge-case verification
python verify_all.py
```

### 4.2 Launching the Web Dashboard
Execute in PowerShell:
```powershell
streamlit run app.py
```
Or double-click `run_app.bat`. The web application will launch at `http://localhost:8501`.

---

## 5. Key Executive Dashboard Features

1. **Multi-Branch Selector:** Filter forecasts, historical comps, and inventory by:
   - `All Branches (Combined)` (3,328 records nationwide)
   - `TN` (Tamil Nadu - 1,896 records)
   - `KL` (Kerala - 988 records)
   - `AP/TS` (Andhra Pradesh & Telangana - 444 records, unified)
2. **5-Driver Valuation Breakdown:**
   - *Market Baseline:* Median price of comparable sales in the selected branch.
   - *Age Depreciation:* Compound depreciation over the vehicle's lifespan.
   - *Mileage Adjustment:* Elasticity penalty/reward relative to 50,000 km standard.
   - *Model Effect (BLUP):* Empirical Bayes shrinkage adjusted vehicle model premium.
   - *Branch Calibration:* Regional price elasticity (+25.2% TN, +17.0% KL, +0.0% AP/TS).
3. **Download Combined Master Excel:** Instant one-click export of `COMBINED_JAN26_AUG26_ALL_BRANCHES.xlsx` with dedicated sheets for each state branch and automated pivot summaries.
4. **Historical Comparable Sales Explorer:** Instant searchable table of past transactions filtered by state and branch.

---
*Developed for TVS Certified Private Limited — Multi-Branch Pricing Intelligence Platform.*
