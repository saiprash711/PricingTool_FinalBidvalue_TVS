import io
import pandas as pd
from data_cleaner import (
    parse_template_data, 
    generate_sample_template_excel, 
    generate_sample_template_csv,
    detect_branch,
    clean_vehicle_data
)
from model_engine import VehiclePricePredictor

def run_tests():
    print("=" * 60)
    print("TEST 1: Sample Template Generation & Parsing")
    print("=" * 60)
    
    # 1. Test Excel Template Generation & Parse
    excel_bytes = generate_sample_template_excel()
    excel_io = io.BytesIO(excel_bytes)
    excel_io.name = "sample_test.xlsx"
    df_excel, summary_excel = parse_template_data(excel_io)
    print(f"Excel template parsed: {len(df_excel)} rows, branches: {summary_excel['branches']}")
    assert len(df_excel) == 4, f"Expected 4 rows, got {len(df_excel)}"
    assert 'BRANCH' in df_excel.columns, "BRANCH column missing"
    assert 'TN' in summary_excel['branches']
    assert 'KL' in summary_excel['branches']
    print("PASS: Excel multi-branch template generation and parsing verified.")

    # 2. Test CSV Template Generation & Parse
    csv_bytes = generate_sample_template_csv()
    csv_io = io.BytesIO(csv_bytes)
    csv_io.name = "sample_test.csv"
    df_csv, summary_csv = parse_template_data(csv_io)
    print(f"CSV template parsed: {len(df_csv)} rows, branches: {summary_csv['branches']}")
    assert len(df_csv) == 4, f"Expected 4 rows, got {len(df_csv)}"
    print("PASS: CSV multi-branch template generation and parsing verified.")

    print("\n" + "=" * 60)
    print("TEST 2: Master Multi-Branch Dataset (TN, KL, AP/TS)")
    print("=" * 60)
    
    df_base, summary_base = parse_template_data("cleaned_vehicle_data.csv")
    print(f"Total master rows: {len(df_base):,}")
    print("State distribution:", summary_base['states'])
    print("Branch distribution:", summary_base['branches'])
    assert len(df_base) >= 3300, f"Expected >= 3300 rows, got {len(df_base)}"
    assert 'TN' in summary_base['states']
    assert 'KL' in summary_base['states']
    assert 'AP/TS' in summary_base['states']
    print("PASS: Master multi-branch dataset verified with complete regional distribution.")

    print("\n" + "=" * 60)
    print("TEST 3: Predictor Multi-Branch & Regional Calibration")
    print("=" * 60)
    
    predictor = VehiclePricePredictor()
    
    # Check catalog by branch
    cat_all = predictor.get_catalog()
    cat_tn = predictor.get_catalog('TN')
    cat_kl = predictor.get_catalog('KL')
    cat_apts = predictor.get_catalog('AP/TS')
    print(f"Makes: All={len(cat_all)}, TN={len(cat_tn)}, KL={len(cat_kl)}, AP/TS={len(cat_apts)}")
    assert len(cat_all) >= len(cat_tn)
    assert len(cat_all) >= len(cat_kl)
    
    # Comps filtering by branch
    comps_all = predictor.get_historical_comps("Honda", "City")
    comps_tn = predictor.get_historical_comps("Honda", "City", branch="TN")
    comps_kl = predictor.get_historical_comps("Honda", "City", branch="KL")
    comps_apts = predictor.get_historical_comps("Honda", "City", branch="AP/TS")
    print(f"Honda City Comps: All={len(comps_all)}, TN={len(comps_tn)}, KL={len(comps_kl)}, AP/TS={len(comps_apts)}")
    assert len(comps_all) >= len(comps_tn)
    assert len(comps_all) >= len(comps_apts)
    assert 'BRANCH' in comps_all.columns
    print("PASS: Historical comps regional filtering verified.")

    # Prediction calibration with branch
    pred_all = predictor.predict("Honda", "City", 2017, 75000)
    pred_tn = predictor.predict("Honda", "City", 2017, 75000, branch="TN")
    pred_kl = predictor.predict("Honda", "City", 2017, 75000, branch="KL")
    pred_apts = predictor.predict("Honda", "City", 2017, 75000, branch="AP/TS")
    
    print(f"Expected price (All):    Rs. {pred_all['expected_price']:,} (Adj: {pred_all['breakdown']['regional_calibration_pct']:+0.1f}%)")
    print(f"Expected price (TN):     Rs. {pred_tn['expected_price']:,} (Adj: {pred_tn['breakdown']['regional_calibration_pct']:+0.1f}%)")
    print(f"Expected price (KL):     Rs. {pred_kl['expected_price']:,} (Adj: {pred_kl['breakdown']['regional_calibration_pct']:+0.1f}%)")
    print(f"Expected price (AP/TS):  Rs. {pred_apts['expected_price']:,} (Adj: {pred_apts['breakdown']['regional_calibration_pct']:+0.1f}%)")
    
    assert pred_tn['expected_price'] > pred_apts['expected_price'], "TN expected price should reflect regional premium over AP/TS"
    assert pred_kl['expected_price'] > pred_apts['expected_price'], "KL expected price should reflect regional premium over AP/TS"
    print("PASS: Regional branch calibration effects verified.")

    print("\n" + "=" * 60)
    print("TEST 4: Predictor Dynamic Dataset Update")
    print("=" * 60)
    predictor.update_dataset(df_csv)
    comps_updated = predictor.get_historical_comps("Honda", "City")
    print(f"Comps count after updating with 4-row template: {len(comps_updated)}")
    assert len(comps_updated) == 1
    # Restore predictor
    predictor.update_dataset(df_base)
    comps_restored = predictor.get_historical_comps("Honda", "City")
    assert len(comps_restored) >= 65
    print("PASS: Predictor dynamic dataset updating verified.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
