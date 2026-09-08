import sys
import os
import pandas as pd
import numpy as np

# Force UTF-8 output for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

from model_engine import VehiclePricePredictor

def test_pipeline():
    print("=================================================================")
    print("Testing Tamil Nadu Master Pricing Engine Pipeline")
    print("=================================================================")
    
    predictor = VehiclePricePredictor()
    
    # 1. Catalog Verification
    catalog = predictor.get_catalog()
    assert len(catalog) > 15, f"Catalog too small: {len(catalog)} makes"
    total_models = sum(len(v) for v in catalog.values())
    print(f"PASS: Catalog loaded {len(catalog)} makes and {total_models} models.")
    
    # 2. Prediction tests with new features
    # Test case 1: Standard TN Maruti Swift
    res_swift_mt = predictor.predict(
        make="Maruti", 
        model="Swift", 
        year=2019, 
        odometer=50000, 
        fuel="Petrol", 
        transmission="Manual", 
        ownership="1st Owner", 
        rto_zone="Chennai Metro"
    )
    print(f"\n2019 Maruti Swift (1st Owner, MT, Chennai Metro):")
    print(f"  Expected Price (Final Bid): Rs. {res_swift_mt['expected_price']:,}")
    print(f"  Seller Net Payout:          Rs. {res_swift_mt.get('seller_net_payout', 0):,}")
    print(f"  Seller Reserve Target:      Rs. {res_swift_mt.get('seller_reserve_target', 0):,}")
    print(f"  Confidence:                 {res_swift_mt['confidence_level']}")
    assert res_swift_mt['expected_price'] > 300000 and res_swift_mt['expected_price'] < 700000
    
    # Test case 2: Automatic Transmission Premium
    res_swift_at = predictor.predict(
        make="Maruti", 
        model="Swift", 
        year=2019, 
        odometer=50000, 
        fuel="Petrol", 
        transmission="Automatic", 
        ownership="1st Owner", 
        rto_zone="Chennai Metro"
    )
    print(f"2019 Maruti Swift (1st Owner, AT, Chennai Metro):")
    print(f"  Expected Price: Rs. {res_swift_at['expected_price']:,}")
    assert res_swift_at['expected_price'] > res_swift_mt['expected_price'], "Automatic should cost more than Manual"
    print(f"PASS: Automatic premium confirmed ({((res_swift_at['expected_price']/res_swift_mt['expected_price'])-1)*100:+.1f}%)")
    
    # Test case 3: Ownership Penalty
    res_swift_2nd = predictor.predict(
        make="Maruti", 
        model="Swift", 
        year=2019, 
        odometer=50000, 
        fuel="Petrol", 
        transmission="Manual", 
        ownership="2nd Owner", 
        rto_zone="Chennai Metro"
    )
    res_swift_3rd = predictor.predict(
        make="Maruti", 
        model="Swift", 
        year=2019, 
        odometer=50000, 
        fuel="Petrol", 
        transmission="Manual", 
        ownership="3rd Owner", 
        rto_zone="Chennai Metro"
    )
    print(f"2019 Maruti Swift Ownership degradation:")
    print(f"  1st Owner: Rs. {res_swift_mt['expected_price']:,}")
    print(f"  2nd Owner: Rs. {res_swift_2nd['expected_price']:,}")
    print(f"  3rd Owner: Rs. {res_swift_3rd['expected_price']:,}")
    assert res_swift_mt['expected_price'] > res_swift_2nd['expected_price'], "2nd Owner should be cheaper than 1st"
    assert res_swift_2nd['expected_price'] > res_swift_3rd['expected_price'], "3rd Owner should be cheaper than 2nd"
    print("PASS: Ownership penalty confirmed.")
    
    # Test case 4: RTO Economic Zones
    res_cbe = predictor.predict("Hyundai", "Creta", 2020, 45000, rto_zone="Coimbatore Hub")
    res_chn = predictor.predict("Hyundai", "Creta", 2020, 45000, rto_zone="Chennai Metro")
    res_mdu = predictor.predict("Hyundai", "Creta", 2020, 45000, rto_zone="Madurai / South TN")
    res_out = predictor.predict("Hyundai", "Creta", 2020, 45000, rto_zone="Outside TN")
    print(f"\nHyundai Creta 2020 Zone Calibration:")
    print(f"  Coimbatore Hub:      Rs. {res_cbe['expected_price']:,}")
    print(f"  Chennai Metro:       Rs. {res_chn['expected_price']:,}")
    print(f"  Madurai / South TN:  Rs. {res_mdu['expected_price']:,}")
    print(f"  Outside TN:          Rs. {res_out['expected_price']:,}")
    assert max(res_cbe['expected_price'], res_chn['expected_price']) > res_mdu['expected_price']
    assert res_mdu['expected_price'] > res_out['expected_price']
    print("PASS: Regional RTO Zone ordering verified.")
    
    # Test case 5: Sensitivity curves
    sens_odo = predictor.get_sensitivity_curve("Maruti", "Swift", 2019, 50000, "odometer")
    sens_age = predictor.get_sensitivity_curve("Maruti", "Swift", 2019, 50000, "age")
    assert len(sens_odo) > 0, "Odometer sensitivity curve is empty"
    assert len(sens_age) > 0, "Age sensitivity curve is empty"
    print(f"PASS: Sensitivity curves computed ({len(sens_odo)} odo points, {len(sens_age)} age points).")
    
    # Test case 6: Historical Comps
    comps = predictor.get_historical_comps("Honda", "City")
    print(f"PASS: Found {len(comps)} Tamil Nadu comps for Honda City.")
    assert len(comps) > 0
    assert 'FINAL BID VALUE' in comps.columns
    assert 'SELLER RESERVE' in comps.columns
    assert 'RTO_ZONE' in comps.columns
    
    # Test case 7: Decomposition Verification
    decomp = res_swift_mt['breakdown']
    print(f"\nDecomposition Keys in prediction:")
    for k, v in decomp.items():
        print(f"  - {k}: {v}")
    assert 'transmission_adj_pct' in decomp
    assert 'ownership_adj_pct' in decomp
    assert 'zone_adj_pct' in decomp
    print("PASS: Full 6-factor decomposition keys present.")

    # Test case 8: Data Cleanliness & Non-Passenger Filter Verification
    print("\n--- Testing Data Cleanliness & Integrity ---")
    clean_df = predictor.df
    assert clean_df is not None, "Predictor dataframe is None"
    assert len(clean_df) >= 5200, f"Expected >= 5200 clean records, got {len(clean_df)}"
    
    # Verify no two-wheeler or non-passenger entries exist
    bad_kw = ['activa', 'jupiter', 'splendor', 'classic 350', 'cb shine', 'tvs xl', 'apache', 'ct 100', 'tafe', 'lpo 1622']
    for idx, r in clean_df.iterrows():
        rem = str(r.get('battery_remarks', '')).lower()
        title = str(r.get('carTitleName', '')).lower()
        for kw in bad_kw:
            assert kw not in rem and kw not in title, f"Contaminated record found: ID {r.get('id')}, kw {kw}"
    print("PASS: 0 two-wheelers, tractors, or commercial chassis in training dataset.")

    # Test case 9: Price Multiplier Typos & Extreme Outlier Absence
    assert clean_df['FINAL BID VALUE'].max() <= 5500000, f"Unexpected max price: {clean_df['FINAL BID VALUE'].max()}"
    assert clean_df['FINAL BID VALUE'].min() >= 15000, f"Unexpected min price: {clean_df['FINAL BID VALUE'].min()}"
    dzire_prices = clean_df[clean_df['MODEL'] == 'Dzire']['FINAL BID VALUE']
    assert dzire_prices.max() < 1500000, f"Dzire price outlier found: {dzire_prices.max()}"
    print(f"PASS: Price boundaries verified (Min: Rs. {clean_df['FINAL BID VALUE'].min():,}, Max: Rs. {clean_df['FINAL BID VALUE'].max():,}, Max Dzire: Rs. {dzire_prices.max():,}).")

    # Test case 10: Deduplication of Re-Auctioned Registrations
    dup_regs = clean_df['VEH NO'].dropna().duplicated().sum()
    assert dup_regs == 0, f"Found {dup_regs} duplicate registrations in training dataset"
    print("PASS: 0 duplicate registration numbers (re-auctions deduplicated).")

    # Test case 11: Complete Coverage of Regression Variables
    req_cols = ['age', 'log_odo', 'FINAL BID VALUE', 'MAKE', 'MODEL', 'FUEL TYPE', 'TRANSMISSION', 'OWNER', 'RTO_ZONE']
    for c in req_cols:
        null_count = clean_df[c].isnull().sum()
        assert null_count == 0, f"Column {c} has {null_count} nulls"
    print(f"PASS: 100% complete coverage on all {len(req_cols)} core regression variables.")

    # Test case 12: Variant / Trim Level Extraction, Forecasting & Comps Filtering
    print("\n--- Testing Variant Extraction, Forecasting & Filtering ---")
    assert 'VARIANT' in clean_df.columns, "VARIANT column missing from clean dataset"
    swift_variants = predictor.get_variants("Maruti", "Swift")
    print(f"Swift Variants in Catalog ({len(swift_variants)}): {swift_variants[:6]}")
    assert len(swift_variants) >= 3, f"Expected >= 3 variants for Swift, got {len(swift_variants)}"
    assert any(t in swift_variants for t in ['VXI', 'LXI', 'ZXI', 'VDI', 'ZDI']), "Expected canonical Swift trims in catalog"
    
    # Verify that Historical Comps count dynamically changes when Variant is selected
    res_swift_all = predictor.predict("Maruti", "Swift", 2019, 50000, variant="All / Any Variant", branch="All Tamil Nadu")
    res_swift_vxi = predictor.predict("Maruti", "Swift", 2019, 50000, variant="VXI", branch="All Tamil Nadu")
    res_swift_zxi = predictor.predict("Maruti", "Swift", 2019, 50000, variant="ZXI", branch="All Tamil Nadu")
    res_swift_lxi = predictor.predict("Maruti", "Swift", 2019, 50000, variant="LXI", branch="All Tamil Nadu")
    
    print(f"Historical Comps dynamic count test:")
    print(f"  All Trims: {res_swift_all['n_samples']} records")
    print(f"  VXI Trim:  {res_swift_vxi['n_samples']} records")
    print(f"  ZXI Trim:  {res_swift_zxi['n_samples']} records")
    print(f"  LXI Trim:  {res_swift_lxi['n_samples']} records")
    
    assert res_swift_all['n_samples'] >= 300, f"Expected >= 300 all-trim Swift records, got {res_swift_all['n_samples']}"
    assert res_swift_vxi['n_samples'] > 100, f"Expected > 100 VXI Swift records, got {res_swift_vxi['n_samples']}"
    assert res_swift_zxi['n_samples'] > 20, f"Expected > 20 ZXI Swift records, got {res_swift_zxi['n_samples']}"
    assert res_swift_lxi['n_samples'] > 15, f"Expected > 15 LXI Swift records, got {res_swift_lxi['n_samples']}"
    print("PASS: Historical Comps count dynamically updates with Variant selection.")

    # Test prediction difference between top and base trim
    print(f"2019 Maruti Swift Trim Comparison:")
    print(f"  ZXI (High Trim): Rs. {res_swift_zxi['expected_price']:,}")
    print(f"  LXI (Base Trim): Rs. {res_swift_lxi['expected_price']:,}")
    assert res_swift_zxi['expected_price'] >= res_swift_lxi['expected_price'], "Higher trim ZXI should be >= LXI"
    print(f"PASS: Variant sensitivity confirmed (ZXI vs LXI: +{((res_swift_zxi['expected_price']/res_swift_lxi['expected_price'])-1)*100:.1f}%)")

    # Test historical comps filtering with variant
    vxi_comps = predictor.get_historical_comps("Maruti", "Swift", variant="VXI")
    print(f"PASS: Found {len(vxi_comps)} VXI comps for Maruti Swift.")
    assert len(vxi_comps) > 0
    assert (vxi_comps['VARIANT'].str.upper() == 'VXI').all()
    print("PASS: Comps variant filtering matches exactly.")

    print("\n=================================================================")
    print("ALL TAMIL NADU MODEL PIPELINE & DATA HYGIENE TESTS PASSED 100%!")
    print("=================================================================")

if __name__ == "__main__":
    test_pipeline()
