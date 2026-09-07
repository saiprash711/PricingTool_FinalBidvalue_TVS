"""
data_cleaner.py
Preprocesses and cleans the 8-month AP/Telangana auction dataset from JAN26_AUG26_REPORT.xlsx.
Resolves data entry mislabels, spelling variants, and trim consolidations.
"""

import pandas as pd
import numpy as np

def clean_vehicle_data(excel_path="JAN26_AUG26_REPORT.xlsx"):
    sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
    
    all_dfs = []
    for s in sheets:
        sheet_df = pd.read_excel(excel_path, sheet_name=s)
        sheet_df['SOURCE_MONTH'] = s
        all_dfs.append(sheet_df)
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Standardize string fields
    df['MAKE'] = df['MAKE'].astype(str).str.strip()
    df['MODEL'] = df['MODEL'].astype(str).str.strip()
    df['VEH NO'] = df['VEH NO'].astype(str).str.strip()
    
    # 1. Correct specific mislabeled rows identified in data audit
    # AP37DE4995 is a Volkswagen Vento mislabeled as Hyundai
    df.loc[df['VEH NO'] == 'AP37DE4995', 'MAKE'] = 'Volkswagen'
    
    # AP16BU7806 is a Maruti Wagon-R mislabeled as Hyundai
    df.loc[df['VEH NO'] == 'AP16BU7806', 'MAKE'] = 'Maruti'
    
    # AP16BQ7777 is a Hyundai i10 mislabeled as Maruti
    df.loc[df['VEH NO'] == 'AP16BQ7777', 'MAKE'] = 'Hyundai'
    
    # TS07FP7214 is a Nissan Datsun mislabeled as Renault Datson
    df.loc[df['VEH NO'] == 'TS07FP7214', 'MAKE'] = 'Nissan'
    df.loc[df['VEH NO'] == 'TS07FP7214', 'MODEL'] = 'Datsun'
    
    # AP16CG0018 is a Maruti Ritz (2013, 196k km) entered as Riaz
    df.loc[df['VEH NO'] == 'AP16CG0018', 'MODEL'] = 'Ritz'
    
    # 2. Spelling, Casing and Trim Normalization Mapping
    model_mapping = {
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
        # Trim & generation consolidations as recommended
        'V Brezza': 'Vitara Brezza',
        'Brezza': 'Vitara Brezza',
        'Swift Dzire': 'Dzire',
        'i20 Asta': 'i20',
        'Getz Prime': 'Getz',
        'Celerio-X': 'Celerio',
    }
    df['MODEL'] = df['MODEL'].replace(model_mapping)
    
    # 3. Clean numeric fields
    df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce').astype(int)
    df['ODO METER'] = pd.to_numeric(df['ODO METER'], errors='coerce').astype(float)
    df['FINAL BID VALUE'] = pd.to_numeric(df['FINAL BID VALUE'], errors='coerce').astype(float)
    df['TOTAL INCOME'] = pd.to_numeric(df['TOTAL INCOME'], errors='coerce').astype(float)
    
    # 4. Feature Engineering
    # Reference year is 2026 (data spans Jan-Aug 2026)
    df['age'] = (2026 - df['YEAR']).clip(lower=0.5)
    df['log_price'] = np.log(df['FINAL BID VALUE'])
    df['log_odo'] = np.log(df['ODO METER'].clip(lower=500))
    df['make_model'] = df['MAKE'] + ' | ' + df['MODEL']
    
    return df

if __name__ == "__main__":
    df_cleaned = clean_vehicle_data()
    output_path = "cleaned_vehicle_data.csv"
    df_cleaned.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully saved to {output_path}")
    print(f"Total Rows: {len(df_cleaned)}")
    print(f"Unique Makes: {df_cleaned['MAKE'].nunique()}")
    print(f"Unique Models: {df_cleaned['MODEL'].nunique()}")
    print(f"Unique Make-Model Combinations: {df_cleaned['make_model'].nunique()}")
