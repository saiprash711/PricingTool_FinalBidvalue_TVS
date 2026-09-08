"""
data_cleaner.py
Preprocesses and cleans the multi-branch vehicle auction dataset (TN, KL, AP/TS).
Supports multi-sheet Excel files (JAN26_AUG26_REPORT.xlsx, Jan to july 2026 Conversion Data.xlsx) and CSV files.
Resolves data entry mislabels, spelling variants, trim consolidations,
extracts and assigns regional and city branches (TN, KL, Hyderabad, Vijayawada, Vizag, AP/TS),
and provides automated combination and export of master multi-branch auction reports.
"""

import io
import os
import re
import ctypes
import tempfile
import pandas as pd
import numpy as np

# Known spelling, casing, and trim normalization dictionary
MODEL_MAPPING = {
    # Maruti
    'SWIFTDZIRE': 'Dzire',
    'Swift Dzire': 'Dzire',
    'Swiftdzire': 'Dzire',
    'SXFour': 'SX4',
    'SX 4': 'SX4',
    'V Brezza': 'Vitara Brezza',
    'Brezza': 'Vitara Brezza',
    'Vitara Brezza': 'Vitara Brezza',
    'Celerio-X': 'Celerio',
    'Astar': 'A-Star',
    'A-Star': 'A-Star',
    'M800': '800',
    'Maruti 800': '800',
    'ALTO 800': 'Alto 800',
    'Alto 800': 'Alto 800',
    'ALTO K10': 'Alto K10',
    'Alto K10': 'Alto K10',
    'SPRESSO': 'S-Presso',
    'S-Presso': 'S-Presso',
    'SCROSS': 'S-Cross',
    'S-Cross': 'S-Cross',
    'ZEN ESTILO': 'Zen Estilo',
    'Zen Estilo': 'Zen Estilo',
    'Wagonr': 'WagonR',
    'Wagon R': 'WagonR',

    # Hyundai
    'ITEN': 'i10',
    'I 10': 'i10',
    'I10': 'i10',
    'i10': 'i10',
    'GRAND I10': 'Grand i10',
    'Grand I10': 'Grand i10',
    'Grand i10': 'Grand i10',
    'GRAND I10 NIOS': 'Grand i10 Nios',
    'Grand i10 Nios': 'Grand i10 Nios',
    'I20': 'i20',
    'i20': 'i20',
    'i20 Asta': 'i20',
    'Getz Prime': 'Getz',

    # Honda
    'Wr-v': 'WR-V',
    'Wr-V': 'WR-V',
    'WR-V': 'WR-V',
    'Cr-V': 'CR-V',
    'CR-V': 'CR-V',
    'CRV': 'CR-V',
    'Br-V': 'BR-V',
    'BR-V': 'BR-V',
    'BRV': 'BR-V',
    'Brv': 'BR-V',
    'Br10': 'Brio',
    'Ameze': 'Amaze',

    # Mahindra
    'Xuv500': 'XUV500',
    'XUV 500': 'XUV500',
    'Xuv300': 'XUV300',
    'XUV 300': 'XUV300',
    'Xuv700': 'XUV700',
    'XUV 700': 'XUV700',
    'Kuv 100': 'KUV100',
    'KUV 100': 'KUV100',
    'Tuv 300': 'TUV300',
    'TUV 300': 'TUV300',
    'TUV': 'TUV300',
    'Bolero Pikup': 'Bolero Pikup',
    'BoleroNeo': 'Bolero Neo',
    'NUVO SPORT': 'NuvoSport',
    'NUVOSPORT': 'NuvoSport',

    # Toyota
    'CRYSTA': 'Innova Crysta',
    'ETIOS LIVA': 'Etios Liva',
    'URBAN CRUISERHYRYDER': 'Urban Cruiser Hyryder',
    'URBAN CRUISER': 'Urban Cruiser',

    # Tata
    'SUMO VICTA': 'Sumo Victa',
    'HEXA': 'Hexa',

    # Ford
    'Eco Sport': 'EcoSport',
    'Ecosport': 'EcoSport',
    'ECOSPORT': 'EcoSport',
    'FIESTACLASSIC': 'Fiesta Classic',

    # Datsun
    'Datson': 'Datsun',
    'DATSON GO': 'GO',
    'DATSUN GO': 'GO',
    'Redigo': 'redi-GO',
    'REDI-GO': 'redi-GO',

    # Mercedes-Benz
    'EClass': 'E-Class',
    'C class': 'C-Class',
    'Aclass': 'A-Class',
    'A200': 'A-Class',
    '520D': '5 Series',
    '530I': '5 Series',

    # Skoda
    'Octovia': 'Octavia',
    'Rapid': 'Rapid',
    'Fabia': 'Fabia',
    'Superb': 'Superb',
    'Slavia': 'Slavia',
    'Kushaq': 'Kushaq',

    # Others
    'Beat': 'Beat',
    'Optra Magnum': 'Optra Magnum',
    'OPTRA MAGNUM': 'Optra Magnum',
}

MAKE_MAPPING = {
    'MARUTI SUZUKI': 'Maruti',
    'MARUTI': 'Maruti',
    'Maruti Suzuki': 'Maruti',
    'mercedes benz': 'Mercedes-Benz',
    'MERCEDES BENZ': 'Mercedes-Benz',
    'Mercedes Benz': 'Mercedes-Benz',
    'MG Motors': 'MG',
    'MG MOTORS': 'MG',
    'DATSUN': 'Datsun',
    'Datsun': 'Datsun',
    'VOLKSWAGEN': 'Volkswagen',
    'Volkswagen': 'Volkswagen',
    'HINDUSTAN MOTOR FINANCE CORPORATION LIMITED': 'Mitsubishi',
    'CHEVROLET': 'Chevrolet',
    'Chevrolet': 'Chevrolet',
    'HYUNDAI': 'Hyundai',
    'Hyundai': 'Hyundai',
    'HONDA': 'Honda',
    'Honda': 'Honda',
    'TOYOTA': 'Toyota',
    'Toyota': 'Toyota',
    'TATA': 'Tata',
    'Tata': 'Tata',
    'FORD': 'Ford',
    'Ford': 'Ford',
    'RENAULT': 'Renault',
    'Renault': 'Renault',
    'MAHINDRA': 'Mahindra',
    'Mahindra': 'Mahindra',
    'NISSAN': 'Nissan',
    'Nissan': 'Nissan',
    'SKODA': 'Skoda',
    'Skoda': 'Skoda',
    'KIA': 'Kia',
    'Kia': 'Kia',
    'FIAT': 'Fiat',
    'Fiat': 'Fiat',
    'JEEP': 'Jeep',
    'Jeep': 'Jeep',
    'BMW': 'BMW',
    'CITROEN': 'Citroen',
    'Citroen': 'Citroen',
    'AUDI': 'Audi',
    'Audi': 'Audi',
    'ISUZU': 'Isuzu',
    'Isuzu': 'Isuzu',
    'MITSUBISHI': 'Mitsubishi',
    'Mitsubishi': 'Mitsubishi',
}

# Specific vehicle ID overrides verified through engine displacement and remarks
MODEL_OVERRIDES = {
    1340: ('Maruti', 'Dzire'),      # Labeled Isuzu MUX, remarks MARUTI DZIRE VXI AMT, 1197cc
    1341: ('Maruti', 'Dzire'),      # Labeled Kia Sonet, remarks MARUTI DZIRE VXI AMT, 1197cc
    1230: ('Honda', 'City'),        # Labeled Honda ZR-V, remarks HONDA CITY 1.5 EMT, 1498cc
    2571: ('Mahindra', 'XUV500'),   # Missing model, remarks MAHXUV500RFWDW6BSIV, 2179cc
    4080: ('Hyundai', 'Grand i10'), # Missing model, remarks Grand i10 NIOS, 1197cc
    4927: ('Volkswagen', 'Polo'),   # Missing model, 1199cc Diesel 2012
    3039: ('Honda', 'City'),        # Missing model, 1497cc Petrol 2005
    4776: ('Hyundai', 'Grand i10'), # Missing model, 1197cc Petrol 2015
    7819: ('Hyundai', 'i10'),       # Labeled i20, remarks HYUNDAI MAGNA I10 1.2GL, offered 145k
}

def clean_make_name(val):
    s = str(val).strip()
    if s in MAKE_MAPPING:
        return MAKE_MAPPING[s]
    return s.title()

def clean_model_name(val):
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', '']:
        return 'General'
    if s in MODEL_MAPPING:
        return MODEL_MAPPING[s]
    acronyms = {'WR-V', 'CR-V', 'BR-V', 'XUV500', 'XUV300', 'XUV700', 'TUV300', 'KUV100', 'S-CROSS', 'GLC', 'EQS', 'Q2', 'Q3', 'A4'}
    if s.upper() in acronyms:
        return s.upper()
    s_title = s.title()
    if s_title in MODEL_MAPPING:
        return MODEL_MAPPING[s_title]
    return s_title

# Standard template column aliases for flexible matching across multi-branch sheets
COLUMN_ALIASES = {
    'SL NO': ['SL NO', 'SLNO', 'SL.NO', 'S.NO', 'SNO', 'ID'],
    'FY': ['FY', 'FINANCIAL YEAR', 'FIN YEAR'],
    'DATE': ['DATE', 'AUCTION DATE', 'SALE DATE', 'TRANSACTION DATE'],
    'MONTH': ['MONTH', 'AUCTION MONTH', 'SOURCE_MONTH', 'SALE MONTH'],
    'STATE': ['STATE', 'REGION', 'STATE NAME', 'ZONE'],
    'BRANCH': ['BRANCH', 'BRANCH NAME', 'BRANCH_NAME', 'LOCATION', 'CENTRE', 'CITY'],
    'VEH NO': ['VEH NO', 'VEH_NO', 'VEHICLE NO', 'VEHICLE_NO', 'VEHICLE NUMBER', 'VEHICLE_NUMBER', 'REG NO', 'REGISTRATION NO'],
    'MAKE': ['MAKE', 'BRAND', 'MANUFACTURER'],
    'MODEL': ['MODEL', 'VEHICLE MODEL', 'MODEL NAME'],
    'YEAR': ['YEAR', 'REG YEAR', 'REGISTRATION YEAR', 'MFR YEAR', 'MFG YEAR'],
    'ODO METER': ['ODO METER', 'ODO_METER', 'ODOMETER', 'KM', 'KMS', 'KM RUN', 'KMS DRIVEN'],
    'FINAL BID VALUE': ['FINAL BID VALUE', 'FINAL_BID_VALUE', 'FINAL BID', 'BID VALUE', 'PRICE', 'SOLD PRICE', 'FINAL PRICE'],
    'SELLER NAME': ['SELLER NAME', 'SELLER_NAME', 'SELLER', 'DEALER'],
    'SELLER SEGMENT': ['SELLER SEGMENT', 'SELLER_SEGMENT', 'SEGMENT'],
    'BUYER NAME': ['BUYER NAME', 'BUYER_NAME', 'BUYER', 'CUSTOMER'],
    'TOTAL INCOME': ['TOTAL INCOME', 'TOTAL_INCOME', 'INCOME', 'MARGIN', 'COMMISSION'],
}

def safe_load_excel(file_or_path):
    """
    Safely opens an Excel workbook, falling back to a temporary file copy via Win32 CopyFileW
    if the workbook is exclusively locked by another application (e.g., Microsoft Excel).
    """
    if isinstance(file_or_path, str) and os.path.exists(file_or_path):
        try:
            return pd.ExcelFile(file_or_path)
        except PermissionError:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"temp_{os.path.basename(file_or_path)}")
            # Win32 CopyFileW succeeds even when Excel holds exclusive read/write handle
            copied = ctypes.windll.kernel32.CopyFileW(file_or_path, temp_path, False)
            if copied:
                return pd.ExcelFile(temp_path)
            raise
    return pd.ExcelFile(file_or_path)

def detect_branch(row):
    """
    Identifies the branch location.
    Branches supported:
    - TN (Tamil Nadu)
    - KL (Kerala)
    - AP/TS (Andhra Pradesh & Telangana - unifying Hyderabad, Vijayawada, Vizag)
    """
    # 1. Explicit Branch column if present and not generic
    for col in ['BRANCH', 'Branch', 'BRANCH NAME', 'Branch Name', 'LOCATION', 'Location', 'CITY', 'City']:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip()
            if val and val.lower() not in ['nan', 'none', '', 'all', 'all branches']:
                v_up = val.upper()
                if any(k in v_up for k in ['HYDERABAD', 'VIJAYAWADA', 'VIZAG', 'VISAKHAPATNAM', 'AP/TS']) or v_up in ['AP', 'TS', 'TG']:
                    return 'AP/TS'
                if v_up in ['TN', 'TAMIL NADU', 'TAMILNADU']:
                    return 'TN'
                if v_up in ['KL', 'KERALA']:
                    return 'KL'
                return val.title() if len(val) > 3 else val.upper()

    # 2. State-level branch designation
    state_val = str(row.get('STATE', '')).strip().upper()
    if state_val == 'TN':
        return 'TN'
    if state_val == 'KL':
        return 'KL'
    if state_val in ['AP/TS', 'AP', 'TS', 'TG']:
        return 'AP/TS'

    # 3. Derive from Seller Name
    seller = str(row.get('SELLER NAME', '')).strip().upper()
    if any(k in seller for k in ['HYDERABAD', 'VIJAYAWADA', 'VIZAG', 'VISAKHAPATNAM']):
        return 'AP/TS'

    # 4. Derive from Vehicle Registration RTO Prefix
    veh = str(row.get('VEH NO', '')).strip().upper()
    if veh.startswith('TN'):
        return 'TN'
    if veh.startswith('KL'):
        return 'KL'
    if veh.startswith(('TS', 'TG', 'AP')):
        return 'AP/TS'

    return 'AP/TS'

def normalize_columns(df):
    """Maps arbitrary column headers matching template aliases to standardized column names."""
    rename_dict = {}
    cols_upper = {str(c).strip().upper(): c for c in df.columns}
    
    for standard_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in cols_upper:
                rename_dict[cols_upper[alias]] = standard_col
                break
                
    return df.rename(columns=rename_dict)

def clean_and_standardize_df(df, reference_year=2026, drop_invalid_years=False):
    """
    Cleans, validates, and engineers features for a DataFrame adhering to the vehicle auction template.
    If drop_invalid_years is True, records with missing Year are removed (used for regression training).
    If False, records with missing Year are retained for historical search and inventory comps.
    """
    df = normalize_columns(df.copy())
    
    required_cols = ['MAKE', 'MODEL', 'FINAL BID VALUE']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Uploaded data is missing required template columns: {missing_cols}")

    # Standardize string fields
    df['MAKE'] = df['MAKE'].astype(str).str.strip().str.title().replace({'Maruti Suzuki': 'Maruti'})
    df['MODEL'] = df['MODEL'].astype(str).str.strip()
    
    if 'VEH NO' in df.columns:
        df['VEH NO'] = df['VEH NO'].astype(str).str.strip().str.upper()
    else:
        df['VEH NO'] = [f"TVS{i+1:04d}" for i in range(len(df))]

    if 'SELLER NAME' in df.columns:
        df['SELLER NAME'] = df['SELLER NAME'].astype(str).str.strip()
    else:
        df['SELLER NAME'] = 'TVS Certified'

    if 'BUYER NAME' in df.columns:
        df['BUYER NAME'] = df['BUYER NAME'].astype(str).str.strip()
    else:
        df['BUYER NAME'] = 'Auction Buyer'

    # State Normalization
    if 'STATE' in df.columns:
        df['STATE'] = df['STATE'].astype(str).str.strip().str.upper()
        df['STATE'] = df['STATE'].replace({'AP': 'AP/TS', 'TS': 'AP/TS', 'TG': 'AP/TS'})
    else:
        df['STATE'] = 'AP/TS'

    # Infer STATE from vehicle number if missing, invalid, or nan
    veh_series = df['VEH NO'].astype(str).str.strip().str.upper()
    df.loc[df['STATE'].isna() | df['STATE'].isin(['NAN', '', 'NONE', 'OTHER']), 'STATE'] = np.where(
        veh_series.str.startswith('TN'), 'TN',
        np.where(veh_series.str.startswith('KL'), 'KL', 'AP/TS')
    )

    if 'DATE' in df.columns:
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
    elif 'MONTH' in df.columns:
        month_map = {
            'Jan': '2026-01-15', 'jan': '2026-01-15', 'JAN': '2026-01-15',
            'Feb': '2026-02-15', 'Mar': '2026-03-15', 'Apr': '2026-04-15',
            'May': '2026-05-15', 'Jun': '2026-06-15', 'Jul': '2026-07-15',
            'Aug': '2026-08-15', 'AUG-26': '2026-08-15'
        }
        df['DATE'] = pd.to_datetime(df['MONTH'].map(month_map), errors='coerce')
    else:
        df['DATE'] = pd.Timestamp.now()

    # 1. Correct specific mislabeled rows from data audit
    df.loc[df['VEH NO'] == 'AP37DE4995', 'MAKE'] = 'Volkswagen'
    df.loc[df['VEH NO'] == 'AP16BU7806', 'MAKE'] = 'Maruti'
    df.loc[df['VEH NO'] == 'AP16BQ7777', 'MAKE'] = 'Hyundai'
    df.loc[df['VEH NO'] == 'TS07FP7214', 'MAKE'] = 'Nissan'
    df.loc[df['VEH NO'] == 'TS07FP7214', 'MODEL'] = 'Datsun'
    df.loc[df['VEH NO'] == 'AP16CG0018', 'MODEL'] = 'Ritz'

    # 2. Spelling, Casing, and Trim Normalization Mapping
    df['MODEL'] = df['MODEL'].apply(clean_model_name)

    # 3. Clean numeric fields
    if 'YEAR' in df.columns:
        df['YEAR'] = pd.to_numeric(df['YEAR'], errors='coerce')
    else:
        df['YEAR'] = np.nan

    if 'ODO METER' in df.columns:
        df['ODO METER'] = pd.to_numeric(df['ODO METER'], errors='coerce')
    else:
        df['ODO METER'] = 60000.0

    df['FINAL BID VALUE'] = pd.to_numeric(df['FINAL BID VALUE'], errors='coerce')
    
    if 'TOTAL INCOME' in df.columns:
        df['TOTAL INCOME'] = pd.to_numeric(df['TOTAL INCOME'], errors='coerce').fillna(10000.0)
    else:
        df['TOTAL INCOME'] = 10000.0

    # Filter out invalid or zero bid prices
    df = df[df['FINAL BID VALUE'] > 5000].copy()

    if drop_invalid_years:
        df = df.dropna(subset=['YEAR', 'ODO METER'])
        df = df[(df['YEAR'] >= 1995) & (df['YEAR'] <= reference_year)]
        df['YEAR'] = df['YEAR'].astype(int)

    # 4. Feature Engineering
    df['age'] = (reference_year - df['YEAR']).clip(lower=0.5)
    df['log_price'] = np.log(df['FINAL BID VALUE'])
    df['log_odo'] = np.log(df['ODO METER'].clip(lower=500.0))
    df['make_model'] = df['MAKE'] + ' | ' + df['MODEL']

    # 5. Branch Detection & Assignment
    df['BRANCH'] = df.apply(detect_branch, axis=1)

    return df

def find_sheet_header_and_parse(xl, sheet_name):
    """
    Scans the first 10 rows of a sheet to locate the true column header row,
    gracefully handling title banners, metadata, or notes at the top of worksheets.
    """
    df_peek = xl.parse(sheet_name, header=None, nrows=10)
    if df_peek.empty or len(df_peek.columns) < 2:
        return pd.DataFrame()
        
    header_idx = 0
    cue_words = ['MAKE', 'BRAND', 'MODEL', 'VEHICLE NUMBER', 'VEH NO', 'FINAL BID VALUE', 'BID VALUE', 'YEAR', 'ODOMETER']
    
    for idx, row in df_peek.iterrows():
        row_vals = [str(v).strip().upper() for v in row.values if pd.notna(v)]
        matches = sum(1 for cue in cue_words if any(cue in val for val in row_vals))
        if matches >= 2:
            header_idx = idx
            break
            
    df = xl.parse(sheet_name, header=header_idx)
    df = df.dropna(how='all')
    return df

def parse_template_data(file_or_path, reference_year=2026, drop_invalid_years=False):
    """
    Parses an uploaded file or file path (Excel or CSV).
    Handles multi-sheet Excel files, concatenates valid sheets,
    and returns (cleaned_df, summary_dict).
    """
    all_dfs = []
    sheet_names = []
    
    is_excel = False
    is_csv = False

    if isinstance(file_or_path, str):
        if file_or_path.lower().endswith(('.xlsx', '.xls')):
            is_excel = True
        elif file_or_path.lower().endswith('.csv'):
            is_csv = True
        else:
            is_excel = True
    else:
        fname = getattr(file_or_path, 'name', '').lower()
        if fname.endswith('.csv'):
            is_csv = True
        else:
            is_excel = True

    if is_excel:
        xl = safe_load_excel(file_or_path)
        sheet_names = xl.sheet_names
        for s in sheet_names:
            try:
                sdf = find_sheet_header_and_parse(xl, s)
                if sdf.empty or len(sdf.columns) < 3:
                    continue
                norm_cols = [str(c).upper().strip() for c in sdf.columns]
                has_make = any(k in norm_cols for k in ['MAKE', 'BRAND'])
                has_model = any(k in norm_cols for k in ['MODEL', 'VEHICLE MODEL'])
                if has_make or has_model:
                    sdf['SOURCE_MONTH'] = s
                    all_dfs.append(sdf)
            except Exception:
                continue

        if not all_dfs:
            raise ValueError("No valid sheets with vehicle auction data found in the uploaded Excel workbook.")
        raw_df = pd.concat(all_dfs, ignore_index=True)
    else:
        raw_df = pd.read_csv(file_or_path)
        sheet_names = ["CSV Data"]

    # Clean and standardize
    cleaned_df = clean_and_standardize_df(raw_df, reference_year=reference_year, drop_invalid_years=drop_invalid_years)

    summary = {
        'total_rows': len(cleaned_df),
        'raw_rows': len(raw_df),
        'sheet_names': sheet_names,
        'unique_makes': int(cleaned_df['MAKE'].nunique()),
        'unique_models': int(cleaned_df['MODEL'].nunique()),
        'unique_make_models': int(cleaned_df['make_model'].nunique()),
        'states': cleaned_df['STATE'].value_counts().to_dict(),
        'branches': cleaned_df['BRANCH'].value_counts().to_dict(),
        'total_volume': float(cleaned_df['FINAL BID VALUE'].sum()),
        'avg_price': float(cleaned_df['FINAL BID VALUE'].mean()),
        'median_price': float(cleaned_df['FINAL BID VALUE'].median()),
        'min_year': int(cleaned_df['YEAR'].dropna().min()) if cleaned_df['YEAR'].notna().any() else 2000,
        'max_year': int(cleaned_df['YEAR'].dropna().max()) if cleaned_df['YEAR'].notna().any() else 2026,
        'avg_odometer': float(cleaned_df['ODO METER'].dropna().mean()) if cleaned_df['ODO METER'].notna().any() else 0.0
    }

    return cleaned_df, summary

def combine_and_export_master_dataset(
    conversion_path="Jan to july 2026 Conversion Data.xlsx",
    report_path="JAN26_AUG26_REPORT.xlsx",
    output_csv="cleaned_vehicle_data.csv",
    output_xlsx="COMBINED_JAN26_AUG26_ALL_BRANCHES.xlsx"
):
    """
    Combines 'Jan to july 2026 Conversion Data.xlsx' (Jan-Jul across TN, KL, AP/TS)
    with 'JAN26_AUG26_REPORT.xlsx' (adding August AP/TS records).
    Saves the cleaned master dataset to cleaned_vehicle_data.csv and COMBINED_JAN26_AUG26_ALL_BRANCHES.xlsx.
    """
    dfs = []
    
    # 1. Parse Conversion Data if available
    if os.path.exists(conversion_path):
        try:
            xl_conv = safe_load_excel(conversion_path)
            if 'Conversion Data FY26 - FY27' in xl_conv.sheet_names:
                df_conv_raw = xl_conv.parse('Conversion Data FY26 - FY27', header=2)
            else:
                df_conv_raw = find_sheet_header_and_parse(xl_conv, xl_conv.sheet_names[0])
            
            df_conv = normalize_columns(df_conv_raw)
            if 'MONTH' in df_conv.columns:
                df_conv['SOURCE_MONTH'] = df_conv['MONTH']
            dfs.append(df_conv)
            print(f"Loaded {len(df_conv)} conversion records from {conversion_path}")
        except Exception as e:
            print(f"Warning loading {conversion_path}: {e}")

    # 2. Parse 8-Month Report if available (specifically to get August records)
    if os.path.exists(report_path):
        try:
            xl_rep = safe_load_excel(report_path)
            for s in xl_rep.sheet_names:
                sdf = xl_rep.parse(s)
                if sdf.empty or len(sdf.columns) < 3:
                    continue
                sdf = normalize_columns(sdf)
                if 'MAKE' in sdf.columns or 'MODEL' in sdf.columns:
                    sdf['SOURCE_MONTH'] = s
                    dfs.append(sdf)
            print(f"Loaded records from {report_path}")
        except Exception as e:
            print(f"Warning loading {report_path}: {e}")

    if not dfs:
        raise ValueError("Neither conversion file nor report file could be loaded.")

    combined_raw = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate: if same vehicle is present in the same month across sources, keep first
    if 'VEH NO' in combined_raw.columns:
        combined_raw['temp_veh'] = combined_raw['VEH NO'].astype(str).str.strip().str.upper()
        if 'SOURCE_MONTH' in combined_raw.columns:
            def _norm_month(m):
                match = re.search(r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)', str(m).upper().strip())
                return match.group(1) if match else str(m).upper().strip()
            combined_raw['temp_m'] = combined_raw['SOURCE_MONTH'].apply(_norm_month)
            combined_raw = combined_raw.drop_duplicates(subset=['temp_veh', 'temp_m'], keep='first')
            combined_raw = combined_raw.drop(columns=['temp_veh', 'temp_m'])
        else:
            combined_raw = combined_raw.drop_duplicates(subset=['temp_veh'], keep='first')
            combined_raw = combined_raw.drop(columns=['temp_veh'])

    # Standardize and clean
    master_df = clean_and_standardize_df(combined_raw, reference_year=2026, drop_invalid_years=False)

    # Assign sequential SL NO
    master_df['SL NO'] = range(1, len(master_df) + 1)

    # Reorder columns for clean presentation
    front_cols = ['SL NO', 'FY', 'DATE', 'SOURCE_MONTH', 'STATE', 'BRANCH', 'VEH NO', 'MAKE', 'MODEL', 'YEAR', 'ODO METER', 'FINAL BID VALUE', 'SELLER NAME', 'BUYER NAME', 'TOTAL INCOME', 'make_model', 'age', 'log_price', 'log_odo']
    ordered_cols = [c for c in front_cols if c in master_df.columns]
    remaining_cols = [c for c in master_df.columns if c not in ordered_cols]
    master_df = master_df[ordered_cols + remaining_cols]

    # Save to CSV
    master_df.to_csv(output_csv, index=False)
    print(f"Saved master dataset ({len(master_df)} rows) to {output_csv}")

    # Save to Multi-Sheet Master Excel Workbook
    try:
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            master_df.to_excel(writer, index=False, sheet_name='ALL_AUCTIONS')
            
            # State sheets
            for st_code in ['TN', 'KL', 'AP/TS']:
                st_subset = master_df[master_df['STATE'] == st_code]
                if len(st_subset) > 0:
                    safe_sheet_name = f"BRANCH_{st_code.replace('/', '_')}"
                    st_subset.to_excel(writer, index=False, sheet_name=safe_sheet_name)
                    
            # Pivot summary sheet
            summary_pivot = master_df.pivot_table(
                index=['STATE', 'BRANCH'],
                values=['FINAL BID VALUE', 'ODO METER'],
                aggfunc={'FINAL BID VALUE': ['count', 'mean', 'median'], 'ODO METER': 'mean'}
            )
            summary_pivot.to_excel(writer, sheet_name='BRANCH_SUMMARY')
        print(f"Saved master multi-branch Excel workbook to {output_xlsx}")
    except Exception as e:
        print(f"Note: Excel export skipped or encountered notice: {e}")

    return master_df

def clean_tn_master_data(
    excel_path="masterdata_TN.xlsx", 
    combined_path="COMBINED_JAN26_AUG26_ALL_BRANCHES.xlsx",
    output_csv="cleaned_tn_vehicle_data.csv",
    update_excel=True,
    reference_year=2026
):
    """
    Thoroughly cleans, unifies, and standardizes Tamil Nadu vehicle auction records from BOTH
    'COMBINED_JAN26_AUG26_ALL_BRANCHES.xlsx' (verified conversion sales) and 'masterdata_TN.xlsx' (fit specs & bids).
    
    1. Filters strictly for Tamil Nadu (TN) registration prefixes and RTO economic zones.
    2. Filters out non-passenger vehicles (two-wheelers, mopeds, tractors, commercial chassis/buses/pickups).
    3. Resolves model misclassifications (e.g. disambiguating generic 'Mahindra XUV' into XUV500, XUV300, XUV700, 3XO; 'BMW D' into 3/5 Series, X3).
    4. Performs priority cross-dataset deduplication:
       - Uses verified conversion hammer prices from COMBINED for matching vehicles.
       - Enriches with detailed vehicle specifications (engine CC, trim, fuel, gearbox, photos) from masterdata.
       - Ingests remaining unique valid auction vehicles from masterdata (PostBid / closed PreBids).
    5. Corrects data-entry price multiplier errors (10x extra zero typos or missing zeros).
    6. Cleans and imputes odometers, years, fuels, gearboxes, owners, and RTO economic zones.
    7. Exports clean training CSV and multi-sheet master Excel workbook with full audit trail.
    """
    import json
    
    print("=== EXECUTING UNIFIED TAMIL NADU AUCTION DATA CLEANING PIPELINE ===")

    # --- 1. LOAD COMBINED_JAN26_AUG26_ALL_BRANCHES ---
    df_c_tn = pd.DataFrame()
    if os.path.exists(combined_path):
        try:
            print(f"Loading verified conversion records from {combined_path}...")
            xl_comb = safe_load_excel(combined_path)
            sheet_name = 'ALL_AUCTIONS' if 'ALL_AUCTIONS' in xl_comb.sheet_names else xl_comb.sheet_names[0]
            df_c = xl_comb.parse(sheet_name)
            df_c['clean_reg'] = df_c['VEH NO'].astype(str).str.strip().str.upper().str.replace(' ', '').str.replace('-', '')
            # Filter strictly for Tamil Nadu registrations and deduplicate
            df_c_tn = df_c[df_c['clean_reg'].str.startswith('TN')].copy()
            df_c_tn = df_c_tn.sort_values(['clean_reg', 'FINAL BID VALUE'], ascending=[True, False]).drop_duplicates(subset=['clean_reg'], keep='first').copy()
            print(f"Loaded {len(df_c_tn):,} unique TN auction conversion records from {combined_path}.")
        except Exception as e:
            print(f"Warning loading {combined_path}: {e}")

    # --- 2. LOAD MASTERDATA_TN ---
    if not os.path.exists(excel_path):
        if len(df_c_tn) == 0:
            raise FileNotFoundError(f"Neither {excel_path} nor {combined_path} could be found.")
        df_m_tn = pd.DataFrame()
    else:
        print(f"Loading Tamil Nadu master dataset from {excel_path}...")
        xl_mast = safe_load_excel(excel_path)
        raw_sheet = 'Raw_Source_Original' if 'Raw_Source_Original' in xl_mast.sheet_names else (
            'Sheet1' if 'Sheet1' in xl_mast.sheet_names else xl_mast.sheet_names[0]
        )
        raw_df = xl_mast.parse(raw_sheet)
        raw_df['clean_reg'] = raw_df['registration_number'].astype(str).str.strip().str.upper().str.replace(' ', '').str.replace('-', '')
        # Filter strictly for Tamil Nadu registrations
        df_m_tn = raw_df[raw_df['clean_reg'].str.startswith('TN')].copy()
        print(f"Loaded {len(df_m_tn):,} raw TN records from {excel_path} (sheet '{raw_sheet}').")

    # --- 3. AUDIT & PURGE NON-PASSENGER VEHICLES ---
    two_wheeler_kw = [
        'activa', 'jupiter', 'splendor', 'passion pro', 'passion', 'pulsar', 'apache', 'cb shine', 
        'bullet', 'royal enfield', 'classic 350', 'livo', 'ct 100', 'platina', 
        'access 125', 'tvs xl', 'scooty', 'glamour', 'dio', 'fascino', 'ray z',
        'r15', 'duke 200', 'duke', 'avenger', 'discover', 'tvs star',
        'radon', 'hunk', 'unicorn', 'cd deluxe', 'cd dawn', 'xtreme'
    ]
    commercial_kw = [
        'tafe mf', 'tractor', 'massey ferguson', 'john deere', 'swaraj', 'sonalika', 
        'lpo 1622', 'eicher bus', 'ashok leyland bus', 'bharatbenz', 'bolero pikup', 'pick up', 'pikup',
        'super carry', 'dost', 'bada dost', 'intra', 'ace gold', 'tata ace'
    ]

    def audit_non_passenger(row):
        rem = str(row.get('battery_remarks', '')).lower()
        title = str(row.get('carTitleName', '')).lower()
        mod = str(row.get('model_name', '')).lower()
        var = str(row.get('variant_id', '')).lower()
        
        # Guard: authentic passenger cars with 'shine' trim
        if 'shine' in rem and ('hector' in mod or 'c3' in mod or 'c5' in mod or 'citroen' in title or 'mg' in title):
            return True, ''
            
        for kw in two_wheeler_kw:
            if kw in rem or kw in title or kw in var:
                return False, f"Two-Wheeler: {kw.title()}"
                
        for kw in commercial_kw:
            if kw in rem or kw in title or kw in var or kw in mod:
                return False, f"Commercial/Tractor: {kw.title()}"
                
        try:
            d = json.loads(str(row.get('vechileDescription', '')))
            cc = float(d.get('cubicCapacity', 0))
            if 0 < cc < 500:
                for kw in ['shine', 'bike', 'scooter', 'moped', 'motorcycle']:
                    if kw in rem:
                        return False, f"Small CC ({cc}cc) + {kw.title()}"
        except Exception:
            pass
            
        return True, ''

    if len(df_m_tn) > 0:
        audit_res = [audit_non_passenger(r) for idx, r in df_m_tn.iterrows()]
        df_m_tn['is_passenger_car'] = [a[0] for a in audit_res]
        df_m_tn['non_car_reason'] = [a[1] for a in audit_res]
        df_m_pass = df_m_tn[df_m_tn['is_passenger_car']].copy()
        print(f"Non-passenger audit on masterdata: retained {len(df_m_pass):,} passenger cars, filtered {(~df_m_tn['is_passenger_car']).sum():,} non-cars.")
    else:
        df_m_pass = pd.DataFrame()

    # --- 4. MAKE & MODEL DISAMBIGUATION ---
    def resolve_master_make_model(row):
        car_id = row.get('id')
        if car_id in MODEL_OVERRIDES:
            return pd.Series([MODEL_OVERRIDES[car_id][0], MODEL_OVERRIDES[car_id][1]], index=['clean_make', 'clean_model'])
            
        mk_clean = clean_make_name(row.get('make_name', ''))
        raw_mod = str(row.get('model_name', '')).strip()
        title = str(row.get('carTitleName', '')).strip()
        rem = str(row.get('battery_remarks', '')).strip()
        var = str(row.get('variant_id', '')).strip()
        combined_text = f"{raw_mod} {title} {rem} {var}".upper()

        # Specific disambiguation: Mahindra XUV series
        if 'MAHINDRA' in mk_clean.upper() or 'XUV' in combined_text:
            if any(k in combined_text for k in ['XUV700', ' 700 ', 'AX7', 'AX5', 'AX3', 'MX5', 'MX3', 'MX1']):
                return pd.Series(['Mahindra', 'XUV700'], index=['clean_make', 'clean_model'])
            if '3XO' in combined_text or '3-XO' in combined_text:
                return pd.Series(['Mahindra', '3XO'], index=['clean_make', 'clean_model'])
            if 'XUV300' in combined_text or ' 300 ' in combined_text or 'XUV 300' in combined_text:
                return pd.Series(['Mahindra', 'XUV300'], index=['clean_make', 'clean_model'])
            if any(k in combined_text for k in ['XUV500', ' 500 ', 'XUV 500', 'W10', 'W11', 'W9', 'W8', 'W7', 'W6', 'W4']):
                return pd.Series(['Mahindra', 'XUV500'], index=['clean_make', 'clean_model'])
            if 'XUV' in combined_text:
                return pd.Series(['Mahindra', 'XUV500'], index=['clean_make', 'clean_model'])

        # Specific disambiguation: BMW series
        if 'BMW' in mk_clean.upper() or 'BMW' in combined_text:
            if '320D' in combined_text or '320 D' in combined_text or '3 SERIES' in combined_text:
                return pd.Series(['BMW', '3 Series'], index=['clean_make', 'clean_model'])
            if '520D' in combined_text or '520 D' in combined_text or '5 SERIES' in combined_text:
                return pd.Series(['BMW', '5 Series'], index=['clean_make', 'clean_model'])
            if 'X3' in combined_text:
                return pd.Series(['BMW', 'X3'], index=['clean_make', 'clean_model'])
            if 'X1' in combined_text:
                return pd.Series(['BMW', 'X1'], index=['clean_make', 'clean_model'])
            if 'X5' in combined_text:
                return pd.Series(['BMW', 'X5'], index=['clean_make', 'clean_model'])

        # Standard fallback clean
        s_mod = raw_mod
        if not s_mod or s_mod.lower() in ['nan', 'none', '']:
            if title.upper().startswith(mk_clean.upper()):
                s_mod = title[len(mk_clean):].strip()
            else:
                for common in ['Swift', 'City', 'i20', 'i10', 'Amaze', 'Ciaz', 'Ertiga', 'Baleno', 'Celerio', 'Polo', 'Verna', 'Creta', 'Innova', 'Fortuner', 'Duster', 'Kwid', 'Altroz', 'Nexon', 'Tigor', 'Tiago', 'Scorpio', 'Bolero']:
                    if common.lower() in combined_text.lower():
                        s_mod = common
                        break
        mod_clean = clean_model_name(s_mod)
        return pd.Series([mk_clean, mod_clean], index=['clean_make', 'clean_model'])

    if len(df_m_pass) > 0:
        mm_res = df_m_pass.apply(resolve_master_make_model, axis=1)
        df_m_pass['clean_make'] = mm_res['clean_make']
        df_m_pass['clean_model'] = mm_res['clean_model']

    # --- 5. EXTRACT VARIANT & SPECIFICATIONS ---
    trims_master = [
        'ZXI+', 'ZDI+', 'VXI+', 'ZXI (O)', 'VXI (O)', 'LXI (O)',
        'SX (O)', 'SX PLUS', 'E+', 'E PLUS', 'SPORTZ+', 'SPORTZ (O)', 'ASTA (O)',
        'RXT OPTION', 'RXT (O)', 'XZ+', 'XZ (O)', 'XT+', 'TITANIUM+', 'TITANIUM (O)',
        'LXI', 'VXI', 'ZXI', 'LDI', 'VDI', 'ZDI',
        'SIGMA', 'DELTA', 'ZETA', 'ALPHA',
        'ERA', 'MAGNA', 'SPORTZ', 'ASTA',
        'SV', 'VX', 'ZX', 'V', 'S', 'E',
        'SX', 'EX', 'EXI', 'GXI', 'VTEC',
        'G', 'GX', 'Z',
        'XE', 'XM', 'XT', 'XZ',
        'STD', 'RXE', 'RXL', 'RXT', 'RXZ', 'CLIMBER',
        'TRENDLINE', 'COMFORTLINE', 'HIGHLINE', 'GT',
        'AMBIENTE', 'TREND', 'TITANIUM',
        'ACTIVE', 'AMBITION', 'STYLE', 'ELEGANCE', 'MONTE CARLO',
        '5 STR AC', '7 STR', '5 SEATER', '7 SEATER',
        'CREATIVE', 'ADVENTURE', 'PURE', 'ACCOMPLISHED',
        'AX7', 'AX5', 'AX3', 'MX5', 'MX3', 'MX1', 'W11', 'W10', 'W9', 'W8', 'W7', 'W6', 'W4'
    ]

    def extract_variant(row):
        raw_v = str(row.get('variant_id', '')).strip().upper()
        raw_vn = str(row.get('variant_name', '')).strip().upper()
        rem = str(row.get('battery_remarks', '')).strip().upper()
        title = str(row.get('carTitleName', '')).strip().upper()
        
        if raw_v in trims_master:
            return raw_v
            
        combined = f" {raw_v} {raw_vn} {rem} {title} "
        # Separate adjacent abbreviations like FWDW6BSIV -> FWD W6 BSIV
        combined = re.sub(r'(FWD|RWD|AWD)(W\d+)', r'\1 \2', combined)
        combined = re.sub(r'(W\d+)(BS\w+)', r'\1 \2', combined)
        combined = re.sub(r'[\/_]', ' ', combined)
        
        for t in trims_master:
            pattern = r'(?<![A-Z0-9])' + re.escape(t) + r'(?![A-Z0-9])'
            if re.search(pattern, combined):
                if t in ['SX (O)', 'SX(O)', 'SXO']: return 'SX (O)'
                if t in ['ASTA (O)', 'ASTA(O)']: return 'Asta (O)'
                if t in ['SPORTZ+', 'SPORTZ +', 'SPORTZ PLUS']: return 'Sportz+'
                if t in ['ZXI (O)', 'ZXI(O)', 'ZXI O']: return 'ZXI (O)'
                if t in ['VXI (O)', 'VXI(O)', 'VXI O']: return 'VXI (O)'
                if t in ['LXI (O)', 'LXI(O)', 'LXI O']: return 'LXI (O)'
                if t in ['E+', 'E PLUS']: return 'E+'
                return t
                
        if raw_v and raw_v not in ['NAN', 'NONE', 'GENERAL', '']:
            if len(raw_v) <= 12:
                return raw_v
                
        return 'Standard / Base'

    if len(df_m_pass) > 0:
        df_m_pass['clean_variant'] = df_m_pass.apply(extract_variant, axis=1)

        # Transmission
        def clean_trans(row):
            t_str = str(row.get('transmission', '')).strip().upper()
            rem_str = str(row.get('battery_remarks', '')).strip().upper()
            if any(k in t_str for k in ['AUTO', 'AMT', 'CVT', 'DCT', 'AT']): return 'Automatic'
            if any(k in rem_str for k in ['AUTO', 'AMT', 'CVT', 'DCT', ' AT ', 'AT7STR', 'W10AT']): return 'Automatic'
            return 'Manual'
        df_m_pass['clean_transmission'] = df_m_pass.apply(clean_trans, axis=1)

        # Fuel
        def clean_fuel(f):
            s = str(f).strip().upper()
            if 'DIESEL' in s or 'DEISEL' in s: return 'Diesel'
            if 'CNG' in s or 'LPG' in s: return 'CNG/LPG'
            if 'HYBRID' in s or 'EV' in s or 'ELECTRIC' in s: return 'Hybrid/EV'
            return 'Petrol'
        df_m_pass['clean_fuel'] = df_m_pass['fuel_type'].apply(clean_fuel)

        # Ownership
        def clean_owner(o):
            s = str(o).strip()
            try:
                v = float(re.sub(r'[^0-9.]', '', s))
                if v >= 3: return '3+ Owners'
                if v == 2: return '2nd Owner'
                if v == 1: return '1st Owner'
            except Exception:
                pass
            return '1st Owner'
        df_m_pass['clean_owner'] = df_m_pass['vehicle_owner_number'].apply(clean_owner)

        # Displacement CC
        def parse_cc(desc):
            try:
                d = json.loads(str(desc))
                cc = float(d.get('cubicCapacity', 0))
                if 600 <= cc <= 5000: return cc
            except Exception:
                pass
            return np.nan
        df_m_pass['clean_cc'] = df_m_pass['vechileDescription'].apply(parse_cc)

        # Year & Odometer
        df_m_pass['clean_year'] = pd.to_numeric(df_m_pass['year_of_mfg'], errors='coerce')
        df_m_pass.loc[df_m_pass['clean_year'] < 1000, 'clean_year'] = np.nan
        df_m_pass.loc[df_m_pass['clean_year'].isnull(), 'clean_year'] = pd.to_datetime(df_m_pass['rc_regn_date'], errors='coerce').dt.year
        df_m_pass['clean_odo'] = pd.to_numeric(df_m_pass['speedometer'], errors='coerce')

        # Price typo corrections
        def correct_master_prices(row):
            off = float(row.get('offered_price')) if pd.notnull(row.get('offered_price')) else np.nan
            exp = float(row.get('expected_price')) if pd.notnull(row.get('expected_price')) else np.nan
            pay = float(row.get('price')) if pd.notnull(row.get('price')) else np.nan
            
            # Payout typo check
            if pd.notnull(off) and pd.notnull(exp) and pd.notnull(pay):
                if 0.5 <= off / exp <= 2.0 and off / pay > 5.0:
                    pay = off - 10000.0
                    
            # Extra zero check
            if pd.notnull(off):
                ratio_exp = (off / exp) if (pd.notnull(exp) and exp > 0) else 1.0
                ratio_pay = (off / pay) if (pd.notnull(pay) and pay > 0) else 1.0
                if ratio_exp > 5.0 and ratio_pay > 5.0:
                    benchmark = pay if (pd.notnull(pay) and pay > 0) else exp
                    if 0.6 <= (off / 10.0) / benchmark <= 2.5:
                        off = off / 10.0
                if ratio_exp < 0.2 and (exp / 10.0) / off <= 2.0:
                    exp = exp / 10.0

            return pd.Series([off, exp, pay], index=['clean_offered', 'clean_expected', 'clean_payout'])

        mast_prices = df_m_pass.apply(correct_master_prices, axis=1)
        df_m_pass['clean_offered'] = mast_prices['clean_offered']
        df_m_pass['clean_expected'] = mast_prices['clean_expected']
        df_m_pass['clean_payout'] = mast_prices['clean_payout']

        # Sort master records so highest stage priority and latest date comes first
        stage_priority = {'delivered': 4, 'closed': 4, 'priceconfirmation': 3, 'reauction': 2, 'return': 2, 'pending': 1}
        df_m_pass['stage_rank'] = df_m_pass['stage'].map(lambda s: stage_priority.get(str(s).lower().strip(), 0))
        df_m_pass['inward_dt'] = pd.to_datetime(df_m_pass['inward_date'], errors='coerce')
        master_specs_lookup = df_m_pass.sort_values(['clean_reg', 'stage_rank', 'inward_dt'], ascending=[True, False, False]).drop_duplicates(subset=['clean_reg'], keep='first').copy()
        master_specs_lookup.set_index('clean_reg', inplace=True)
        print(f"Built master specs lookup for {len(master_specs_lookup):,} unique TN vehicles.")
    else:
        master_specs_lookup = pd.DataFrame()

    # --- 6. UNIFY & DEDUPLICATE RECORDS ---
    unified_records = []
    seen_regs = set()

    # First: Ingest COMBINED verified conversion sales records
    if len(df_c_tn) > 0:
        for idx, row in df_c_tn.iterrows():
            reg = row['clean_reg']
            seen_regs.add(reg)
            
            make_c = clean_make_name(row.get('MAKE', ''))
            model_c = clean_model_name(row.get('MODEL', ''))
            year_c = float(row.get('YEAR')) if pd.notnull(row.get('YEAR')) and 1995 <= float(row.get('YEAR')) <= reference_year else np.nan
            odo_c = float(row.get('ODO METER')) if pd.notnull(row.get('ODO METER')) and 500 <= float(row.get('ODO METER')) <= 500000 else np.nan
            final_bid = float(row.get('FINAL BID VALUE')) if pd.notnull(row.get('FINAL BID VALUE')) else np.nan
            total_income = float(row.get('TOTAL INCOME')) if pd.notnull(row.get('TOTAL INCOME')) else 0.0
            seller = str(row.get('SELLER NAME', 'TVS Certified TN')).strip()
            buyer = str(row.get('BUYER NAME', 'TVS Auction Buyer')).strip()
            stage = 'converted'
            
            # Enrich from masterdata if available
            if not master_specs_lookup.empty and reg in master_specs_lookup.index:
                m_row = master_specs_lookup.loc[reg]
                variant = m_row.get('clean_variant', 'Standard / Base')
                trans = m_row.get('clean_transmission', 'Manual')
                fuel = m_row.get('clean_fuel', 'Petrol')
                owner = m_row.get('clean_owner', '1st Owner')
                cc = m_row.get('clean_cc', np.nan)
                exp_p = m_row.get('clean_expected', final_bid)
                payout = m_row.get('clean_payout', final_bid - total_income if total_income > 0 else final_bid - 10000.0)
                if pd.isnull(year_c) and pd.notnull(m_row.get('clean_year')):
                    year_c = float(m_row.get('clean_year'))
                if pd.isnull(odo_c) and pd.notnull(m_row.get('clean_odo')):
                    odo_c = float(m_row.get('clean_odo'))
                source = 'COMBINED+Master'
            else:
                variant = 'Standard / Base'
                trans = 'Manual'
                fuel = 'Petrol'
                owner = '1st Owner'
                cc = np.nan
                exp_p = final_bid
                payout = final_bid - total_income if total_income > 0 else final_bid - 10000.0
                source = 'COMBINED_JAN26_AUG26'

            unified_records.append({
                'VEH NO': reg,
                'MAKE': make_c,
                'MODEL': model_c,
                'VARIANT': variant,
                'YEAR': year_c,
                'ODO METER': odo_c,
                'FUEL TYPE': fuel,
                'TRANSMISSION': trans,
                'OWNER': owner,
                'FINAL BID VALUE': final_bid,
                'EXPECTED PRICE': exp_p,
                'PRICE PAYOUT': payout,
                'TOTAL INCOME': max(0.0, final_bid - payout),
                'ENGINE_CC': cc,
                'SELLER NAME': seller,
                'BUYER NAME': buyer,
                'STAGE': stage,
                'SOURCE_FILE': source
            })

    print(f"Ingested {len(unified_records):,} verified records from {combined_path}.")

    # Second: Ingest remaining unique auction vehicles from masterdata_TN
    added_from_master = 0
    if not master_specs_lookup.empty:
        for reg, m_row in master_specs_lookup.iterrows():
            if reg in seen_regs:
                continue
                
            off_price = m_row.get('clean_offered')
            if pd.isnull(off_price) or off_price <= 10000:
                continue
                
            src_file = str(m_row.get('source_file', ''))
            stg = str(m_row.get('stage', '')).lower()
            if src_file not in ['PostBid', 'PreBid'] and stg not in ['delivered', 'closed', 'return', 'priceconfirmation', 'pending']:
                continue

            seen_regs.add(reg)
            make_m = m_row.get('clean_make', '')
            model_m = m_row.get('clean_model', '')
            year_m = float(m_row.get('clean_year')) if pd.notnull(m_row.get('clean_year')) and 1995 <= float(m_row.get('clean_year')) <= reference_year else np.nan
            odo_m = float(m_row.get('clean_odo')) if pd.notnull(m_row.get('clean_odo')) and 500 <= float(m_row.get('clean_odo')) <= 500000 else np.nan
            exp_m = m_row.get('clean_expected', off_price)
            pay_m = m_row.get('clean_payout', off_price - 10000.0)
            
            unified_records.append({
                'VEH NO': reg,
                'MAKE': make_m,
                'MODEL': model_m,
                'VARIANT': m_row.get('clean_variant', 'Standard / Base'),
                'YEAR': year_m,
                'ODO METER': odo_m,
                'FUEL TYPE': m_row.get('clean_fuel', 'Petrol'),
                'TRANSMISSION': m_row.get('clean_transmission', 'Manual'),
                'OWNER': m_row.get('clean_owner', '1st Owner'),
                'FINAL BID VALUE': off_price,
                'EXPECTED PRICE': exp_m,
                'PRICE PAYOUT': pay_m,
                'TOTAL INCOME': max(0.0, off_price - pay_m),
                'ENGINE_CC': m_row.get('clean_cc', np.nan),
                'SELLER NAME': str(m_row.get('customer_name', 'TVS Certified TN')).strip(),
                'BUYER NAME': str(m_row.get('username', 'TVS Auction Buyer')).strip(),
                'STAGE': m_row.get('stage', 'return'),
                'SOURCE_FILE': 'masterdata_TN'
            })
            added_from_master += 1

    print(f"Added {added_from_master:,} unique auction vehicles from {excel_path}.")
    u_df = pd.DataFrame(unified_records)
    u_df = u_df.drop_duplicates(subset=['VEH NO'], keep='first').copy()

    # --- 7. FINAL INTEGRITY & SANITY CLEANING ---
    # RTO Zone Mapping
    chennai_rtos = {'TN01','TN02','TN03','TN04','TN05','TN06','TN07','TN09','TN10','TN11','TN12','TN14','TN18','TN20','TN22','TN85'}
    coimbatore_rtos = {'TN37','TN38','TN39','TN40','TN41','TN42','TN43','TN66','TN99'}
    madurai_rtos = {'TN58','TN59','TN64','TN65','TN67','TN72','TN74','TN76'}

    def get_rto_zone(rto_str):
        code = str(rto_str)[:4].upper()
        if code in chennai_rtos: return 'Chennai Metro'
        if code in coimbatore_rtos: return 'Coimbatore Hub'
        if code in madurai_rtos: return 'Madurai / South TN'
        return 'Rest of TN'

    u_df['RTO_ZONE'] = u_df['VEH NO'].apply(get_rto_zone)
    u_df['BRANCH'] = u_df['RTO_ZONE']
    u_df['STATE'] = 'TN'

    # Filter valid years and calculate age
    u_df = u_df[u_df['YEAR'].notnull() & (u_df['YEAR'] >= 1995) & (u_df['YEAR'] <= reference_year)].copy()
    u_df['YEAR'] = u_df['YEAR'].astype(int)
    u_df['age'] = np.clip(reference_year - u_df['YEAR'], 0.5, 30.0)

    # Odometer imputation
    median_odo_by_make_age = u_df.groupby(['MAKE', 'YEAR'])['ODO METER'].transform('median')
    median_odo_by_age = u_df.groupby('YEAR')['ODO METER'].transform('median')
    u_df['ODO METER'] = u_df['ODO METER'].fillna(median_odo_by_make_age).fillna(median_odo_by_age).fillna(65000.0)
    u_df['log_odo'] = np.log(np.clip(u_df['ODO METER'], 500.0, 500000.0))

    # Price validation & bounds
    u_df = u_df[u_df['FINAL BID VALUE'].notnull() & (u_df['FINAL BID VALUE'] >= 25000) & (u_df['FINAL BID VALUE'] <= 4000000)].copy()
    u_df['log_price'] = np.log(u_df['FINAL BID VALUE'])

    # Make Model Key
    u_df['make_model'] = u_df['MAKE'] + ' | ' + u_df['MODEL']

    # Sort descending by Year and Final Bid Value
    u_df = u_df.sort_values(['YEAR', 'FINAL BID VALUE'], ascending=[False, False]).reset_index(drop=True)
    u_df['SL NO'] = range(1, len(u_df) + 1)

    print(f"\nFinal cleaned & validated Tamil Nadu auction records: {len(u_df):,}")
    print(f"Distinct Makes: {u_df['MAKE'].nunique()}, Distinct Models: {u_df['MODEL'].nunique()}")
    print(f"RTO Zone Breakdown:\n{u_df['RTO_ZONE'].value_counts()}")

    # Export to CSV
    if output_csv:
        u_df.to_csv(output_csv, index=False)
        print(f"Saved {len(u_df):,} verified records to {output_csv}.")

    # Multi-sheet Excel workbook export with audit trail
    if update_excel:
        target_out = "masterdata_TN_cleaned.xlsx"
        try:
            print(f"Saving multi-sheet audited Excel workbook to {target_out}...")
            with pd.ExcelWriter(target_out, engine='xlsxwriter') as writer:
                u_df.to_excel(writer, sheet_name='Cleaned_Training_Data', index=False)
                if len(df_m_tn) > 0:
                    df_m_tn.to_excel(writer, sheet_name='Full_Master_Audited', index=False)
            print(f"Successfully saved audited Excel workbook: {target_out}")
        except Exception as e:
            print(f"Notice saving Excel workbook: {e}")

    return u_df

def clean_vehicle_data(excel_path="cleaned_tn_vehicle_data.csv", prefer_tn=True):
    """
    Default entrypoint to load the active vehicle auction dataset.
    Prioritizes the Tamil Nadu master dataset (cleaned_tn_vehicle_data.csv / masterdata_TN.xlsx).
    """
    if prefer_tn:
        if os.path.exists("cleaned_tn_vehicle_data.csv"):
            try:
                return pd.read_csv("cleaned_tn_vehicle_data.csv")
            except Exception:
                pass
        if os.path.exists("masterdata_TN.xlsx"):
            try:
                return clean_tn_master_data("masterdata_TN.xlsx", "cleaned_tn_vehicle_data.csv")
            except Exception as e:
                print(f"Notice running clean_tn_master_data: {e}")

    if os.path.exists(excel_path) and excel_path.endswith('.csv'):
        try:
            return pd.read_csv(excel_path)
        except Exception:
            pass

    return combine_and_export_master_dataset()

def generate_sample_template_excel():
    """Generates an in-memory sample Excel template conforming to the auction schema."""
    sample_data = {
        'SL NO': [1, 2, 3, 4],
        'FY': ['2025-26', '2025-26', '2025-26', '2025-26'],
        'DATE': ['2026-08-01', '2026-08-05', '2026-08-10', '2026-08-15'],
        'STATE': ['TN', 'KL', 'AP/TS', 'AP/TS'],
        'BRANCH': ['TN', 'KL', 'AP/TS', 'AP/TS'],
        'VEH NO': ['TN09CB1234', 'KL07CD5678', 'TS09EB2697', 'AP39JV0286'],
        'MAKE': ['Maruti', 'Hyundai', 'Honda', 'Tata'],
        'MODEL': ['Swift', 'i20', 'City', 'Nexon'],
        'YEAR': [2018, 2019, 2015, 2021],
        'ODO METER': [65000, 45000, 85000, 38000],
        'FINAL BID VALUE': [425000, 520000, 310000, 680000],
        'SELLER NAME': ['Khivraj Motor', 'Popular Maruti', 'TVS Certified - Hyderabad', 'TVS Certified - Vijayawada'],
        'BUYER NAME': ['Auction Buyer 1', 'Auction Buyer 2', 'Auction Buyer 3', 'Auction Buyer 4'],
        'TOTAL INCOME': [10000, 10000, 12500, 10000]
    }
    df = pd.DataFrame(sample_data)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='AUCTION_DATA')
    return output.getvalue()

def generate_sample_template_csv():
    """Generates an in-memory sample CSV template conforming to the auction schema."""
    sample_data = {
        'SL NO': [1, 2, 3, 4],
        'FY': ['2025-26', '2025-26', '2025-26', '2025-26'],
        'DATE': ['2026-08-01', '2026-08-05', '2026-08-10', '2026-08-15'],
        'STATE': ['TN', 'KL', 'AP/TS', 'AP/TS'],
        'BRANCH': ['TN', 'KL', 'AP/TS', 'AP/TS'],
        'VEH NO': ['TN09CB1234', 'KL07CD5678', 'TS09EB2697', 'AP39JV0286'],
        'MAKE': ['Maruti', 'Hyundai', 'Honda', 'Tata'],
        'MODEL': ['Swift', 'i20', 'City', 'Nexon'],
        'YEAR': [2018, 2019, 2015, 2021],
        'ODO METER': [65000, 45000, 85000, 38000],
        'FINAL BID VALUE': [425000, 520000, 310000, 680000],
        'SELLER NAME': ['Khivraj Motor', 'Popular Maruti', 'TVS Certified - Hyderabad', 'TVS Certified - Vijayawada'],
        'BUYER NAME': ['Auction Buyer 1', 'Auction Buyer 2', 'Auction Buyer 3', 'Auction Buyer 4'],
        'TOTAL INCOME': [10000, 10000, 12500, 10000]
    }
    df = pd.DataFrame(sample_data)
    return df.to_csv(index=False).encode('utf-8')

if __name__ == "__main__":
    print("Executing master multi-branch dataset combination...")
    master = combine_and_export_master_dataset()
    print(f"\nSuccessfully combined master dataset!")
    print(f"Total Transactions: {len(master)}")
    print(f"Unique Makes: {master['MAKE'].nunique()}")
    print(f"Unique Models: {master['MODEL'].nunique()}")
    print("\nState Breakdown:")
    print(master['STATE'].value_counts())
    print("\nBranch Breakdown:")
    print(master['BRANCH'].value_counts())
