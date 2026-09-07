import pandas as pd
import numpy as np

excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']

dfs = [pd.read_excel(excel_path, sheet_name=s) for s in sheets]
df = pd.concat(dfs, ignore_index=True)

print("--- NUMERICAL SUMMARY ---")
print(df[['YEAR', 'ODO METER', 'FINAL BID VALUE', 'TOTAL INCOME']].describe())

print("\n--- ZERO OR NEGATIVE CHECKS ---")
print("Year <= 1990:", (df['YEAR'] <= 1990).sum())
print("Odo <= 0:", (df['ODO METER'] <= 0).sum(), df[df['ODO METER'] <= 0][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'ODO METER', 'FINAL BID VALUE']])
print("Bid <= 0:", (df['FINAL BID VALUE'] <= 0).sum())

print("\n--- UNIQUE MAKES ---")
print(sorted(df['MAKE'].dropna().astype(str).unique()))

print("\n--- MAKE-MODEL COMBINATIONS ---")
make_models = df.groupby(['MAKE', 'MODEL']).size().reset_index(name='count')
print(f"Total Make-Model combinations: {len(make_models)}")

print("\n--- CHECK FOR SAME MODEL UNDER MULTIPLE MAKES ---")
model_to_makes = df.groupby('MODEL')['MAKE'].unique()
for model, makes in model_to_makes.items():
    if len(makes) > 1:
        print(f"Model '{model}' appears under multiple makes: {makes}")
        print(df[df['MODEL'] == model][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

print("\n--- ALL UNIQUE MODELS PER MAKE ---")
for make, group in df.groupby('MAKE'):
    print(f"\nMake: {make} (Total {len(group)} rows)")
    print(sorted(group['MODEL'].dropna().unique()))
