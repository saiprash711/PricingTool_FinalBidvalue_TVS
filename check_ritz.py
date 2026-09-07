import pandas as pd

excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
df = pd.concat([pd.read_excel(excel_path, sheet_name=s) for s in sheets], ignore_index=True)

print("--- Ritz rows ---")
print(df[df['MODEL'].str.contains('Ritz', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE', 'ODO METER']])

print("--- Ciaz rows ---")
print(df[df['MODEL'].str.contains('Ciaz', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE', 'ODO METER']])
