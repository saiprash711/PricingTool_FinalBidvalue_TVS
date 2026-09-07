import pandas as pd

excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
df = pd.concat([pd.read_excel(excel_path, sheet_name=s) for s in sheets], ignore_index=True)

print("--- Check 'Riaz' ---")
print(df[df['MODEL'].str.contains('Riaz', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

print("\n--- Check 'Datsun' / 'Datson' ---")
print(df[df['MODEL'].str.contains('Dat', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

print("\n--- Check Brezza variants ---")
print(df[df['MODEL'].str.contains('Brezza', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

print("\n--- Check Swift / Swift Dzire / Dzire ---")
print(df[df['MODEL'].str.contains('Dzire', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']].head(10))

print("\n--- Check i20 / i20 Asta ---")
print(df[df['MODEL'].str.contains('i20', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

print("\n--- Check Celerio / Celerio-X ---")
print(df[df['MODEL'].str.contains('Celerio', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

print("\n--- Check Indigo / Indigo CS ---")
print(df[df['MODEL'].str.contains('Indigo', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

print("\n--- Check Getz / Getz Prime ---")
print(df[df['MODEL'].str.contains('Getz', case=False, na=False)][['VEH NO', 'MAKE', 'MODEL', 'YEAR', 'FINAL BID VALUE']])

