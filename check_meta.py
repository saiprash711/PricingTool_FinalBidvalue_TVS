import pandas as pd

excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
df = pd.concat([pd.read_excel(excel_path, sheet_name=s) for s in sheets], ignore_index=True)

print("STATE values:", df['STATE'].value_counts())
print("FY values:", df['FY'].value_counts())
print("Date min/max:", df['DATE'].min(), df['DATE'].max())
print("Total income stats:\n", df['TOTAL INCOME'].describe())
