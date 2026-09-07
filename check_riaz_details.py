import sys, io, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
df = pd.concat([pd.read_excel(excel_path, sheet_name=s) for s in sheets], ignore_index=True)

print(df[df['VEH NO'] == 'AP16CG0018'].to_dict(orient='records'))
