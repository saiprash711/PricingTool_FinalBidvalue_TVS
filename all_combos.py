import sys
import io
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

excel_path = "JAN26_AUG26_REPORT.xlsx"
sheets = ['JAN-26', 'FEB-26', 'MAR-26', 'APRIL-26', 'MAY-26', 'JUNE-26', 'JULY-26', 'AUG-26']
df = pd.concat([pd.read_excel(excel_path, sheet_name=s) for s in sheets], ignore_index=True)

mm = df.groupby(['MAKE', 'MODEL']).agg(
    count=('FINAL BID VALUE', 'count'),
    mean_price=('FINAL BID VALUE', 'mean'),
    min_year=('YEAR', 'min'),
    max_year=('YEAR', 'max')
).reset_index().sort_values(by=['MAKE', 'MODEL'])

for idx, row in mm.iterrows():
    print(f"{row['MAKE']:<12} | {row['MODEL']:<16} | count: {row['count']:<3} | years: {int(row['min_year'])}-{int(row['max_year'])} | avg_price: Rs. {row['mean_price']:,.0f}")
