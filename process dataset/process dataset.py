import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('splitted_covid_dump.csv', on_bad_lines='skip')  # Reemplaza con la ruta de tu archivo CSV

# Guardar como archivo Excel
df.to_excel('splitted_covid_dump.xlsx', index=False, sheet_name='NombreHoja')  # Reemplaza con la ruta donde deseas guardar el nuevo archivo
