"""Google ML crash course learning exercises"""

import pandas as pd

CITY_NAMES = pd.Series(['San Francisco', 'San Jose', 'Sacramento'])
POPULATION = pd.Series([852469, 1015785, 485199])
CITIES = pd.DataFrame({'City Name':CITY_NAMES, 'Population':POPULATION})
CITIES['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
CITIES['Population density'] = CITIES['Population'] / CITIES['Area square miles']

CITIES['Large Saint City'] = CITIES['City Name'].apply(lambda name: name.startswith('San')) & (CITIES['Area square miles']>50)

print(CITIES)
