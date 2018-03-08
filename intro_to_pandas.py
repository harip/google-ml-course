"""Google ML crash course learning exercises"""

import pandas as pd
import matplotlib.pyplot as plt

URL = "https://storage.googleapis.com/mledu-datasets/california_housing_train.csv"
CALIFORNIA_HOUSING_DATAFRAME = pd.read_csv(URL, sep=",")

CALIFORNIA_HOUSING_DATAFRAME.hist('housing_median_age')
plt.show()
