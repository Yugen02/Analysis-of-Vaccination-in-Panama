import pandas as pd
import numpy as np
import csv
import datetime
import re

df = pd.read_csv(r'csv')

df = df.loc[df['pred'] == 0]
print(df)

df.to_csv(r'csv')

