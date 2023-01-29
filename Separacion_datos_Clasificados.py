import pandas as pd
import numpy as np
import csv
import datetime
import re

df = pd.read_csv(r'C:\Users\efrai\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Sentimiento CLasificado\Vacunas General\Prediccion_VG_2021.csv')

df = df.loc[df['pred'] == 0]
print(df)

df.to_csv(r'C:\Users\efrai\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Sentimiento CLasificado\Vacunas General\Negativo\VG_PRE_POS.csv')

