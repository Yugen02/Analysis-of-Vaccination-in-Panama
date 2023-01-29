import pandas as pd
import numpy as np
import csv
import datetime
import re

df1 = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Parced\P_Astrazeneca\Datos_Final\1ra_Mitad_final.csv', parse_dates=["time"])
df2 = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Parced\P_Astrazeneca\Datos_Final\2da_Mitad_final.csv', parse_dates=["time"])
df3 = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Parced\P_Astrazeneca\Datos_Final\3ra_Mitad_final.csv', parse_dates=["time"])
df4 = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Parced\P_Astrazeneca\Datos_Final\4ta_Mitad_final.csv', parse_dates=["time"])

df = pd.concat([df1,df2,df3,df4])
df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Parced\P_Vacunas\Prueba_RE_2021.csv', parse_dates=["time"])
df = df.drop_duplicates(subset="text")
df['Date'] = pd.to_datetime(df['time'], errors='coerce')


# print(df)
df['dayofyear'] = df['Date'].dt.dayofyear
df['dayofweek'] = df['Date'].dt.week
print(df)

#~~~~~~~~~~~~~~ DATOS TOTALES CONCATENADOS~~~~~~~~~~~~~~~~~~~
# df.to_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Parced\P_Astrazeneca\Datos_Final_AZ',columns=['text', 'words','time','dayofyear','dayofweek'])
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
weeks0 = np.arange(1,54)

for i in weeks0:
    a = df.loc[df['dayofweek'] == i]
    if i < 10:  
        a.to_csv(f'csv',columns=['text','time','dayofyear','dayofweek'])
    else:           
        a.to_csv(f'csv',columns=['text','time','dayofyear','dayofweek'])
# df.to_csv('Pfizer_dayofyear.xlsx', columns=['text', 'words','time','dayofyear','dayofweek'])
print('GUARDADO!')

# size = 100000

# df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Total_rumbo_a_parcear.csv', chunksize=size)
# # Definimos el tamaño máximo de filas de cada partición (1MM)
# # Creamos un objeto iterador 'df_chunk' que contendrá las particiones con 1MM de filas en cada iteración
# # Creamos una variable booleana verdadera 'header' para exportar las cabeceras una única vez
# header = True
# # Ahora vamos a recorrer cada partición, realizar el filtro solicitado, y exportar el resultado a un nuevo CSV
# # El atributo mode='a' (APPEND) sirve para no sobreescribir el archivo Resultado.csv en cada iteración, sino para siempre adjuntar los nuevos resultados
# # Luego de la 1ra iteración 'header' vuelve a ser falsa para no colocar nuevamente las cabeceras en el CSV final
# for chunk in df['words']:
#     chunk_filter = chunk[chunk['words'] == 1]
#     chunk_filter.to_csv('Resultado.csv', header=header, mode='a')
#     header = False

# df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\AstraZeneca_parced_prueba.csv')

# tweetno = 1
# tweetlist, lengthvec, depwordcount = [], [], []
# words = ["AstraZeneca,","astrazeneca,","Astra Zeneca,","Astra Zeneca",",AstraZeneca",",astrazeneca",'Astra-Zeneca','Astra-Zeneca,',',Astra-Zeneca','astra-zeneca',',astra-zeneca','astra-zeneca,','astra-seneca',',astra-seneca','astra-seneca,','Astra-Seneca,','Astra-Seneca',',Astra-Seneca','AstraSeneca','AstraSeneca,',',AstraSeneca','AztraSeneca,','AztraSeneca',',AztraSeneca,','aztraseneca',',aztraseneca','aztraseneca,','AztraZeneca','AztraZeneca,',',AztraZeneca','aztrazeneca','aztrazeneca,',',aztrazeneca']
# formatted = []
# weekno = 0
# deprwo = set(words)

# for row in df['text']:
#     tweetno += 1
#     test = row
#     test = re.sub("@[A-Za-z0-9_]+", "", test)  # Remove @ sign
#     test = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", test)  # Remove http links
#     # test = re.sub("#[A-Za-z0-9_]+", "", test) # Remove hashtags
#     tweetlist.append(test)
#     lengthvec.append(len(test))
#     listtest = test.split()



#     if set(listtest).intersection(words):
#         wordcount = len(set(listtest).intersection(words))
#         depwordcount.append(wordcount)
#     else:
#         depwordcount.append(0)


# df.drop('text', axis = 1, inplace = True)
# df['text'] = tweetlist
# # print(df["text"])
# # df['time'] = formatted
# # print(df['time'])
# df['length'] = lengthvec
# df['words'] = depwordcount
# # df.to_csv('Total_rumbo_a_parcear.csv',)

# # df['dayofyear'] = df['time'].dt.dayofyear

# a = df.loc[df['words'] >= 1]

# print(a)

# a.to_csv('AstraZeneca_parced_prueba_1.csv',columns=['text', 'words','time'])

