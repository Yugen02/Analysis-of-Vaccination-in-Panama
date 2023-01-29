import pandas as pd
import numpy as np
import csv
import datetime
import re
# from WordsSuggestingDepr import words

words1 = ["Pfizer,","pfizer,","Pfizer","pfizer",",Pfizer",",pfizer",'Pizer','Faizer','Piser','Faiser',',Pizer',',Faizer',',Piser',',Faiser','Pizer,','Faizer,','Piser,','Faiser,','pizer','faizer','piser','faiser',',pizer',',faizer',',piser',',faiser','pizer,','faizer,','piser,','faiser,']
words2 = ["AstraZeneca,","astrazeneca,","Astra Zeneca,","Astra Zeneca",",AstraZeneca",",astrazeneca",'Astra-Zeneca','Astra-Zeneca,',',Astra-Zeneca','astra-zeneca',',astra-zeneca','astra-zeneca,','astra-seneca',',astra-seneca','astra-seneca,','Astra-Seneca,','Astra-Seneca',',Astra-Seneca','AstraSeneca','AstraSeneca,',',AstraSeneca','AztraSeneca,','AztraSeneca',',AztraSeneca,','aztraseneca',',aztraseneca','aztraseneca,','AztraZeneca','AztraZeneca,',',AztraZeneca','aztrazeneca','aztrazeneca,',',aztrazeneca']
words = ['Vacunas','Vacunas,',',Vacunas','vacunas','vacunas,',',vacunas','Vacunas.','.Vacunas','vacunas.','.vacunas','Vacunas','Vacuna,',',Vacuna','vacuna','vacuna,',',vacuna','Vacuna.','.Vacuna','vacuna.','.vacuna','Vacunas','Vacunas,',',Vacunados','vacunados','vacunados,',',vacunados','Vacunados.','.Vacunados','vacunados.','.vacunados','Vacunados','Vacuna,',',Vacuna','vacuna','vacuna,',',vacuna','Vacuna.','.Vacuna','vacuna.','.vacuna']

# words = 0

weekno = 0
deprwo = set(words)

# df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\All2021_PrimeraMitad.csv', names=['id', 'text', 'hashtags',
#                     'created_at', 'geo', 'like_count', 'quote_count', 'reply_count', 'retweet_count'], header=None)
# df = df[:1054]

# df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\All2021_PrimeraMitad2.csv', names=['id', 'text', 'hashtags',
#                     'created_at', 'geo', 'like_count', 'quote_count', 'reply_count', 'retweet_count'], header=None)

# df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\All2021_OctNovDec20.csv', names=['id', 'text', 'hashtags',
#                     'created_at', 'geo', 'like_count', 'quote_count', 'reply_count', 'retweet_count'], header=None)

# df1 = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Datos_2020\All2020_Cont.csv', names=['id', 'text', 'hashtags',
#                     'created_at', 'geo', 'like_count', 'quote_count', 'reply_count', 'retweet_count'], header=None)
df = pd.read_csv(r'csv', names=['id', 'text', 'hashtags',
                    'created_at', 'geo', 'like_count', 'quote_count', 'reply_count', 'retweet_count'], header=None)
# df = df[:1054]
# df = pd.concat([df1,df2])



print(len(df))
df = pd.DataFrame(df)
df = df.replace("\n"," ", regex = True)
print(len(df))
df.dropna(subset = ['text'], inplace=True)

# df.to_csv('una_decima_parte.csv',)
print("SE A GUARDADO EN LA BASE DE DATOS")
print('Data imported correctly, shape:', np.shape(df))

created = df['created_at']
formatted = []

# n = 0
# conteo_pal = []
# for i in df['text']:
#     if "AstraZeneca" in i:
#         n += 1
#         conteo_pal.append(n)
#     else:
#         conteo_pal.append(n)
#     n = 0

# print("CONTEO DE PALABRAS: " + str(len(conteo_pal)))
# df['words'] = conteo_pal
# print(df['words'])





# words_list = {'tried', 'mobile', 'abc'}

# df = pd.DataFrame({'col': ['to make it possible or easier for someone to do',
#                            'unable to acquire a buffer item very likely',
#                            'The organization has tried to make',
#                            'Broadway tried a variety of mobile Phone for the']})

# df = df[df['text'].str.contains('AstraZeneca')]
# print(df)

# df['matches'] = df['text'].str.split().apply(lambda x: set(x) & words)
# df = df.replace('{}', np.nan)
# print(df)







for times in created:
    # print(times)
    try:
        date_time_obj = datetime.datetime.strptime(str(times), '%Y-%m-%dT%H:%M:%S.%fZ')
        # print(times)
        hours = date_time_obj.strftime('%Y/%m/%d-%H')
        formatted.append(hours)
    except:
        print('Se Cago')
        hours = "0/0/0-0"
        formatted.append(hours)
        pass

print(np.shape(formatted))

tweetno = 1
tweetlist, lengthvec, depwordcount = [], [], []

for row in df['text']:
    tweetno += 1
    test = row
    test = re.sub("@[A-Za-z0-9_]+", "", test)  # Remove @ sign
    test = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", test)  # Remove http links
    test = re.sub("#[A-Za-z0-9_]+", "", test) # Remove hashtags
    tweetlist.append(test)
    lengthvec.append(len(test))
    listtest = test.split()

    if set(listtest).intersection(words):
        wordcount = len(set(listtest).intersection(words))
        depwordcount.append(wordcount)
    else:
        depwordcount.append(0)

print("tamaño de la lista de tweets: " + str(len(tweetlist)))

df.drop('text', axis = 1, inplace = True)
df['text'] = tweetlist
# print(df["text"])
df['time'] = formatted
print(df['time'])
df['length'] = lengthvec
df['words'] = depwordcount
df.to_csv(r'csv',columns=['text', 'words','time'])

# df['dayofyear'] = df['time'].dt.dayofyear

a = df.loc[df['words'] >= 1]

print(a)

# a.to_csv('AstraZeneca_2020.csv',columns=['text', 'words','time','matches'])

#print(df)
# df.to_csv('Pfizer.xlsx', columns=['text', 'words','time'])

print('List exported successfully')

###########################################################
# THIS PORTION OF THE CODE SEPARATES THE FILES INTO WEEKS #
###########################################################

# df = pd.read_csv(r'C:\Users\Efrain\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Pfizer_parced.csv', names=['text','words','time'], header=None)

df['isodate'] = pd.to_datetime(a['time'], errors='coerce')
print('Dates interpreted correctly!', np.shape(df))

df.dropna(subset=['isodate'], inplace=True)

df['weekdate'] = df['isodate'].dt.strftime('%Y-%W')
print('Dates converted into week format!, shape:', np.shape(df))

weeklytotals = []

while weekno<54 :
    if weekno < 10:
        print(weekno)
        weeklist = df['weekdate'].str.contains(f'2020-0{weekno}')
        weekdf = df[weeklist]
        weekdf.to_csv(r'csv')
        # weekdf.to_csv(f"D:/GITTS/Data/Costa Rica/Weekly/2019/0{weekno}-2019.csv", mode='a', header=True) #if it's a second run
        print(np.shape(weekdf), weekdf.head())
        weekno += 1
        weeklist = []
        weeklytotals.append(len(weekdf))

    else:
        print(weekno)
        weeklist = df['weekdate'].str.contains(f'2020-{weekno}')
        weekdf = df[weeklist]
        weekdf.to_csv(r'csv')
        # weekdf.to_csv(f"D:/GITTS/Data/Costa Rica/Weekly/2019/{weekno}-2019.csv", mode='a', header=False) #if it's a second run
        print(np.shape(weekdf), weekdf.head())
        weekno += 1
        weeklist = []
        weeklytotals.append(len(weekdf))

print("Done!")

print(sum(weeklytotals), len(df))
print(weeklytotals)

############################################################
# THIS PORTION OF THE CODE SEPARATES THE FILES INTO MONTHS #
############################################################

# jan = np.column_stack([df['created_at'].str.contains("2021-01", na=False, regex=False)])
# feb = np.column_stack([df['created_at'].str.contains("2021-02", na=False, regex=False)])
# mar = np.column_stack([df['created_at'].str.contains("2021-03", na=False, regex=False)])
# apr = np.column_stack([df['created_at'].str.contains("2021-04", na=False, regex=False)])
# may = np.column_stack([df['created_at'].str.contains("2021-05", na=False, regex=False)])
# jun = np.column_stack([df['created_at'].str.contains("2021-06", na=False, regex=False)])
# jul = np.column_stack([df['created_at'].str.contains("2021-07", na=False, regex=False)])
# aug = np.column_stack([df['created_at'].str.contains("2021-08", na=False, regex=False)])
# sep = np.column_stack([df['created_at'].str.contains("2021-09", na=False, regex=False)])
# octo = np.column_stack([df['created_at'].str.contains("2021-10", na=False, regex=False)])
# nov = np.column_stack([df['created_at'].str.contains("2021-11", na=False, regex=False)])
# dec = np.column_stack([df['created_at'].str.contains("2021-12", na=False, regex=False)])

# print("Dates extracted successfully! The total ammount of tweets per month is as follows: (jan, feb, mar, ...)")
# print(sum(jan),sum(feb),sum(mar),sum(apr),sum(may),sum(jun),sum(jul),sum(aug),sum(sep))

# df_jan = df[jan]
# df_feb = df[feb]
# df_mar = df[mar]
# df_apr = df[apr]
# df_may = df[may]
# df_jun = df[jun]
# df_jul = df[jul]
# df_aug = df[aug]
# df_sep = df[sep]
#df_oct = df[octo]
#df_nov = df[nov]
#df_dec = df[dec]
# print("Data saved into monthly dataframes successfully!")

#df_jan.to_csv("D:/GITTS/Data/Monthly/2019/jan2019.csv")
#df_feb.to_csv("D:/GITTS/Data/Monthly/2019/feb2019.csv")
#df_mar.to_csv("D:/GITTS/Data/Monthly/2019/mar2019.csv")
#df_apr.to_csv("D:/GITTS/Data/Monthly/2019/apr2019.csv")
#df_may.to_csv("D:/GITTS/Data/Monthly/2019/may2019.csv")
#df_jun.to_csv("D:/GITTS/Data/Monthly/2019/jun2019.csv")
#df_jul.to_csv("D:/GITTS/Data/Monthly/2019/jul2019.csv")
#df_aug.to_csv("D:/GITTS/Data/Monthly/2019/aug2019.csv")
#df_sep.to_csv("D:/GITTS/Data/Monthly/2019/sep2019.csv")
#df_oct.to_csv("D:/GITTS/Data/Monthly/2019/oct2019.csv")
#df_nov.to_csv("D:/GITTS/Data/Monthly/2019/nov2019.csv")
#df_dec.to_csv("D:/GITTS/Data/Monthly/2019/dec2019.csv")

# df_jan.to_csv("D:/GITTS/Data/Monthly/2021/jan2021.csv", mode='a', header=False)
# df_feb.to_csv("D:/GITTS/Data/Monthly/2021/feb2021.csv", mode='a', header=False)
# df_mar.to_csv("D:/GITTS/Data/Monthly/2021/mar2021.csv", mode='a', header=False)
# df_apr.to_csv("D:/GITTS/Data/Monthly/2021/apr2021.csv", mode='a', header=False)
# df_may.to_csv("D:/GITTS/Data/Monthly/2021/may2021.csv", mode='a', header=False)
# df_jun.to_csv("D:/GITTS/Data/Monthly/2021/jun2021.csv", mode='a', header=False)
# df_jul.to_csv("D:/GITTS/Data/Monthly/2021/jul2021.csv", mode='a', header=False)
# df_aug.to_csv("D:/GITTS/Data/Monthly/2021/aug2021.csv", mode='a', header=False)
# df_sep.to_csv("D:/GITTS/Data/Monthly/2021/sep2021.csv", mode='a', header=False)
#df_oct.to_csv("D:/GITTS/Data/Monthly/2021/oct2021.csv", mode='a', header=False)
#df_nov.to_csv("D:/GITTS/Data/Monthly/2021/nov2021.csv", mode='a', header=False)
#df_dec.to_csv("D:/GITTS/Data/Monthly/2021/dec2021.csv", mode='a', header=False)

# print("Data seggregated into monthly files successfully!")