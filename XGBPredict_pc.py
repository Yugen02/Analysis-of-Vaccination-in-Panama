from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame, ExcelWriter

from sklearn.feature_extraction.text import TfidfVectorizer

import xgboost as xgb

import nltk
from langid.langid import LanguageIdentifier, model
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import numpy as np
import pandas as pd
import csv
import re

from tokenizers import Tokenizer
from transformers import AutoTokenizer

identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
identifier.set_languages(['es','en'])


#Ground Truth

excel = pd.ExcelFile(r'C:\Users\efrai\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Verano\PC\XGBoost\CompleteManualReview_1.xlsx')

neg = pd.read_excel(excel, 'Negatives')
pos = pd.read_excel(excel, 'Positives')

negatags = [1]*len(neg)
positags = [0]*len(pos)

neg['Tags'] = negatags
pos['Tags'] = positags

tags = np.append(negatags, positags)

# Function Definitions
stop_words = set(stopwords.words("spanish"))
snowball = nltk.SnowballStemmer(language='spanish')

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

vectorizer = CountVectorizer()                 #CAMBIADO
# print(vectorizer)

def stemming(data):
   text = snowball.stem(data)
   return text

# Importing Data
# Lowercasing
neg["Tweets"] = neg["Tweets"].str.lower()
pos["Tweets"] = pos["Tweets"].str.lower()

# Pre-Processing
neg['Stopword Filtered'] = neg['Tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
pos['Stopword Filtered'] = pos['Tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

neg['Stemmed'] = neg['Stopword Filtered'].apply(lambda x: stemming(x))
pos['Stemmed'] = pos['Stopword Filtered'].apply(lambda x: stemming(x))

# Adding the two lists
numpynegs = neg['Stemmed'].to_numpy()
numpypos = pos['Stemmed'].to_numpy()

numpymerged = np.append(numpynegs, numpypos)

# Vectorizing ground truth
# vectorized = vectorizer.fit_transform(numpymerged)
vectorized = vectorizer.fit_transform(numpymerged)


class XGBPred:
   def __init__(self, path):
       longtweets, languages, percentages, estweets = [], [], [], []
       # Calling data to be classified
       self.path = path
       self.df = pd.read_csv(path)
       self.numberoftweets = len(self.df)
       self.df = self.df[~self.df.text.str.contains("RT ", regex=False, na=False)]
       self.numberoftweetssansrt = len(self.df)
       #### FILTRADO POR KEYWORD !!!!!!!
       # self.bono = self.df[self.df.text.str.contains("bono|bono solidario|vale solidario|vale panama|vale digital", regex=True, na=False)]
       # print("Tweets que hablan del bono: ", np.shape(self.bono), self.bono)

       # Pre-Processing
       self.df["text"] = self.df["text"].str.lower()

       ###### ELIMINACION DE CESGO VENEZOLANO
       # print('Cantidad de tweets antes de remocion de cesgo venezolano: ' + str(len(self.df)))
       # self.df = self.df[~self.df.text.str.contains("venezuela|maduro|marico|chamo|venezolano|ayuda humanitaria|vzla|guaido|guaidó",
       #                               regex=True, na=False)]
       # print('Cantidad de tweets despues de remocion de cesgo venezolano: ' + str(len(self.df)))

       tweetlist = []
       for test in self.df['text']:
           try:
               test = re.sub("@[A-Za-z0-9_]+", "", test)
               test = re.sub("#[A-Za-z0-9_]+", "", test)
               test = re.sub("\n", "", test)
               test = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", test)
               # test = re.sub('"[A-Za-z0-9]"',"", test) # removes quoted content
               tweetlist.append(test)
           except:
               print('PASSED')
               pass
       print('TWEETSLIST: ' + str(tweetlist))
       self.tweets = pd.DataFrame(tweetlist, columns=['Filtered'])
       self.tweets['Stopword Filtered'] = self.tweets['Filtered'].apply(
           lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
       
       # ~~~~~~~~~~~~~~~~~~~~~STEMER NO FUNCIONAL~~~~~~~~~~~~~~~~~~~~~~~~~
    #    self.tweets['Stemmed'] = self.tweets['Stopword Filtered'].apply(lambda x: stemming(x))
       
       # ~~~~~~~~~~~~~~~~~~~~~~STEMMER FUNCIONAL ~~~~~~~~~~~~~~~~~~~~~~~~~
       df_1 = pd.DataFrame(self.tweets['Stopword Filtered'])
       df_1['Stopword Filtered'] = df_1['Stopword Filtered'].str.split()
      #  print(df_1)
       pd.set_option('display.max_colwidth', -1)

       df_1['Stemmed'] = df_1['Stopword Filtered'].apply(lambda x: [snowball.stem(y) for y in x])
      #  print(df_1)
       count = 0

       for i in df_1['Stemmed']:
            a = ' '.join(i)
            df_1.loc[count,'Stemmed'] = a
            count = count + 1
       self.tweets['Stemmed'] = df_1['Stemmed']
       # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

       print(np.shape(self.tweets), self.tweets.head())
    #    pd.DataFrame(self.tweets).to_csv("C:/Users/efrai/OneDrive - Universidad Tecnológica de Panamá/Universidad/Proyecto de la GITTS/Verano/Graficas/prueba_F.csv")
    #    np.savetxt("C:/Users/efrai/OneDrive - Universidad Tecnológica de Panamá/Universidad/Proyecto de la GITTS/Verano/Graficas/prueba.txt", a, delimiter=",")

       for tweet in self.tweets['Stemmed']:
           lang, percent = identifier.classify(tweet)
           languages.append(lang)
           percentages.append(percent)

       print("Languages detected!, Tweets classified: " + str(np.shape(languages)))
       self.tweets['Languages'] = languages
       self.tweets['Probability'] = percentages
      
       # NO OLVIDAR DE QUITAR LOS COMENTARIOS
       self.estweets = self.tweets[self.tweets['Languages'].str.contains('es')]
       print("Spanish tweets extracted. Tweets extracted: " + str(np.shape(self.estweets)))
       print(self.estweets.head(10))
      # HASTA AQUIIIIIIIII
    #    pd.DataFrame(self.estweets).to_csv("C:/Users/50764/OneDrive/Escritorio/XGBoost/prueba_F.csv")
       
       # Vectorizing predictable data
       self.vectorized_X = vectorizer.transform(self.estweets['Stemmed'])

    #    self.vectorized_X = tokenizer.encode(self.estweets(['Stemmed']))
       print("Tweets vectorized. Number of tweets: " + str(np.shape(self.vectorized_X)))

       ##### XGBOOST CLASSIFIER MODEL V1.0

       model = xgb.XGBRegressor(tree_method="gpu_hist",gpu_id=0,colsample_bytree=0.7, learning_rate=0.03, n_estimators=500, max_depth=200,
                                subsample=0.9, objective='binary:logistic', eval_metric=['auc', 'rmse', 'map'])
       model.load_model(r"C:\Users\efrai\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Verano\PC\XGBoost\XGB_OMICRON_V2.model")
       
       self.predictions = model.predict(self.vectorized_X)
       print('Predictions: ' + str(self.predictions))

       self.rounded_negs = np.where(self.predictions > 0.75, 1, 0)
       print('Redondeo Nega: ' + str(self.rounded_negs))

       self.rounded_pos = np.where(self.predictions < 0.25, 1, 0)
       print('Redondeo Posi: ' + str(self.rounded_pos))

       self.pred_negs = sum(self.rounded_negs)
       print('SUMA Nega: ' + str(self.pred_negs))

       self.pred_pos = sum(self.rounded_pos)
       print('SUMA POSI: ' + str(self.pred_pos))

       self.negativityrate = self.pred_negs / len(self.predictions)
       print('Negative Rate: ' + str(self.negativityrate))

       self.positivityrate = self.pred_pos / len(self.predictions)
       print('Positivy Rate: ' + str(self.positivityrate))

       self.sentiment_difference = self.positivityrate - self.negativityrate
       print('DIFERENCIA: ' + str(self.sentiment_difference))

       self.numberofusable = len(self.predictions)
       print('FUSABLE: ' + str(self.numberofusable))

      # TEXTO Y VALOR DE LA PREDICCION
      #  self.estweets['pred'] = self.predictions
      #  print(self.estweets.head(10))
      #  self.estweets.to_csv(f'C:/Users/efrai/OneDrive - Universidad Tecnológica de Panamá/Universidad/Proyecto de la GITTS/Vacunas/Data/Parced/Predicciones_vacunas.csv',columns=['Filtered', 'words'])


pred20 = XGBPred(r'C:\Users\efrai\OneDrive - Universidad Tecnológica de Panamá\Universidad\Proyecto de la GITTS\Vacunas\Data\Parced\P_Pfizer\Datos_Final\1ra_Mitad_final.csv')


# print('Sentiment difference, negativity rate and positivity rate:', pred20.sentiment_difference, pred20.negativityrate, pred20.positivityrate)
