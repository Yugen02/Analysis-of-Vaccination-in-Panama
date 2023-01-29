import matplotlib.pyplot as plt
from XGBPredict_pc import XGBPred
from pathlib import Path
import pandas as pd
import numpy as np
import time
# from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import nltk
from nltk.probability import FreqDist
from prettytable import PrettyTable

weeks = np.arange(1, 54)

x = PrettyTable()

startTime = time.time()

#Running it for 2019
# sentdif19, negrate19, posrate19, notweets19, usatweets19, norttweets19 = [], [], [], [], [], []
# pathlist19 = Path('D:/GITTS/Data/Weekly/2019').glob('*.csv')
# for path in pathlist19:
#     print(str(path))
#     pred19 = XGBPred(path)
#     print('Sentiment difference, negativity rate and positivity rate:', pred19.sentiment_difference, pred19.negativityrate, pred19.positivityrate)
#     sentdif19.append(pred19.sentiment_difference)
#     negrate19.append(pred19.negativityrate)
#     posrate19.append(pred19.positivityrate)
#     notweets19.append(pred19.numberoftweets)
#     norttweets19.append(pred19.numberoftweetssansrt)
#     usatweets19.append(pred19.numberofusable)
#
# print(str(sentdif19))
# print(str(negrate19))
# print(str(posrate19))
# print(str(notweets19))
# print(str(usatweets19))
# print("Number of Tweets from 2019 Retrieved: " + str(sum(notweets19)))
# print("Number of Tweets from 2019 without RTs: " + str(sum(norttweets19)))
# print("Number of Tweets from 2019 Used: " + str(sum(usatweets19)))
#
#
# x.add_column("Week No.", weeks)
# x.add_column("No. of Tweets", notweets19)
# x.add_column("No. of T. W/O RTs", norttweets19)
# x.add_column("No. of Usable", usatweets19)
# print(x)

#Running it for 2020
sentdif20, negrate20, posrate20, notweets20, usatweets20, norttweets20 = [], [], [], [], [], []
pathlist20 = Path(r'csv').glob('*.csv')

for path in pathlist20:
    print(str(path))
    a = pd.read_csv(path)
    if len(a) >= 3:
        pred20 = XGBPred(path)
        print('Sentiment difference, negativity rate and positivity rate:', pred20.sentiment_difference, pred20.negativityrate, pred20.positivityrate)
        sentdif20.append(pred20.sentiment_difference)
        negrate20.append(pred20.negativityrate)
        posrate20.append(pred20.positivityrate)
        notweets20.append(pred20.numberoftweets)
        norttweets20.append(pred20.numberoftweetssansrt)
        usatweets20.append(pred20.numberofusable)
    else:
        sentdif20.append(0)
        negrate20.append(0)
        posrate20.append(0)
        notweets20.append(0)
        norttweets20.append(0)
        usatweets20.append(0)

print(str(sentdif20))
print(str(negrate20))
print(str(posrate20))
print(str(notweets20))
print(str(usatweets20))
print("Number of Tweets from 2020 Retrieved: " + str(sum(notweets20)))
print("Number of Tweets from 2020 without RTs: " + str(sum(norttweets20)))
print("Number of Tweets from 2020 Used: " + str(sum(usatweets20)))

y = PrettyTable()
y.add_column("Week No.", weeks)
y.add_column("No. of Tweets", notweets20)
y.add_column("No. of T. W/O RTs", norttweets20)
y.add_column("No. of Usable", usatweets20)
print(y)

executionTime = (time.time() - startTime)
print('Time elapsed: ', str(executionTime))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

plt.figure(1)
plt.plot(weeks, negrate20, label="Negativo", marker='*')
plt.plot(weeks, sentdif20, label="Index", marker='*')
plt.plot(weeks, posrate20, label="Positivo", marker='*')
plt.xlabel("Weeks")
plt.ylabel("Polarity Difference")
plt.title('Polaridad Pfizer 2021 CON RT')
plt.legend()
plt.minorticks_on()
plt.xticks(np.arange(1,53))
plt.grid(which='minor', axis='x', color='0.8')
plt.show()

# plt.figure(2)
# # plt.plot(weeks, negrate19, label="2019", marker='*')
# plt.plot(weeks, negrate20, label="2020", marker='*')
# plt.xlabel("Weeks")
# plt.ylabel("Negativity Rate")
# plt.title('Weekly Negativity Rate of Tweets Produced')
# plt.legend()
# plt.minorticks_on()
# plt.xticks(np.arange(1,53))
# plt.grid(which='minor', axis='x', color='0.8')
# plt.show()

# plt.figure(3)
# plt.plot(weeks, posrate19, label="2019", marker='*')
# plt.plot(weeks, posrate20, label="2020", marker='*')
# plt.xlabel("Weeks")
# plt.ylabel("Positivity Rate")
# plt.title('Weekly Positivity Rate of Tweets Produced')
# plt.legend()
# plt.minorticks_on()
# plt.xticks(np.arange(1,53))
# plt.grid(which='minor', axis='x', color='0.8')
# plt.show()