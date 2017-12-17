import pandas as pd
from sklearn.externals import joblib
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler


def addTechnicalFeatures(df):
    df['Data'] = pd.to_datetime(df['Data'])
    df.columns = ['Data', 'Open','High', 'Low', 'Close', 'Wol']
    #Wskazniki analizy technicznej
    k, dfast = talib.STOCH(np.array(df['High']),np.array(df['Low']),
                           np.array(df['Close'])) # uses high, low, close by default
    df['k'] = k
    df['dfast'] = dfast
    df['dslow'] = talib.SMA(dfast, timeperiod=5)
    df['momentum'] = talib.MOM(np.array(df['Close']), timeperiod=4)
    df['roc'] = talib.ROC(np.array(df['Close']), timeperiod=5)
    df['willR'] = talib.WILLR(np.array(df['High']), np.array(df['Low']),
                        np.array(df['Close']), timeperiod = 5)
    #ad = talib.ADOSC(np.array(df['High']), np.array(df['Low']),
    #                          np.array(df['Close']), np.array(df['Wol']))
    df['disp5'] = df['Close'] / talib.SMA(np.array(df['Close']), 5) * 100
    df['disp10'] = df['Close'] / talib.SMA(np.array(df['Close']), 10) * 100
    df['oscp'] = ((talib.SMA(np.array(df['Close']), 5) - talib.SMA(np.array(df['Close']), 10)) / 
                                                        talib.SMA(np.array(df['Close']), 5)) 
    df['rsi'] = talib.RSI(np.array(df['Close']))
    df['CCI'] = talib.CCI(np.array(df['High']),np.array(df['Low']), np.array(df['Close']))
    #Tworzenie zmiennej celu
    df['target1'] = df['Close'].shift(-1) -df['Open']
    df['target5'] = df['Close'].shift(-5) -df['Open']
    df['target3'] = df['Close'].shift(-3) -df['Open']
    #zostawienie tylko zmiennej celu i wskaznikow technicznych
    df.drop(['Data','Open', 'High', 'Low', 'Close', 'Wol'],axis=1, inplace=True)
    return df

#Wczytanie modelu do przewidywania SPY 
rfSPY = joblib.load("/home/olaf/Documents/Modelowanie-AT/modele/RandomForestSPY.pkl")
#Wczytanie bazy danych do przewidywania ostatniego dnia
df = pd.read_csv("https://stooq.pl/q/d/l/?s=es.f&i=d")
#zapisanie daty do zmiennej by wyslac ja w mailu i powtierdzic aktualnosc danych
dataSPY = df.iloc[-1,0]

df.drop("LOP", axis=1, inplace=True)
df = addTechnicalFeatures(df)

#dropujemy zmienne celu  i NA
X = df.drop(['target1','target3','target5'], axis=1)
X.dropna(inplace=True, axis=0, how="any")
#przeskalowanie danych do modelu(bo na takich byl budowany)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#Przewidaywania nastepnego dnia. Gorne 20 percentyli zaczyna sie od 6.59
spy_pred = rfSPY.predict(X[-1:,])[0]

#Nasdaq
rfNDQ = joblib.load("/home/olaf/Documents/Modelowanie-AT/modele/randomForestNasdaq.pkl")

df = pd.read_csv("https://stooq.pl/q/d/l/?s=nq.f&i=d")

dataNDQ = df.iloc[-1,0]

df.drop("LOP", axis=1, inplace=True)
df = addTechnicalFeatures(df)

#dropujemy zmienne celu  i NA
X = df.drop(['target1','target3','target5'], axis=1)
X.dropna(inplace=True, axis=0, how="any")
#przeskalowanie danych do modelu(bo na takich byl budowany)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#Przewidywania nastepnego dnia. Gorne percentyli od 19.64
ndq_pred = rfNDQ.predict(X[-1:,])[0]

#DIJA US 30
rfDIJA = joblib.load("/home/olaf/Documents/Modelowanie-AT/modele/randomForestDIJA.pkl")

df  = pd.read_csv("https://stooq.pl/q/d/l/?s=ym.f&i=d")

dataDIJA = df.iloc[-1,0]

df.drop("LOP", axis=1, inplace=True)
df = addTechnicalFeatures(df)

#dropujemy zmienne celu  i NA
X = df.drop(['target1','target3','target5'], axis=1)
X.dropna(inplace=True, axis=0, how="any")
#przeskalowanie danych do modelu(bo na takich byl budowany)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
#20 percentyl to 63
dija_pred = rfDIJA.predict(X[-1:,])[0]

#Wysylanie maila z tymi wszystkimi danymi 
import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("muzgil@gmail.com", "1qazxsw2")
msg = ("\nSPY " + str(spy_pred.round(2)) + " Musi byc wieksze niz 6.59 " + "data: " + str(dataSPY) +
		 "\n" +
		 "NDQ " + str(ndq_pred.round(2)) + " Musi byc wieksze niz 19.64 " + "data: " + str(dataNDQ) +
		 "\n" +
		 "DIJA " + str(dija_pred.round(2)) + " Musi byc wieksze niz 63 " + "data: " + str(dataDIJA))  
print(msg)
server.sendmail("muzgil@gmail.com", "olafsk123@gmail.com", msg)