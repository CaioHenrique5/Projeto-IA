import csv
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.linear_model import LinearRegression

##Tratando o Price

def removeSecoundPrice(price):
  price = price.split(',')[0]
  price = price.replace('$', '')

  return float(price)

## Iniciando a Regressão
lr = LinearRegression()

## Lendo o banco de dados
Item = pd.read_csv("Item.csv").dropna()

## Estabelecendo o dados que vão ser usados no treinamento e na regressão
tempPrices = Item.apply(lambda row: removeSecoundPrice(row['prices']), axis=1)
Item['prices'] = Item.apply(lambda row: removeSecoundPrice(row['prices']), axis=1)
X = pd.DataFrame(Item, columns=['rating', 'totalReviews'])
y = pd.DataFrame(tempPrices)

## Treianndo os dados
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=9)
Treino = lr.fit(X_train,y_train)
Y_esperado = Treino.predict(X_test)

## Definindo X,Y e Z
X_pos = X_test["rating"]
Y_pos = X_test["totalReviews"]
Z_pos = y_test

fig = plt.figure()


plt.title("Teste")
plt.scatter(X_pos,Y_pos, Z_pos)
plt.xlabel('rating')
plt.ylabel('prices')
plt.zlabel('totalReviews')
plt.plot(X_test, Y_esperado, color= "red")
plt.show()





