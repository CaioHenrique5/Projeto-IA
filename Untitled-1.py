import csv
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from matplotlib.collections import LineCollection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_validate
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
ridge = RidgeClassifier()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=9)
Treino = lr.fit(X_train,y_train)

Y_esperado = Treino.predict(X_test)



## Definindo X,Y e Z
xs = X_test["rating"]
ys = X_test["totalReviews"]
zs = y_test

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.title("Teste")
plt.xlabel('rating')
plt.ylabel('prices')
ax.scatter(xs, ys, zs, zdir='z')
ax.plot(xs, Y_esperado)
plt.show()





