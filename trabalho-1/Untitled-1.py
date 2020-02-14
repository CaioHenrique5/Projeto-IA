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


lr = LinearRegression()
Item = pd.read_csv("Item.csv").dropna()
print(Item["totalReviews"])
X = Item['totalReviews','rating']
y = Item['prices']

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state=10)
Treino = lr.fit(x_train,y_train)

Treino.predict(x_test)
Treino.predict(y_test)

