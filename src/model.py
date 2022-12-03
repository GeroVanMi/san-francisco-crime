import pandas as pd
from keras.models import Sequential
from keras.layers import Input, Dense

train_data = pd.read_csv('../data/train.csv')

model = Sequential()
model.add(Input())
model.add(Dense())
model.add(Dense())