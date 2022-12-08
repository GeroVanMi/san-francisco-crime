import pandas as pd
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

x_clean = pd.read_csv('../data/x_train_cleaned.csv')
y_clean = pd.read_csv('../data/y_train_cleaned.csv')

# TODO: Deal with the Address. Maybe we can use it in a useful manner
x_clean.drop(['Address'], inplace=True, axis=1)

# TODO: We want to use these variables later on, but they probably need to be normalized first
x_clean.drop(['X', 'Y'], inplace=True, axis=1)

label_encoder = LabelEncoder()
y_clean = label_encoder.fit_transform(y_clean['Category'])

x_train, x_test, y_train, y_test = train_test_split(x_clean, y_clean)

model = Sequential(name='san_francisco_sequential')
model.add(Input(shape=x_train.shape[1]))

print(x_train.shape)

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(39, activation='softmax'))

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

history = model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=20,
    validation_split=.1
)

for j in list(history.history.keys()):
    plt.plot(history.history[j])
    plt.title(j + ' over epochs')
    plt.ylabel(j)
    plt.xlabel('Epochs')
    plt.show()

print(model.evaluate(x_test, y_test))
