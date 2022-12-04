import pandas as pd
from keras.layers import Input, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

x_clean = pd.read_csv('../data/x_train_cleaned.csv')
y_clean = pd.read_csv('../data/y_train_cleaned.csv')

# TODO: Deal with the Address. Maybe we can use it in a useful manner
x_clean.drop(['Address'], inplace=True, axis=1)

# TODO: We want to use these variables later on, but they probably need to be normalized first
x_clean.drop(['X', 'Y', 'year', 'month', 'day', 'hour', 'minute'], inplace=True, axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_clean, y_clean)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train['Resolution'])
y_test = label_encoder.transform(y_test['Resolution'])

model = Sequential(name='san_francisco_sequential')
model.add(Input(shape=x_train.shape[1]))
model.add(Dense(1000, activation='relu', name='hidden_layer'))
model.add(Dense(17, activation='softmax', name='output_layer'))

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.fit(
    x_train,
    y_train,
    batch_size=16,
    epochs=20,
    validation_split=.1
)

model.evaluate(x_test, y_test)
