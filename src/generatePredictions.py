import keras
import pandas as pd

categories = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
              'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
              'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING', 'LARCENY/THEFT',
              'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT',
              'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE',
              'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS',
              'VANDALISM', 'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

model = keras.models.load_model('./small_model')
df = pd.read_csv('../data/test_cleaned.csv')

df.drop(['Address'], inplace=True, axis=1)
df.drop(['Id'], inplace=True, axis=1)

arr = df.to_numpy()

predictions = model.predict(arr)

for prediction in predictions:
    prediction_max = prediction.max()
    prediction[prediction != prediction_max] = 0
    prediction[prediction != 0] = 1

print(predictions.shape)

predictions = predictions.astype('int32')

df_predictions = pd.DataFrame(predictions, columns=categories)

df_predictions.to_csv('predictions.csv', index_label='Id')
