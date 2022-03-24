'''
Brett Morris and Aiden Subers
3/24/22
'''

import pandas as pd
import pandas_technical_indicators as ti
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


#Smooths the data with a default alpha value of 0.9 that may be changed
def smoothing(data, alpha=0.9):
    return data.ewm(alpha=alpha).mean()

#Calculates and appends technical indicators to the dataframe using the pandas_technical_indicators library
def feature_extraction(data):
    for i in [7, 14, 28, 56, 112]:
        data = ti.relative_strength_index(data, n=i)
        data = ti.stochastic_oscillator_d(data, n=i)
        data = ti.accumulation_distribution(data, n=i)
        data = ti.average_true_range(data, n=i)
        data = ti.momentum(data, n=i)
        data = ti.money_flow_index(data, n=i)
        data = ti.rate_of_change(data, n=i)
        data = ti.on_balance_volume(data, n=i)
        data = ti.commodity_channel_index(data, n=i)
        data = ti.ease_of_movement(data, n=i)
        data = ti.trix(data, n=i)
        data = ti.vortex_indicator(data, n=i)
    data = ti.macd(data, n_fast=14, n_slow=28)
    del(data['Open'])
    del(data['High'])
    del(data['Low'])
    del(data['Volume'])
    return data
   
#Creates a prediction int to append onto the dataframe for the target date n days away
def compute_prediction_int(df, n):
    pred = (df.shift(-n)['Close'] >= df['Close'])
    pred = pred.iloc[:-n]
    return pred.astype(int)

#Calls feature_extraction and compute_prediction_int and cleans the resulting data
def prepare_data(df, target_date):
    data = feature_extraction(df).dropna().iloc[:-target_date]
    data['pred'] = compute_prediction_int(data, n=target_date)
    del(data['Close'])
    return data.dropna()

#Random Forest classifier that returns the accuracy of the model on test data
def classifier_RF(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data)
    rf = RandomForestClassifier(n_jobs=-1, n_estimators=80)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    accuracy = accuracy_score(pred, y_test)
    return(accuracy)

#Main function encapsulating imperative statements and additional cleaning
def main(target_date=50):
    data = pd.read_csv('AAPL.csv')
    del(data['Date'])
    del(data['Adj Close'])
    data_smooth = smoothing(data)
    data = prepare_data(data_smooth, target_date)
    y_data = data['pred']
    features = [x for x in data.columns if x not in ['gain', 'pred']]
    X_data = data[features]
    output = classifier_RF(X_data, y_data)
    print("Accuracy:", output)

#Add optional parameter target_date to main() call to make a next X-day prediction
if __name__ == "__main__":
    main()