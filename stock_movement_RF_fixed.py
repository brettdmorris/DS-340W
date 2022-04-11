'''
Brett Morris and Aiden Subers
4/10/22
'''

import pandas as pd
import numpy as np
import pandas_technical_indicators as ti
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


#Calculates and appends technical indicators to the dataframe using the pandas_technical_indicators library
#These are new features that are used instead of price and volume features from the dataset
def feature_extraction(data):
    for i in [10, 25, 50, 100, 200]:
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
#These are used in the output class as a binary of whether the stock price increased
def compute_prediction_int(df, n):
    pred = (df.shift(-n)['Close'] >= df['Close'])
    pred = pred.iloc[:-n]
    return pred.astype(int)

#Calls feature_extraction to generate features and compute_prediction_int and cleans then splits the resulting data into train/test splits
def prepare_data(df, target_date):
    data = feature_extraction(df).dropna().iloc[:-target_date]
    data['pred'] = compute_prediction_int(data, n=target_date)
    data = data.dropna()
    del(data['Close'])
    y_data = data['pred']
    features = [x for x in data.columns if x not in ['gain', 'pred']]
    X_data = data[features]
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=np.random.seed(42))
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    return X_train, X_test, y_train, y_test

#Random Forest classifier that returns the accuracy of the model given the split data and optionally calls grid_search for grid search cross validation
def classifier_RF(X_train, X_test, y_train, y_test, search):
    rf = RandomForestClassifier(n_estimators=80)
    if search==True:
        grid_search(rf, X_train, y_train)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    accuracy = accuracy_score(pred, y_test)
    return(accuracy)

#Optionally called function that does a grid search cross-validation on given values for applicable parameters and prints the optimal values
def grid_search(model, X_train, y_train):
    params = {'n_estimators':[80, 100, 120, 140],
              'max_depth': [None, 1, 2, 3],
              'bootstrap': [True, False],
              'min_samples_leaf': [1, 2, 3, 4]}
    gs = GridSearchCV(model, params)
    gs.fit(X_train, y_train)
    print(gs.best_params_)
    

#Main function encapsulating imperative statements and calling the classifier_GB function on the prepare_data function on the given dataset
#The optional arguments allow changing of the dataset, the target date for prediction, and whether to perform grid search cross validation
#Leave search=False to only run the model on the given data
def main(dataset='AAPL.csv', target_date=10, search=False):
    data = pd.read_csv(dataset)
    del(data['Date'])
    del(data['Adj Close'])
    output = classifier_RF(*prepare_data(data, target_date), search)
    print("Accuracy:", output)


if __name__ == "__main__":
    main()