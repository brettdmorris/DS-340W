'''
Brett Morris and Aiden Subers
4/21/22
'''

import pandas as pd
import pandas_technical_indicators as ti
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


#Calculates and appends technical indicators to the dataframe using the pandas_technical_indicators library
#These are new features that are used instead of price and volume features from the dataset
def feature_extraction(data, tar):
    data = ti.relative_strength_index(data, tar)
    data = ti.stochastic_oscillator_d(data, tar)
    data = ti.accumulation_distribution(data, tar)
    data = ti.average_true_range(data, tar)
    data = ti.momentum(data, tar)
    data = ti.money_flow_index(data, tar)
    data = ti.rate_of_change(data, tar)
    data = ti.on_balance_volume(data, tar)
    data = ti.commodity_channel_index(data, tar)
    data = ti.ease_of_movement(data, tar)
    data = ti.trix(data, tar)
    data = ti.vortex_indicator(data, tar)
    data['ema5'] = data['Close'] / data['Close'].ewm(tar).mean()
    data = ti.macd(data, n_fast=12, n_slow=26)
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
    data = feature_extraction(df, target_date).dropna().iloc[:-target_date]
    data['pred'] = compute_prediction_int(data, n=target_date)
    data = data.dropna()
    del(data['Close'])
    y = data['pred']
    features = [x for x in data.columns if x not in ['gain', 'pred']]
    X = data[features]
    splits = 2 * len(X) // 3
    X_train = X[:splits]
    X_test = X[splits:]
    y_train = y[:splits]
    y_test = y[splits:]
    return X_train, X_test, y_train, y_test

#Gradient Boosting classifier that returns the accuracy of the model given the split data and optionally calls grid_search for grid search cross validation
def classifier_GB(X_train, X_test, y_train, y_test, search):
    boost = GradientBoostingClassifier(n_estimators=80, random_state=42)
    if search==True:
        grid_search(boost, X_train, y_train)
    boost.fit(X_train, y_train)
    pred = boost.predict(X_test)
    accuracy = accuracy_score(pred, y_test)
    return(accuracy)

#Optionally called function that does a grid search cross-validation on given values for applicable parameters and prints the optimal values
def grid_search(model, X_train, y_train):
    params = {'n_estimators': [60, 80, 100],
              'learning_rate': [0.01, 0.1, 1, 10],
              'min_samples_leaf': [1, 2, 3],
              'max_depth': [3, 4, 5]}
    gs = GridSearchCV(model, params)
    gs.fit(X_train, y_train)
    print(gs.best_params_)
    

#Main function encapsulating imperative statements and calling the classifier_GB function on the prepare_data function on the given dataset
#The optional arguments allow changing of the dataset, the target date for prediction, and whether to perform grid search cross validation
#Leave search=False to only run the model on the given data
def main(dataset='AAPL.csv', target_date=30, search=False):
    data = pd.read_csv(dataset)
    del(data['Date'])
    del(data['Adj Close'])
    output = classifier_GB(*prepare_data(data, target_date), search)
    print("Accuracy:", output)


if __name__ == "__main__":
    main()
    #for i in range(30,95,5):
        #main(target_date=i)