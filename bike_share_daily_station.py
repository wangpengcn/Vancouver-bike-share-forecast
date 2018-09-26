# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 13:27:03 2018

@author: Peng

1. How many bikes are rented out each hour (demand)
2. How many bikes are rented and returned each day for high-traffic location (num>300)

Temperature: provided in the original data
Other weather info and air quality: BC Air Data Archive https://www2.gov.bc.ca/gov/content/environment/air-land-water/air/air-quality/current-air-quality-data/bc-air-data-archive
Used observations from YVR monitoring station, instead of Vancouver Harbour station which doe not have air quality or wind speed

"""

import pandas as pd
import numpy as np
import glob, pickle 
from datetime import date, timedelta, datetime
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer, mean_squared_error # import metrics from sklearn
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import ShuffleSplit # Random permutation cross-validator

'''
Update column names on original datasets to be consistant
201801	rename 'Formula' to 'Membership type' and remove blanks after
201804	rename 'ID' to 'Account'
201807	rename 'Membership Type' to 'Membership type'
'''
sns.set()
# Date range 1/1/2013 - 08/31/2018. Use 08/25-08/31/2018 for testing.
BEGIN_DATE = date(2017,1,1)
END_DATE = date(2018,7,31)
TEST_START_DATE = date(2018,7,25)
TEST_END_DATE = date(2018,7,31)
PATH_DATA =r'./data' 
PATH_MODEL =r'./models' 
# Load bike share data
data_files = glob.glob(PATH_DATA + "/Mobi*.xlsx")
data_orig = pd.DataFrame()
list_ = []
for file_ in data_files:
    df = pd.read_excel(file_)
    list_.append(df)
data_orig = pd.concat(list_, ignore_index=True, sort=False)

#data_orig.isnull().sum()
data_orig['Date'] = data_orig['Departure'].dt.date
data_orig = data_orig[(BEGIN_DATE<=data_orig['Date']) & (data_orig['Date']<=END_DATE)]
data_orig['Year'] = data_orig['Departure'].dt.year
data_orig['Month'] = data_orig['Departure'].dt.month
data_orig['DayOfMonth'] = data_orig['Departure'].dt.day
data_orig['DayOfWeek'] = data_orig['Departure'].dt.dayofweek
data_orig['HourOfDay'] = data_orig['Departure'].dt.hour
seasons = {1:'Winter', 2:'Winter', 3:'Spring', 4:'Spring', 5:'Spring', 6:'Summer', 7:'Summer', 8:'Summer', 
           9:'Autumn', 10:'Autumn', 11:'Autumn', 12:'Winter'}
data_orig['Season'] = [seasons[x] for x in data_orig['Month'].values]

# Extract bike station stats
stations_depart = data_orig.groupby(['Year','Season','Month','DayOfMonth','Date','Departure station'], as_index=False).agg(
                       {'Departure':'count'})
stations_depart.rename(columns={'Departure station':'Station'}, inplace=True)
stations_return = data_orig.groupby(['Year','Season','Month','DayOfMonth','Date','Return station'], as_index=False).agg(
                       {'Return':'count'})
stations_return.rename(columns={'Return station':'Station'}, inplace=True)
stations = pd.merge(stations_depart, stations_return, how='left', on=['Year','Season','Month','DayOfMonth','Date','Station'])
stations['Return'].fillna(0.0, inplace=True)

# Get weather info
weather_vancouver = pd.read_csv(PATH_DATA+'/Weather_YVR_20170101_20180731.csv')
# Remove extra spaces from header
weather_vancouver.rename(columns=lambda x: x.strip(), inplace=True)
# Remove extra spaces from columns
weather_vancouver[weather_vancouver.columns] = weather_vancouver.apply(lambda x: x.str.strip())
# Change midnight datetime to same format as data_orig
midnight = weather_vancouver[weather_vancouver['Date Time'].str.contains('24:00')]['Date Time']
midnight_date = pd.to_datetime(midnight.str.split(expand=True)[0]) + timedelta(1)
weather_vancouver.loc[midnight.index, 'Date Time'] = midnight_date.astype(str)
weather_vancouver['Date Time'] = pd.to_datetime(pd.to_datetime(weather_vancouver['Date Time']).dt.strftime('%Y/%m/%d %H:%M:%S'))
# Missing entry for the first hour of the first day - copy the next entry
weather_vancouver = pd.concat([weather_vancouver.head(1), weather_vancouver], axis=0, ignore_index=True)
# Update to the correct time i.e. 00:00:00
weather_vancouver['Date'] = weather_vancouver['Date Time'].dt.date
weather_vancouver['TEMP_MEAN'] = pd.to_numeric(weather_vancouver['TEMP_MEAN'])
daily_ave_temp = weather_vancouver.groupby(['Date'], as_index=False).agg({'TEMP_MEAN':'mean'})
stations = pd.merge(stations, daily_ave_temp, how='left', on=['Date'])
stations.rename(columns={'TEMP_MEAN':'Temperature'}, inplace=True)
stations['Daily departure median'] = stations.groupby(['Station'])['Departure'].transform(lambda x: x.median())

# -----------  Data visualization ----------
ax = sns.catplot(y='Station', x='Departure', 
            data=stations[stations['Daily departure median'] >= 25.0]
            .sort_values(by='Daily departure median', ascending=False), kind='box', orient ='h', height=8)
#ax.set_xticklabels(rotation=90, size=10)
ax.fig.suptitle('Top bike rental locations \n{} to {}'.format(BEGIN_DATE, END_DATE),fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.92])

sns.lineplot(x='Date', y='Departure', data=stations[stations['Daily departure median'] >= 25.0]
            .sort_values(by='Daily departure median', ascending=False), hue='Station')

# Get BC holidays
# Data from https://www.officeholidays.com/
holidays_bc = pd.read_csv('./data/holidays_bc.csv')
holidays_bc.rename(columns={'Holiday':'Holiday_bc'}, inplace=True)
holidays_bc['Date'] = pd.to_datetime(holidays_bc['Date']).dt.date
data = pd.merge(stations, holidays_bc[['Date','Holiday_bc']], how='left', on='Date')
data['Holiday_bc'].fillna('', inplace=True)

# Let's just focus on stations where have sufficient rentals
data = data[data['Daily departure median'] >= 8.0]
# ********************** DEBUG *******************
data = data[data['Station'] =='0209 Stanley Park - Information Booth']
# ********************** DEBUG END *******************
data.rename(columns={'Departure':'Number of bikes rented'}, inplace=True)
# Convert categorical columns
data = pd.get_dummies(data, columns=['Holiday_bc','Season'])
# Drop PM25 and WindSpeed as they do not appear to contribute prediction on bike rentals
data.drop(columns=['Holiday_bc_','Station','Return'], inplace=True)

sns.lineplot(x='Date', y='Number of bikes rented', data=data)
# ---------------------------------------------------
# ------------- Train Model -------------------------
# ---------------------------------------------------
# *** Split training/test data ***
train_data = data[data['Date'] < TEST_START_DATE]
test_data = data[(data['Date'] >= TEST_START_DATE) & (data['Date'] <= TEST_END_DATE)]
test_data.reset_index(inplace=True, drop=True)
train_data = train_data.drop(['Date'], axis=1)
test_data = test_data.drop(['Date'], axis=1)
train_x = train_data.drop(['Number of bikes rented'], axis=1)
train_y = train_data['Number of bikes rented']
test_x = test_data.drop(['Number of bikes rented'], axis=1)
test_y = test_data['Number of bikes rented']

# *** Random Forest Regression ***
rf_regressor = RandomForestRegressor(random_state=29, n_jobs=-1)
# Random grid search on hyperparameters
cv_sets = ShuffleSplit(random_state = 4) # shuffling for cross-validation
rf_parameters = {'n_estimators':[60, 80, 100], 
                 'min_samples_leaf':[1, 2, 3], 
                 'max_depth':[8, 10, 12],
                 'min_impurity_decrease':[0.0],
                 'min_samples_split':[2]}
scorer = make_scorer(r2_score)
n_iter = 24
rf_grid_search = RandomizedSearchCV(rf_regressor, 
                              rf_parameters, 
                              n_iter = n_iter, 
                              scoring = scorer, 
                              cv = cv_sets,
                              random_state= 50)
start = time() # start time
rf_grid_fit = rf_grid_search.fit(train_x, train_y)
rf_best = rf_grid_search.best_estimator_
end = time() # end time
# Calculate training time
rf_time = (end-start)/60.
print('---------------------------------------')
print('Took {0:.2f} minutes to find optimized parameters for RF model'.format(rf_time))
print('Best parameters for RF model: {}'.format(rf_grid_fit.best_params_))
print('---------------------------------------')

# *** XGBoost Regression ***
xgb_regressor = xgb.XGBRegressor(random_state=29, n_jobs=-1)
cv_sets = ShuffleSplit(random_state = 4) # shuffling for cross-validation
xgb_parameters = {'objective': ['reg:linear'],
                  'n_estimators': [60, 80, 100],
                  'learning_rate': [0.01, 0.01, 0.1, 1],
                  'gamma': [0.01, 0.2, 1.0],
                  'reg_lambda': [0.01, 0.5, 0.1],
                  'max_depth': [5,7,9], # Max depth of tree. Deeper -> overfitting
                  'subsample': [0.5, 0.6, 0.7], # Subsample ratio of training instances
                  'colsample_bytree': [0.6, 0.7, 0.8], # Subsample ratio of columns of each tree
                  'silent': [0] # Printing running msg
                  }
scorer = make_scorer(r2_score)
n_iter = 24
xgb_grid_search = RandomizedSearchCV(xgb_regressor, 
                              xgb_parameters, 
                              n_iter = n_iter, 
                              scoring = scorer, 
                              cv = cv_sets,
                              verbose=1,
                              random_state= 50)
start = time() # start time
xgb_grid_fit = xgb_grid_search.fit(train_x, train_y)
xgb_best = xgb_grid_search.best_estimator_
end = time() # end time
# Calculate training time
xgb_time = (end-start)/60.
print('---------------------------------------')
print('Took {0:.2f} minutes to find optimized parameters for XGB model'.format(xgb_time))
print('Best parameters for XGB model: {}'.format(xgb_grid_fit.best_params_))
print('---------------------------------------')

# ---------------------------------------------------
# ------------- Prediction -------------------------
# ---------------------------------------------------
# Prediction using RF
rf_preds = rf_best.predict(test_x).astype(int) # Convert fractions to whole numbers
# RF R2-score and MSLE
rf_r2 = r2_score(test_y, rf_preds)
rf_msle = np.sqrt(mean_squared_error(test_y, rf_preds))
# Prediction using XGB
xgb_preds = xgb_best.predict(test_x).astype(int) # Convert fractions to whole numbers
xgb_r2 = r2_score(test_y, xgb_preds)
xgb_msle = np.sqrt(mean_squared_error(test_y, xgb_preds))
# Compare the scores
regres_perform = {'R2': [rf_r2,xgb_r2],
                  'MSLE': [rf_msle,xgb_msle]} 
index_name = ['RF', 'XGB']
regres_perform = pd.DataFrame(data=regres_perform, index=index_name)
regres_perform

# Visualize XGBoost prediction
pred = pd.concat([test_data, pd.DataFrame({'Prediction':xgb_preds})], axis=1)
NUM_SUBPLOTS = TEST_END_DATE.day - TEST_START_DATE.day + 1
fig, axes = plt.subplots(NUM_SUBPLOTS,1, figsize=(18,9),sharex=True)
fig.suptitle('Predicted number of bikes rented \n{} to {}'.format(TEST_START_DATE, TEST_END_DATE))
for i, ax in enumerate(axes):    
    curr_pred = pred.loc[pred['DayOfMonth']==TEST_START_DATE.day + i]
    ax.plot(curr_pred['HourOfDay'], curr_pred['Number of bikes rented'], 'o-')
    ax.plot(curr_pred['HourOfDay'], curr_pred['Prediction'], 'o-')
    ax.set_xticks(curr_pred['HourOfDay'])
    ax.set_ylabel(TEST_START_DATE + timedelta(days=i))    
axes[int(NUM_SUBPLOTS/2)].figure.text(0.05,0.5, "Number of bikes rented", \
    ha="center", va="center", rotation=90, fontsize='large')
axes[0].legend(loc='upper left')
plt.xlabel('Hour of the day', fontsize='large')
# Plot important feature scores
xgb.plot_importance(xgb_best)

# Save prediction and model
pickle.dump(xgb_best, open(PATH_DATA + '/bike_rental_xgb.model','wb'))
pred.to_csv(PATH_MODEL+'/predictions.csv', index=False)