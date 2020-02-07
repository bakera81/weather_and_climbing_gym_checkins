import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import timedelta
sys.path.append('/home/mike/dsi/capstones/climbing_gym_checkins_eda')
from src.paths import data_path, prj_path
from src.funcs import conditions_dict, resample, parse_datetime, get_freq
from src.decomp import Decomp
from src.hyp_test import HypTestWeather
from src.sql_exec import SqlExec
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import statsmodels.api as sm

class DesignMatrix():
    
    def __init__(self, sql, counts='day', checks=False):
        
        self.sql = sql
        
        if counts[0].lower() == 'h':
            self.df = self.sql.hour_counts()
        if counts[0].lower() == 'd':
            self.df = self.sql.day_counts()
        
        if checks:
            raise NameError("Use counts 'day' or 'hour'")
        
        self.freq = get_freq(self.df)
        
        self.weather = False
        
    def add_weather(self, y_cols=['tot_checkins', 'mem_checkins', 'nmem_checkins'], 
                   w_cols = ['temp_f', 'wind_lo_mph', 'precip_in', 'snow_depth_in'],  
                   agg_dict = {'temp_f':np.max, 'wind_lo_mph':np.mean, 'precip_in':np.sum, 'snow_depth_in':np.mean}):
        
        if self.weather:
            raise AttributeError('Weather was already added. Set self.weather to False to bypass')
            
        df = self.df
        
        if self.freq == 'D':
            weather = resample(self.sql.weather(), self.freq, agg_dict)
        else:
            raise NameError("Haven't set this up yet!")
        
        weather = weather[w_cols]
        
        weather.index = weather.index.to_period(self.freq)
        
        self.df = df.join(weather)
        self.weather = True
        
        return self.df

    def get_date_splits(self, return_bool = False):
        
        df = self.df
        
        e_open = pd.to_datetime('2018-08-30')

        hold_dates_bool = (df.index < e_open) & (df.index > (e_open - timedelta(days=365)))
        test_dates_bool = (df.index < e_open - timedelta(days=365)) & (df.index > (e_open - 2 * timedelta(days=365)))
        train_dates_bool = (df.index < e_open - 2 * timedelta(days=365))
        
        if return_bool:
            return train_dates_bool, test_dates_bool, hold_dates_bool
        
        hold_dates = df.index[hold_dates_bool]
        test_dates = df.index[test_dates_bool]
        train_dates = df.index[train_dates_bool]
        
        return train_dates, test_dates, hold_dates
    
    def train_test_split(self, X_cols, y_col):
        
        df = self.df
        
        train_dates, test_dates, hold_dates = self.get_date_splits(return_bool=True)
        
        hold_df = df[hold_dates]
        test_df = df[test_dates]
        train_df = df[train_dates]
        
        df_list = [train_df, test_df, hold_df]
        X_list = []
        y_list = []
        
        for df in df_list:
            X_list.append(df[X_cols].values)
            y_list.append(df[y_col].values)
        
        return X_list, y_list, df_list

class Model():
    
    def __init__(self, model, X_train, X_test, y_train, y_test, train_dates, test_dates):
        
        self.model = model
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.y_train_ser = self.to_series(y_train, train_dates)
        self.y_test_ser = self.to_series(y_test, test_dates)
        
        self.train_dates = train_dates
        self.test_dates = test_dates

        self.decomp_train = Decomp(pd.DataFrame(self.y_train_ser)).decomp_linear()
        self.decomp_test = Decomp(pd.DataFrame(self.y_test_ser)).decomp_linear()
        
        self.model.fit(self.X_train, self.decomp_train.seasonal_linear)
        self.lin_model, self.intercept = self.fit_linear_trend(self.y_train_ser)
    
    def to_series(self, arr, dates):
        
        return pd.Series(arr, index=dates)
    
    def destandard(self, pred, trend):
            
        return trend + pred
    
    def predict_seasonal(self, X):
        
        return self.model.predict(X)

    def _to_col_vector(self, arr):

        return arr.reshape(-1, 1)

    def _make_design_matrix(self, arr):

        return sm.add_constant(self._to_col_vector(arr), prepend=False)

    def fit_linear_trend(self, y_ser):

        series = y_ser
        
        X = self._make_design_matrix(np.arange(len(series)) + 1)
        
        linear_trend_ols = sm.OLS(series.values, X).fit()
        
        return linear_trend_ols, X[-1, 0]
    
    def predict_linear(self, dates, intercept=0):
        
        X = self._make_design_matrix(np.arange(len(dates)) + 1 + intercept)

        return pd.Series(self.lin_model.predict(X), index=dates)
            
    def predict(self, X, dates, intercept=False):
        
        if isinstance(intercept, bool):
            intercept = self.intercept * intercept
        
        pred_standard = self.predict_seasonal(X)
        
        pred_trend = self.predict_linear(dates, intercept)
                    
        preds = self.destandard(pred_standard, pred_trend)
        
        return preds
    
    def predict_all(self):
        
        self.train_preds = self.predict(self.X_train, self.train_dates)
        self.test_preds = self.predict(self.X_test, self.test_dates, intercept=True)
        
        return self.train_preds, self.test_preds
    
    def score(self, y_true, y_pred, rmse=True, r2=True, mae=False):

        errors = {}
        if rmse:
            errors['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        if r2:
            errors['r2'] = r2 = r2_score(y_true, y_pred)
        if mae:
            errors['mae'] = mean_absolute_error(y_true, y_pred)
        
        return errors
    
    def score_train(self, rmse=True, r2=True, mae=True):
        
        return self.score(self.y_train, self.train_preds, rmse, r2, mae)
    
    def score_test(self, rmse=True, r2=True, mae=True):
        
        return self.score(self.y_test, self.test_preds, rmse, r2, mae)

class Predictor():
    
    def __init__(self, model, df, X_cols, y_col='tot_checkins'):
        
        self.df = df
        self.X_cols = X_cols
        self.y_col = y_col
        
        X_list, y_list, df_list = self.train_test_split(df, X_cols, y_col)
        
        dates = [df.index for df in df_list]
        self.train_dates, self.test_dates, self.hold_dates = dates
        self.X_train, self.X_test, self.X_hold = X_list
        self.y_train, self.y_test, self.y_hold = y_list
        
        self.model_test = Model(model, X_train, X_test, y_train, y_test, self.train_dates, self.test_dates)
        self.model_hold = Model(model, X_train, X_hold, y_train, y_hold, self.train_dates, self.hold_dates)

        self.y_train_ser, self.y_test_ser, self.y_hold_ser = [self.model_test.to_series(y, dates) for y, dates in zip(y_list, dates)]
        
        self.train_pred, self.test_pred = self.model_test.predict_all()
        self.train_pred, self.hold_pred = self.model_hold.predict_all()

    def train_test_split(self, df, X_cols, y_col):
        
        e_open = pd.to_datetime('2018-08-30')
        
        hold_df = df[(df.index < e_open) & (df.index > (e_open - timedelta(days=365)))]
        test_df = df[(df.index < e_open - timedelta(days=365)) & (df.index > (e_open - 2 * timedelta(days=365)))]
        train_df = df[(df.index < e_open - 2 * timedelta(days=365))]
        
        df_list = [train_df, test_df, hold_df]
        X_list = []
        y_list = []
        
        for df in df_list:
            X_list.append(df[X_cols].values)
            y_list.append(df[y_col].values)
        
        return X_list, y_list, df_list    
        
    def pred_to_plot(self, train=False, test=True, hold=False):
        
        counts_sers = []
        labels = []
        
        if train:
            counts_sers.append(self.train_pred)
            labels.append('Predicted training data')
        if test:
            counts_sers.append(self.test_pred)
            labels.append('Predicted testing data')
        if hold:
            counts_sers.append(self.hold_pred)
            labels.append('Predicted holdout data')
        
        return counts_sers, labels
    
    def data_to_plot(self, train=True, test=True, hold=False):
    
        counts_sers = []
        labels = []
        
        if train:
            counts_sers.append(self.y_train_ser)
            labels.append('True training data')
        if test:
            counts_sers.append(self.y_test_ser)
            labels.append('True test data')
        if hold:
            counts_sers.append(self.y_hold_ser)
            labels.append('True holdout data')
        
        return counts_sers, labels
    
    def plot(self, ax, preds=[False, True, False], data=[True, True, False]):
        
        preds, pred_labels = self.pred_to_plot(*preds)
        data, data_labels = self.data_to_plot(*data)
        alphas = [1] * len(preds) + [1] * len(data)
        
        for counts, label, alpha in zip(data + preds, data_labels + pred_labels, alphas):
            counts_ts = counts.to_timestamp()
            ax.plot(counts_ts, label=label, alpha=alpha)
    
if __name__ == '__main__':
    
    sql = SqlExec('gol')
    
    day_design = DesignMatrix(sql)
    
    df = day_design.add_weather()
    
    X_cols = ['dow', 
              'week', 
              'month', 
              'season', 
              'is_holiday', 
              'temp_f', 
              'wind_lo_mph',
              'precip_in', 
              'snow_depth_in'
              ]
    
    X_list, y_list, df_list = day_design.train_test_split(X_cols=X_cols, y_col='tot_checkins')
    X_train, X_test, X_hold = X_list
    y_train, y_test, y_hold = y_list
    df_train, df_test, df_hold = df_list
    
    model = Model(RandomForestRegressor(n_estimators=1000), X_train, X_test, y_train, y_test, df_train.index, df_test.index)
    
    y_pred_test, y_pred_train = model.predict_all()
    
    print(model.score_train(), model.score_test())

    rf_predictor = Predictor(RandomForestRegressor(n_estimators=1000, max_features='sqrt'), df, X_cols=X_cols, y_col='tot_checkins')
    
    fig, ax = plt.subplots(figsize=(30, 20))
    rf_predictor.plot(ax, preds=[True, True, False])
    ax.legend()
    plt.tight_layout()
    plt.show()