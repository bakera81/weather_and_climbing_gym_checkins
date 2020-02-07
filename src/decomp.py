import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import psycopg2
import getpass
import statsmodels.api as sm
from datetime import timedelta

sys.path.append('/home/mike/dsi/capstones/public-repos/climbing_gym_checkins')
from src.paths import data_path, prj_path
from src.funcs import get_freq, resample

class Decomp:
    
    def __init__(self, data):
        
        # self.data = pd.DataFrame(data)
        self.data = data
        
        index_len = len(self.data.columns)
        if index_len > 1:
            self.data.columns = ['raw_data'] + [col for col in data.columns[1:]]
        else:
            self.data.columns = ['raw_data']
        
        self.freq = get_freq(data)
    
    def _to_col_vector(self, arr):
        """
        Convert a one dimensional numpy array to a column vector.
        """
        return arr.reshape(-1, 1)

    def _make_design_matrix(self, arr):
        """
        Construct a design matrix from a numpy array, including an intercept term.
        """
        return sm.add_constant(self._to_col_vector(arr), prepend=False)

    def fit_linear_trend(self):
        """
        Fit a linear trend to a time series.  Return the fit trend as a DataFrame.
        """
        series = self.data.iloc[:, 0].copy()
        
        X = self._make_design_matrix(np.arange(len(series)) + 1)
        
        linear_trend_ols = sm.OLS(series.values, X).fit()
        linear_trend = linear_trend_ols.predict(X)
        
        return pd.Series(linear_trend, index=self.data.index)
    
    def fit_annual_trend(self):
        """
        Fit annually resampled mean as trend to time series. Return the fit trend as a DataFrame.
        """
        df = self.data.copy()
        
        try:
            df.index = df.index.to_timestamp()
        except AttributeError:
            pass
        
        annual_trend_h = df.raw_data.resample('A').mean().resample('H').interpolate()
        annual_trend_res = resample(annual_trend_h, self.freq, np.mean)
        annual_trend_res.name = 'trend_annual'
                           
        annual_trend_df = df.merge(annual_trend_res, right_index=True, left_index=True, how='outer')

        annual_trend_df = annual_trend_df.sort_index()
        
        return annual_trend_df.trend_annual
        
    def decomp_linear(self):
        """
        Return dataframe of raw_data, trend component, seasonal components with linear trend.
        """
        df = self.data.copy()
 
        df['trend_linear'] = self.fit_linear_trend()
        df['seasonal_linear'] = df.raw_data - df.trend_linear
        
        return df
    
    def decomp_annual(self):
        """
        Return dataframe of raw_data, trend component, seasonal components with linear trend.
        """
        df = self.data.copy()
        df['trend_annual'] = self.fit_annual_trend()
        df['seasonal_annual'] = df.raw_data - df.trend_annual
        
        return df
        
    def decomp(self, linear=True, annual=True):
        """
        Return dataframe of raw_data, trend component, seasonal components. Options for method used to calculate trend.
        """
        df = self.data.copy()
        
        if linear:
            df_lin = self.decomp_linear()
            
            df_concat = pd.concat([df, df_lin[['trend_linear', 'seasonal_linear']]], axis=1)
        
        if annual:
            df_ann = self.decomp_annual()

            df_concat = pd.concat([df_concat, df_ann[['trend_annual', 'seasonal_annual']]], axis=1)
            
            df_concat.index = df.index
            
        return df_concat
    
if __name__ == '__main__':
    
    pass