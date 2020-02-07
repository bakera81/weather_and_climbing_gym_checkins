import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import psycopg2
from scipy import stats
sys.path.append('/home/mike/dsi/capstones/climbing_gym_checkins_eda')
from src.paths import data_path, prj_path, img_path
from src.funcs import conditions_dict, resample, parse_datetime
from src.decomp import Decomp
from src.sql_exec import SqlExec

class HypTestWeather:
    
    def __init__(self, series, good_weather, equal_var=True):
        
        self.series = series
        
        if any(self.series.isna()):
            raise ValueError('Series can not contain null values')
        
        self.good_series = self.series[good_weather]
        self.good = MeanDist(self.good_series)

        self.bad_series = self.series[(~good_weather)]
        self.bad = MeanDist(self.bad_series)
        
        self.mean_diff = self.bad.mean - self.good.mean
        
        self.N = len(series)
        
        self.t, self.p = stats.ttest_ind(self.good_series, self.bad_series, equal_var=equal_var)
                
    def plot(self, ax, bad_label, good_label, title, group):
        
        sns.distplot(self.good.series, kde=True, rug=False, hist=False, ax=ax, label=good_label, color='green')
        sns.distplot(self.bad.series, kde=True, rug=False, hist=False, ax=ax, label=bad_label, color='brown')
        
        ax.set_xticks([self.good.mean, self.bad.mean])
        ax.set_xticklabels(['$\mu_{good}$', '$\mu_{bad}$'])
        ax.set_yticks([], [])
        
        if title:
            ax.set_title(title)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(None)
        ax.axvline(self.good.mean, color = 'green', linestyle = '--', label='Good Weather Mean $\mu$')
        ax.axvline(self.bad.mean, color = 'brown', linestyle = '--', label='Bad Weather Mean $\mu$')
        
        p_annotated = np.format_float_scientific(self.p, precision=3)
        alpha_annotated = np.format_float_scientific(0.0041666, precision=3)
        
        ax.annotate(
            '$\mu_{bad} - \mu_{good}$ =' + 
            f''' {self.mean_diff:.0f}\n\n
            There are {self.mean_diff:.0f} more checkins\n
            per day during {bad_label.lower()}\n
            on {group} with a p-value:\n
            p = {p_annotated}\n
            α = {alpha_annotated}\n
            N = {self.N}''',
            xy=(0.7, 0.3), xycoords='axes fraction')
        
class MeanDist:
    
    def __init__(self, series):
        
        self.series = series
        self.mean = np.mean(series)
        self.sem = stats.sem(series)
        self.dist = stats.norm(loc=self.mean, scale=self.sem)
        self.interval = self.dist.interval(0.05)
        
    def plot(self, ax, samples=1000):
        
        x_min = self.mean - 4 * self.sem
        x_max = self.mean + 4 * self.sem
        x = np.linspace(x_min, x_max, 1000)
        y = self.dist.pdf(x)
        
        ax.plot(x, y)

def run_hyp_tests(gym_prefix, sql=None):
    
    if not sql:
        sql = SqlExec(gym_prefix)
    
    ## Weather data
    
    df_w_hours = sql.weather(parse_datetime=True)
    df_w_days = df_w_hours.groupby('date').agg({'temp_f':np.max, 'precip_in':np.sum, 'snow_depth_in':np.mean})
    
    df_w_days.index = pd.to_datetime(df_w_days.index).to_period('D')
    
    # Good temps condition
    good_temps_cond = ((df_w_days.temp_f > 50) & (df_w_days.temp_f < 90))
    good_precip_cond = (df_w_days.precip_in == 0)
    good_snow_cond = (df_w_days.snow_depth_in == 0)
    
    weather_strs = ['temp', 'rain', 'snow']
    
    good_weather_conditions = dict(zip(weather_strs, [good_temps_cond, good_precip_cond, good_snow_cond]))
    
    ## Get counts data from SQL executer
    cnt_days = sql.day_counts()
    cnt_days = cnt_days.to_timestamp()

    # Decompose time series and get seasonal part by removing the linear fit
    days_decomp = Decomp(cnt_days)
    df_decomp = days_decomp.decomp()
    
    if gym_prefix ==  'gol':
        seasonal_ser = df_decomp.seasonal_annual
    if gym_prefix == 'eng':
        seasonal_ser = df_decomp.seasonal_linear
    
    # Add hours, day of week, date, etc. columns to seasonal series
    df_seasonal = parse_datetime(seasonal_ser, col='index')
    df_filt = conditions_dict(df_seasonal, dt_col='date', season=False)
    
    seasonal_ser = seasonal_ser.to_period('D')
    
    # Set up output
    results_raw = {}
    
    # Pandas Series to test
    weekend_cond = (df_filt['dow']['sat'] | df_filt['dow']['sun'])
    mon_thu_cond = (~(weekend_cond) & ~(df_filt['dow']['fri']))
    
    weekends = seasonal_ser[weekend_cond]
    results_raw['weekends'] = weekends
    
    weekdays = seasonal_ser[mon_thu_cond]
    results_raw['weekdays'] = weekdays
    
    results = {}
    
    for weather, good_weather in good_weather_conditions.items():
        for k, s in results_raw.items():
            if weather in results.keys():
                results[weather].update({k:HypTestWeather(s.dropna(), good_weather)})
            else:
                results.update({weather:{k:HypTestWeather(s.dropna(), good_weather)}})
            
    return results
        
        
if __name__ == '__main__':
    
    # g_results = run_hyp_tests('gol')
    # e_results = run_hyp_tests('eng')
        
    # print(g_results)
    # print(e_results)

    # print('GOLDEN')
    # print('\n'.join([f'{k}: mu_diff = {res["weekdays"].mean_diff}, p = {res["weekdays"].p}, N = {res["weekdays"].N}' for k, res in g_results.items()]))

    # print('ENGLEWOOD')
    # print('\n'.join([f'{k}: mu_diff = {res["weekdays"].mean_diff}, p = {res["weekdays"].p}, N = {res["weekdays"].N}' for k, res in e_results.items()]))

    # fig, ax = plt.subplots(figsize=(10, 10))
        
    # g_results['rain']['weekends'].series.plot()
        
    # plt.show()
    
    pass