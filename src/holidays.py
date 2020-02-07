import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import psycopg2
import calmap
from datetime import datetime
from datetime import timedelta
from scipy import stats
sys.path.append('/home/mike/dsi/public-repos/climbing_gym_checkins')
from src.paths import data_path, prj_path
from src.funcs import conditions_dict, resample, parse_datetime

class Holidays:
    
    def __init__(self, start_year=2013, end_year=2020):
        
        df = pd.read_csv(os.path.join(data_path, 'misc/usholidays.csv'), parse_dates=['Date'], index_col='Date')
        df = df[(df.index.year >= start_year) & (df.index.year < end_year)]
        df = df.drop('Unnamed: 0', axis=1)
        self.df = parse_datetime(df)

    def get_date_range_from_week(self, p_year,p_week):
        
        firstdayofweek = datetime.strptime(f'{p_year}-W{int(p_week )- 1}-1', "%Y-W%W-%w").date()
        
        return [firstdayofweek + timedelta(days=days) for days in range(7)]

    def holidays(self):
        
        df = self.df
        
        mon_wknd = ['Memorial Day', "Washington's Birthday", 'Labor Day', 'Columbus Day']
        long_wknd = ['Independence Day', 'Veterans Day']
        long_week = ['Christmas Day', 'Thanksgiving Day']
        
        df_mon_wknd = df.query('Holiday in @mon_wknd')
        df_long_wknd = df.query('Holiday in @long_wknd')
        df_long_week = df.query('Holiday in @long_week')

        mon_wknd_dates = np.hstack(
            [np.array(
                [date - timedelta(days=2), date - timedelta(days=1), date]) for date in df_mon_wknd.date.values]
            ).reshape(1, -1)

        df_tgiv = df_long_week.query('Holiday == "Thanksgiving Day"')
        tgiv_start_dates = df_tgiv.apply(lambda row: row.date - timedelta(days=row.dow), axis=1).values
        tgiv_dates = np.hstack(
            [self.get_date_range_from_week(date.year, date.isocalendar()[1]) for date in tgiv_start_dates]
            ).reshape(1, -1)
        
        xmas_dates = np.hstack(
            [np.array(
                [self.get_date_range_from_week(year, week) for week in (51, 52, 1)]
                    ) for year in df_long_week.year.values]
            ).reshape(1, -1)
        
        long_wknd_early = np.hstack(
            [np.array(
                [date - timedelta(days=(n)) for n in range(date.weekday()+3)]
                ) for date in df_long_wknd.date[df_long_wknd.dow < 2]]
        ).reshape(1, -1)
        
        long_wknd_late = np.hstack(
            [np.array(
                [date + timedelta(days=(n)) for n in range((7-date.weekday()))]
                ) for date in df_long_wknd.date[df_long_wknd.dow > 2]]
        ).reshape(1, -1)
        
        holiday_arr = np.hstack([mon_wknd_dates, tgiv_dates, xmas_dates, long_wknd_early, long_wknd_late]).flatten()
    
        df_holi = pd.DataFrame(holiday_arr) 
        df_holi = df_holi.reset_index()
        df_holi.columns = ['index', 'date']
        df_holi = df_holi.drop('index', axis=1)
        df_holi['date2'] = df_holi.date
        df_holi = df_holi.set_index('date2')
        
        return df_holi
        
if __name__ == '__main__':

    holiday_class = Holidays()
    
    df_holi = holiday_class.holidays()
    
