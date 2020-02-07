import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import psycopg2
from psycopg2.extensions import AsIs
from datetime import timedelta
sys.path.append('/home/mike/dsi/capstones/climbing_gym_checkins_eda')
    
def parse_datetime(df, col='index', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True):
        
        if col == 'index':
            datetime_col = df.reset_index().iloc[:, 0]
        else:
            datetime_col = df.reset_index()[col]
        
        datetime_col.index = df.index
            
        parsed_cols = []
        parsed_col_names = []
        
        if hour:
            parsed_cols.append(datetime_col.dt.hour)
            parsed_col_names.append('hour')
            
        if dow:
            parsed_cols.append(datetime_col.dt.dayofweek)
            parsed_col_names.append('dow')
            
        if date:
            parsed_cols.append(datetime_col.dt.date)
            parsed_col_names.append('date')
            
        if date:
            parsed_cols.append(datetime_col.dt.week)
            parsed_col_names.append('week')
        
        if month:
            parsed_cols.append(datetime_col.dt.month)
            parsed_col_names.append('month')
            
        if season:
            seasons_col = datetime_col.apply(find_season)
            parsed_cols.append(seasons_col)
            parsed_col_names.append('season')
            
        if year:
            parsed_cols.append(datetime_col.dt.year)
            parsed_col_names.append('year')
            
        df_parsed = pd.concat(parsed_cols, axis=1)
        
        df_parsed.columns = parsed_col_names
        
        df_merged = pd.concat([df, df_parsed], axis=1)
        
        df_merged.index = datetime_col
        
        return df_merged
    
def conditions_dict(df, dt_col='index', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True):
    
    if dt_col == 'index':
        datetime_col = df.reset_index().iloc[:, 0].dt
        season = False
    elif dt_col == 'parsed':
        datetime_col = df
    else:
        datetime_col = pd.to_datetime(df[dt_col]).dt
    
    cond_dict = {}
    
    if hour:
        
        hour_cond = {}
        hours = range(0, 24)

        for hour in hours:
            hour_cond[hour] = (datetime_col.hour == hour).values

        cond_dict.update({'hour':hour_cond})
    
    if dow:

        dow_cond = {}
        dow = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        
        for i, day in enumerate(dow):
            if dt_col == 'parsed':
                dow_cond[day] = (datetime_col.dow == i).values
            # elif dt_col == 'index':
            else:
                dow_cond[day] = (datetime_col.dayofweek == i).values

        cond_dict.update({'dow':dow_cond})
    
    if date:
    
        date_cond = {} 
        dates = np.unique(datetime_col.date)
        
        for date in dates:
            date_cond[date] = (datetime_col.date == date).values

        cond_dict.update({'date':date_cond})

    if week:
        
        week_cond = {}
        weeks = range(53)
        
        for week in weeks:
            week_cond[week] = (datetime_col.week == week).values

        cond_dict.update({'week':week_cond})
    
    if month:
        
        month_cond = {}
        months = range(1, 13)
        
        for month in months:
            month_cond[month] = (datetime_col.month == month).values

        cond_dict.update({'month':month_cond})
        
    if season:
        
        season_cond = {}
        seasons = range(0, 4)
        
        for season in seasons:
            seasons_col=datetime_col.season
                
        for season in seasons:
            season_cond[season] = (seasons_col == season).values

        cond_dict.update({'season':season_cond})   
        
    if year:
        
        year_cond = {}
        years = np.unique(datetime_col.year)
        
        for year in years:
            year_cond[year] = (datetime_col.year == year).values

        cond_dict.update({'year':year_cond})

    return cond_dict

def resample(df, freq, func, as_period=False):
    
    gb_df_res = df.resample(freq)
    
    if func == 'interpoloate':
        df_res = gb_df_res.interpolate().dropna()
    if isinstance(func, dict):
        df_res = gb_df_res.agg(func).dropna()
    else:
        df_res = gb_df_res.apply(func).dropna()
    
    if isinstance(df_res, pd.DataFrame):
        if 'tot_checkins' in df_res.columns:
            
            tot_counts_zero = df_res.iloc[:,0] == 0
            df_res = df_res[~tot_counts_zero]
        
    if as_period:
        
        df_res.index = df_res.index.to_period(freq)

    return df_res

def find_season(ts):
                
        m = ts.month
        season_sets = {
                0:{3, 4, 5},    #Spring
                1:{6, 7, 8},    #Summer
                2:{9, 10, 11},  #Fall
                3:{12, 1, 2}}   #Winter
                
        for season, season_set in season_sets.items():
            if m in season_set:
                return season
            
def get_freq(df):
    
    time_diff = pd.to_timedelta(df.index[1] - df.index[0])
    
    if time_diff < timedelta(hours = 2):
        return 'H'
    
    if time_diff < timedelta(hours=30):
        return 'D'
    
    if time_diff < timedelta(days=8):
        return 'W'
    
    if time_diff < timedelta(days=35):
        return 'M'
    
    if time_diff < timedelta(days=100):
        return 'Q'
    
    if time_diff < timedelta(weeks=55):
        return 'Y'
    
    raise ValueError('Could not determine frequency')

if __name__ == '__main__':
    
    # Test funcs
    gol_sql = SqlExec('eng')
    df_ch1 = gol_sql.checkins()
    df_us1 = gol_sql.users()
    df_co1 = gol_sql.hour_counts()
    df_w1 = gol_sql.weather()
    
    print('executed')
    
    df_ch2 = gol_sql.checkins()
    df_us2 = gol_sql.users()
    df_co2 = gol_sql.hour_counts()
    df_w2 = gol_sql.weather()
    
    print('recall')
    
    print('checkins parsed')
    print(df_ch2.iloc[0])
    print('checkins not parsed')
    print(gol_sql.checkins(update=True, parse_datetime=False).iloc[0])
    print('users')
    print(df_us2.iloc[0])
    print('counts')
    print(df_co2.iloc[0])
    print('weather')
    print(df_w2.iloc[0])
    
    
    print(conditions_dict(df_ch1)['dow']['thu'][:10])
    print(df_ch1.dow.iloc[:10])
    