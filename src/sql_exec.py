import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import psycopg2
from psycopg2.extensions import AsIs
from datetime import timedelta
sys.path.append('/home/mike/dsi/capstones/climbing_gym_checkins_eda')
from src.paths import data_path, prj_path, img_path
from src.funcs import conditions_dict, resample, parse_datetime
from src.decomp import Decomp
from src.holidays import Holidays

class SqlExec:
    
    def __init__(self, gym_prefix):
        
        self.gym_prefix = gym_prefix
        self.__upass = os.environ['PASS']
        self.df_dict = {}

    def _psql_to_df(self, conn, sql_query='''SELECT * FROM %s''', table=None, index=None, convert_dt=None):
        '''
        Inputs: 
            sql_query(str): PostgreSQL query to be executed
            conn: psycopg2 connection to database
            index(str): column of table that should be used as the DataFrame index
            convert_dt(array_like): list of column names to be converted to datetime with MST timezone
        Returns:
            df(DataFrame): pandas DataFrame from sql_query
        '''
        cur = conn.cursor()
        
        if table:
            cur.execute(sql_query, (AsIs(table),))
        else:
            cur.execute(sql_query)
            
        df = pd.DataFrame(
            cur.fetchall(), 
            columns=[desc[0] for desc in cur.description])
        
        if index:
            df = df.set_index(index).sort_index()
        
        if convert_dt:
            for col in convert_dt:
                if col == 'index':
                    df.index = pd.to_datetime(df.index, utc=True).tz_convert('MST')
                else:
                    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert('MST')
        
        return df[~df.index.duplicated()]
    
    def _parse_datetime(self, df, col='datetime', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True):
        
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
            
        if week:
            parsed_cols.append(datetime_col.dt.week)
            parsed_col_names.append('week')
        
        if month:
            parsed_cols.append(datetime_col.dt.month)
            parsed_col_names.append('month')
            
        if season:
            seasons_col = datetime_col.apply(self._find_season)
            parsed_cols.append(seasons_col)
            parsed_col_names.append('season')
            
        if year:
            parsed_cols.append(datetime_col.dt.year)
            parsed_col_names.append('year')
            
        df_parsed = pd.concat(parsed_cols, axis=1)
        
        df_parsed.columns = parsed_col_names
        
        df_merged = pd.concat([df, df_parsed], axis=1)
        
        return df_merged

    def holidays(self, start_year = 2013, end_year = 2021):
        
        holiday_class = Holidays(start_year=start_year, end_year=end_year)
        
        return holiday_class.holidays()
        
    def _find_season(self, ts):
                
        m = ts.month
        season_sets = {
                0:{3, 4, 5},    #Spring
                1:{6, 7, 8},    #Summer
                2:{9, 10, 11},  #Fall
                3:{12, 1, 2}}   #Winter
                
        for season, season_set in season_sets.items():
            if m in season_set:
                return season
        
        raise ValueError('Month not between 1 and 12')

    def checkins(self, update=False, parse_datetime=False, col='datetime', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True):
        
        table = self.gym_prefix + '_checkins'
        
        if not update:
            if table in self.df_dict:
                return self.df_dict[table]
        
        conn = psycopg2.connect(database="climbing_gym", user="postgres", password=self.__upass, host="localhost", port="5432")

        checkin_params = {'index':'id', 'convert_dt':['datetime']}

        # Golden Checkin Data
        df = self._psql_to_df(conn=conn, table=table, **checkin_params)
        
        conn.close()
        
        if parse_datetime:
            df = self._parse_datetime(df, col='datetime', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True)
        
        self.df_dict[table] = df
        
        return df.copy()
    
    def users(self, update=False):
        
        table = self.gym_prefix + '_users'
        
        if not update:
            if table in self.df_dict:
                return self.df_dict[table]
        
        conn = psycopg2.connect(database="climbing_gym", user="postgres", password=self.__upass, host="localhost", port="5432")

        user_params = {'index':'guid', 'convert_dt':['first_checkin', 'last_checkin']}
        
        df = self._psql_to_df(conn=conn, table=table, **user_params)
        
        conn.close()

        self.df_dict[table] = df
        
        return df.copy()

    def hour_counts(self, update=False, parse_datetime=False, col='datetime', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True):
        
        table = self.gym_prefix + '_counts'
        
        if not update:
            if table in self.df_dict:
                return self.df_dict[table]
        
        conn = psycopg2.connect(database="climbing_gym", user="postgres", password=self.__upass, host="localhost", port="5432")

        table = self.gym_prefix + '_counts'

        df = self._psql_to_df(conn=conn, table=table, index='datetime', convert_dt=['index'])
        
        conn.close()
        
        if parse_datetime:
            df = self._parse_datetime(df, col='index', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True)
            
        self.df_dict[table] = df
        
        return df.copy()

    def day_counts(self, update=False, holiday=True, parse_datetime=False, col='datetime', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True):
        
        counts = self.hour_counts(update=False, parse_datetime=False, col='datetime', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True)
        
        day_counts = self._parse_datetime(resample(counts, 'D', np.sum))
        
        df_holi = self.holidays()
        
        df_day_counts = pd.merge(day_counts, df_holi, how='left', left_on='date', right_index=True, suffixes=('_counts', '_holiday'))

        df_day_counts = df_day_counts[~df_day_counts.index.duplicated()]
        
        df_day_counts['is_holiday'] = df_day_counts.date_holiday.notna()
        
        df_day_counts = df_day_counts.drop(['date_holiday', 'date_counts'], axis=1).set_index('date')
        
        df_day_counts.index = pd.to_datetime(df_day_counts.index)
        df_day_counts = df_day_counts.to_period('D')
        
        return df_day_counts
    
    def weather(self, update=False, parse_datetime=False, col='datetime', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True, select_all=False):
        
        table = 'weather'
        
        if table in self.df_dict and not update:
            df =  self.df_dict[table]
        else:
            conn = psycopg2.connect(database="climbing_gym", user="postgres", password=self.__upass, host="localhost", port="5432")
            
            if select_all:
                df = self._psql_to_df(sql_query='''
                SELECT 
                    *
                FROM 
                    weather
                ''', conn=conn, index='datetime', convert_dt=['index'])
            else:
                df = self._psql_to_df(sql_query='''
                    SELECT 
                        datetime,
                        temp_f,
                        cloud_cover,
                        wind_lo_mph,
                        precip_in,
                        snow_depth_in
                    FROM 
                        weather
                    ''', conn=conn, index='datetime', convert_dt=['index'])
            
            conn.close()
                    
        if parse_datetime:
            df = self._parse_datetime(df, col='index', hour=True, dow=True, date=True, week=True, month=True, season=True, year=True)

        self.df_dict[table] = df

        return df.copy()
