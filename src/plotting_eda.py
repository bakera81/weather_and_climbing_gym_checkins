import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os
import sys
import psycopg2
from tabulate import tabulate
from datetime import timedelta
sys.path.append('/home/mike/dsi/public-repos/weather_and_climbing_gym_checkins')
from src.paths import data_path, prj_path, img_path
from src.funcs import conditions_dict, resample, parse_datetime
from src.decomp import Decomp
from src.hyp_test import run_hyp_tests
from src.sql_exec import SqlExec

red = '#D63A1E'
blue = '#2437CA'

def prune_ends(df, date_part):
    
    df = parse_datetime(df)
    
    cond_dict = conditions_dict(df)

    first = df[date_part].iloc[0]
    first_year = df.year.iloc[0]
    last = df[date_part].iloc[-1]
    last_year = df.year.iloc[-1]
    rem_cond = (
    (cond_dict[date_part][first] & cond_dict['year'][first_year]) 
    | (cond_dict[date_part][last] & cond_dict['year'][last_year])
    )
    
    return df[~rem_cond]

def overview_plot(fig, axes, df_1_cnt, df_2_cnt, freq='W', df_1_params=None, df_2_params=None, xlabels2=None):
    
    # Resample and plot counts
    freq_str = {'W':'Weekly', 'A':'Annual', 'D':'Daily', 'H':'Hourly'}
        
    df_1_res = resample(df_1_cnt, freq, sum)
        
    ser_1_res = df_1_res.iloc[:, 0]
    # ser_1_res.index = ser_1_res.index.to_timestamp()
    
    df_2_res = resample(df_2_cnt, freq, sum)
        
    ser_2_res = df_2_res.iloc[:, 0]
    # ser_2_res.index = ser_2_res.index.to_timestamp()

    axes[0].plot(ser_1_res, **df_1_params)
    axes[0].plot(ser_2_res, **df_2_params)
        
    axes[0].axvline(e_open, color='k', linestyle='--')
    axes[0].annotate('Englewood Opens', xy=(e_open, 2000), 
                    xytext=(e_open + timedelta(weeks=15), 1850), 
                    arrowprops={'color':'k', 'width':1, 'headwidth':10})
    axes[0].legend()

    axes[1].plot(ser_2_res.index, ser_1_res.loc[ser_2_res.index], **df_1_params)
    axes[1].plot(ser_2_res.index, ser_2_res, **df_2_params)
    
    if xlabels2:
        axes[1].set_xticklabels(xlabels2)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(f'Check-ins')
        ax.set_ylim(bottom=0)
        ax.set_yticks([], [])
        

    fig.suptitle(f'{freq_str[freq]} Check-ins')
    
    plt.tight_layout()

def weather_trend_plot(ax, df_counts_res, df_weather_res, title, weather_col='temp_f'):
    
    color = 'green'
    if 'eng' in title.lower():
        gym_label = 'Englewood'
        color = blue
    if 'gol' in title.lower():
        gym_label = 'Golden'
        color = red
        
    if weather_col == 'temp_f':
        w_label = 'Daily Max Temperature'
        w_ylabel = 'Max Temperature (°F)' 
        w_color = 'k'
    if weather_col == 'precip_in':
        w_label = 'Daily Accumulated Precipitation'
        w_ylabel = 'Inches of Rainfall'
        w_color = 'blue'
    if weather_col == 'snow_depth_in':
        w_label = 'Average Daily Snow Depth'
        w_ylabel = 'Inches of Snow on Ground' 
        w_color = 'orange'
        
    df_weather_res = df_weather_res.loc[df_counts_res.index]
    
    ax.set_title(title)
    
    ax_c = ax.twinx()
    axes = [ax, ax_c]
    
    if isinstance(df_counts_res, pd.DataFrame):
        ser_counts_res = df_counts_res.iloc[:, 0]
    else:
        ser_counts_res = df_counts_res
    
    ax_c.plot(ser_counts_res, color = color, linewidth=3, label=f'{gym_label} Checkins')
    ax_c.set_ylabel('Relative Check-ins/hr')
    ax_c.set_yticklabels([], [])
    ax.plot(df_weather_res[weather_col], linestyle='--', linewidth=2, label=w_label, color=w_color)
    ax.set_ylabel(w_ylabel)
    
    plt.grid(b=None)

def joint_plot(x, y, title):
    
    color = 'green'
    if title.lower().startswith('e'):
        color = blue
    if title.lower().startswith('g'):
        color = red
    
    # Useful variables
    dow = ('Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun')
    hour_str_dict = {k:v for k, v in zip(range(1, 25), [f'{hour} AM' for hour in range(1, 12)] 
                                                    + [f'{hour} PM' for hour in [12, *range(1, 12)] 
                                                    + ['12 AM']])
            }
    
    
    y_lim = list(range(4, 24, 2))
    y_ticklabels = [hour_str_dict[y] for y in y_lim]
    # y_ticklabels = [hour_str_dict[y] for y in y_lim][::-1]
    xlabel = 'Day of Week'
    ylabel = 'Hour of Day'
    x_lim = list(range(0, 7))
    # unique_hours = np.unique(y)
    # hour_reverser = dict(zip(unique_hours, unique_hours[::-1]))
    # y = list(map(lambda y: hour_reverser[y], y))
    
    ax = sns.jointplot(x, y, kind='kde', color=color, ylim=[0, 24])
    ax_j = ax.ax_joint
    ax_j.set_xticklabels(dow)
    ax_j.set_yticks(y_lim)
    ax_j.set_xticks(x_lim)
    ax_j.set_yticklabels(y_ticklabels)
    ax_j.set_xlabel(xlabel)
    ax_j.set_ylabel(ylabel)
    ax_j.invert_yaxis()
    ax_j.grid(False)
    
    ax_m = ax.ax_marg_x
    ax_m.set_title(title)

    
def box_plot(ax, df_parsed, title):
    
    color = 'green'
    if title.lower().startswith('e'):
        color = blue
    if title.lower().startswith('g'):
        color = red
    
    y_lim = np.unique(df_parsed.hour.values)
    hour_str_dict = {k:v for k, v in zip(range(1, 25), [f'{hour} AM' for hour in range(1, 12)] 
                                                    + [f'{hour} PM' for hour in [12, *range(1, 12)] 
                                                    + ['12 AM']])
            }
    y_ticklabels = [hour_str_dict[y] for y in y_lim]
    
    # df_parsed['date'] = pd.to_datetime(df_parsed.date)
    df_hour_date = df_parsed.groupby(['hour', 'date']).count()['guid']
    
    df_hour_list = [df_hour_date.xs(h).values for h in y_lim]
    
    bp = ax.boxplot(df_hour_list)
    ax.set_xticklabels(y_ticklabels)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Hourly Check-ins')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=color)
    ax.set_title(title)

    ax.set_yticks([], [])
      
def bar_plot(ax, x, y, title, xlabel, ylabel):
    
    color = 'green'
    if title.lower().startswith('e'):
        color = blue
    if title.lower().startswith('g'):
        color = red
    
    ax.bar(x, y, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    ax.set_yticks([], [])

def resampling_plot(fig, axes, counts, title):
    
    color = 'green'
    if title.lower().startswith('e'):
        color = blue
    if title.lower().startswith('g'):
        color = red
    
    df_date_parsed = parse_datetime(counts)
    
    freq_str = {'W':'Weekly', 'A':'Annually', 'D':'Daily', 'H':'Hourly'}
    freq_list = ['H', 'D', 'W', 'A']
    freq_funcs = dict(zip(freq_list, ['hour', 'date', 'week', 'year']))
    
    last_ax = axes[-1]
    
    for freq, ax in zip(freq_list, axes):
        
        series = resample(df_date_parsed, freq, np.mean).iloc[:, 0]
        series.index = pd.to_datetime(series.index)
        ax.set_yticks([], [])
        
        ax.plot(series, color=color)
        ax.set_title(f'Resampled {freq_str[freq]}')
        
        if freq == 'H':
            ax.set_ylabel('Check-ins/hr')
        else:
            ax.set_ylabel('Average Check-ins/hr')
            
        last_ax.plot(series, label=freq_str[freq])

    last_ax.set_title('All Resampling Data')
    last_ax.set_ylabel('Average Check-ins/hr')
    
    last_ax.legend()
    fig.suptitle(title, fontsize=25)
    
    plt.tight_layout(pad=3)

def decomp_plot(fig, axes, Decomp, title, xtick_labels=None):
    
    color = 'green'
    if title.lower().startswith('e'):
        color = blue
    if title.lower().startswith('g'):
        color = red
    
    df_decomposed = Decomp.decomp()
        
    axes[0].plot(Decomp.data.iloc[:,0], label='Data', color=color)
    axes[0].plot(df_decomposed.trend_linear, label='Linear Fit')
    axes[0].plot(df_decomposed.trend_annual, label='Annual Average')

    axes[0].legend()
    axes[0].set_title(title)

    # Linear Trend
    axes[1].plot(df_decomposed.trend_linear, label='Linear Fit', color=color)

    axes[1].legend()
    axes[1].set_title('Trend Component LF')

    # Seasonality Linear Fit

    axes[2].plot(df_decomposed.seasonal_linear, label='Seasonal Data LF', color=color)
    axes[2].axhline(0, color='k', linewidth=1, linestyle='--', label='Flat')

    axes[2].legend()
    axes[2].set_title('Seasonal Component LF')

    # Annual Trend
    
    axes[3].plot(df_decomposed.trend_annual, label='Annual Average', color=color)

    axes[3].legend()
    axes[3].set_title('Trend Component Annual')
    
    # Seasonality Annual Sample

    axes[4].plot(df_decomposed.seasonal_annual, label='Seasonal Data Annual', color=color)
    axes[4].axhline(0, color='k', linewidth=1, linestyle='--', label='Flat')

    axes[4].legend()
    axes[4].set_title('Seasonal Component Annual')
    
    for ax in axes:
        ax.set_ylabel('Avg Check-ins/hr')
        ax.set_yticks([], [])


def hypothesis_plot(fig, axes, results, gym_prefix):
    
    if gym_prefix == 'gol':
        fig.suptitle('Golden Hypothesis Tests', fontsize=50)
    if gym_prefix == 'eng':
        fig.suptitle('Englewood Hypothesis Tests', fontsize=50)
    
    groups = ['weekdays', 'weekends']
    weather_conditions = ['temp', 'rain', 'snow']
    weather_strs = [
        ('Extreme Temperature', 'Good Temperatures', 'Temperature'),
        ('Wet Days', 'Dry Days', 'Precipitation'),
        ('Snowy Days', 'Dry Days', 'Snow Depth')]
    weather_label_dict = dict(zip(weather_conditions, weather_strs))
    groups = ['weekdays', 'weekends']
    
    axes[0, 0].set_title('Weekdays', fontsize=35)
    axes[0, 1].set_title('Weekends', fontsize=35)
    
    
    for i, weather_condition in enumerate(weather_conditions):
        for ax, group in zip(axes[i, :], groups):
            ax.set_yticks([], [])
            
            group_name = group
            bad_label, good_label, weather_parameter = weather_label_dict[weather_condition]
            
            if group == 'weekdays':
                group_name = 'Mon - Thur'
                ax.set_ylabel(weather_parameter, fontsize=35)
                
            hyp_class = results[weather_condition][group]
            hyp_class.plot(ax, bad_label, good_label, None, group_name)
    
def single_hyp_plot(ax, hyp_class, title, label=True):
    
    labels = [None] * 4
    if label:
        labels = ['Good Weather Mean', 'Good Weather Distribution', 'Bad Weather Mean', 'Good Weather Distribution']
    
    ax.hist(hyp_class.good.series, bins=30, color='green', alpha=0.5, density=True, label=None)
    ax.axvline(hyp_class.good.mean, linestyle='--', color='green', label = labels[0])
    sns.kdeplot(hyp_class.good.series, ax=ax, legend=False, color='green', label=labels[1])
    
    ax.hist(hyp_class.bad.series, bins=30, color='brown', alpha=0.5, density=True, label=None)
    ax.axvline(hyp_class.bad.mean, linestyle='--', color='brown', label=labels[2])
    sns.kdeplot(hyp_class.bad.series, ax=ax, legend=False, color='brown', label=labels[3])
    
    ax.set_xlabel('Adjusted Daily Checkins')
    ax.set_ylabel('Probability Density')
    ax.set_yticklabels([], [])
    ax.set_xticklabels([], [])
    ax.set_title(title)
    ax.grid(False)

    
if __name__ == '__main__':
    
    # # Gym Selector
    # gyms = ['gol', 'eng']
    
    # # Create SQL executers
    # g_sql = SqlExec('gol')
    # e_sql = SqlExec('eng')
    
    # # Get counts data for resampling
    # g_counts = g_sql.hour_counts()
    # e_counts = e_sql.hour_counts()

    # # Useful Variables
    # e_open = e_counts.index[0] # ET Englewood Open

    # # Plot Params
    # plt.style.use('fivethirtyeight')

    # g_params = {'label':'Golden', 'color':red, 'alpha':0.75}
    # e_params = {'label':'Englewood', 'color':blue, 'alpha':0.75}
    # w_params = {'color':'red', 'linewidth':1, 'alpha':0.75, 'linestyle':'--'}

    # col_params = {'tot_checkins':{'linewidth':2}, 
    #             'mem_checkins':{'linewidth':1},
    #             'nmem_checkins':{'linewidth':1, 'linestyle':'--'}}
    
    # #### Line Plots ####
    # ##---------------------------------------------------------------------------------------------------------------#

    # ## Main resampling plot
    # plt.rcParams['font.size'] = 20

    # # Weekly Resampling
    
    # fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    # xlabels2 = ['Sep 2018', 'Nov 2018', 'Jan 2019', 'Mar 2019', 'May 2019', 'Jul 2019', 'Sep 2019', 'Nov 2019', 'Jan 2020']
    
    # overview_plot(fig, axes, prune_ends(g_counts, 'week'), prune_ends(e_counts, 'week'), xlabels2=xlabels2, df_1_params=g_params, df_2_params=e_params)
    
    # axes[1].set_yticklabels([' ', 2000, 4000, 6000])
    
    # plt.savefig(os.path.join(img_path, 'weekly_overview.png'))
    
    # # Daily Resampling
    
    # fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    
    # overview_plot(fig, axes, prune_ends(g_counts, 'date'), prune_ends(e_counts, 'date'), xlabels2=xlabels2, df_1_params=g_params, df_2_params=e_params, freq='D')
    
    # # axes[1].set_yticklabels([' ', 2000, 4000, 6000])
    
    # plt.savefig(os.path.join(img_path, 'daily_overview.png'))

    
    # ## Weather comparison plot
    
    # # Get weather data
    # w_df = g_sql.weather()
    
    # g_counts_wkly = resample(g_counts, 'W', np.mean)
    # g_decomp = Decomp(g_counts_wkly)
    # g_counts_wkly_seasonal = g_decomp.decomp()['seasonal_annual']
    # # e_counts_wkly = resample(e_counts, 'W', np.mean)
    # # e_counts_wkly_seasonal = Decomp(e_counts_wkly).decomp()[['seasonal_annual', 'trend_annual']]
    
    # w_df_wkly = resample(w_df, 'W', {'temp_f':np.max, 'precip_in':np.sum, 'snow_depth_in':np.mean})
    # # w_df_wkly = w_df.resample('W').agg({'temp_f':np.max, 'precip_in':np.sum, 'snow_depth_in':np.mean})
    
    # fig, axes = plt.subplots(3, 1, figsize=(25, 15))
    # axes = axes.flatten()
    
    # weather_trend_plot(axes[0], g_counts_wkly_seasonal, w_df_wkly, 'Weekly Golden Check-ins vs Max Temp')
    
    # weather_trend_plot(axes[1], g_counts_wkly_seasonal, w_df_wkly, 'Weekly Golden Check-ins vs Precipitation', weather_col='precip_in')
    
    # weather_trend_plot(axes[2], g_counts_wkly_seasonal, w_df_wkly, 'Weekly Golden Check-ins vs Snow Depth', weather_col='snow_depth_in')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(img_path, 'weather_compare.png'))
    
    # #### Bar Plots ####
    # ##---------------------------------------------------------------------------------------------------------------#
    
    # ## Seasonal bar plot
    
    # plt.rcParams['font.size'] = 12
    
    # # Create filter object
    # g_filter = conditions_dict(parse_datetime(g_counts), dt_col='parsed')
    # e_filter = conditions_dict(parse_datetime(e_counts), dt_col='parsed')
    # seasons = range(0, 4)
    # season_strs = dict(zip(seasons, ['Spring', 'Summer', 'Fall', 'Winter']))
    
    # # Aggregate check-ins seasonally
    # g_season_cnts = []
    # e_season_cnts = []

    # for s in seasons:
    #     g_season_cnts.append(parse_datetime(g_counts)[g_filter['season'][s]].tot_checkins.mean())
    #     e_season_cnts.append(parse_datetime(e_counts)[e_filter['season'][s]].tot_checkins.mean())
    
    # x = season_strs.values()
    # g_y = g_season_cnts
    # e_y = e_season_cnts
    
    # # Plot
    # fig, ax = plt.subplots(figsize=(5, 5))
    
    # bar_plot(ax, x, g_y, 'Golden Check-ins', 'Season', 'Average check-ins/hr')
    
    # plt.savefig(os.path.join(img_path, 'gol_check-ins_by_season.png'))

    
    # fig, ax = plt.subplots(figsize=(5, 5))

    
    # bar_plot(ax, x, e_y, 'Englewood Check-ins', 'Season', 'Average check-ins/hr')

    # plt.savefig(os.path.join(img_path, 'eng_check-ins_by_season.png'))
    
    # ## Monthly bar plot

    # # Create filter object
    # g_filter = conditions_dict(g_counts)
    # e_filter = conditions_dict(e_counts)
    # months = range(1, 13)
    # month_strs = dict(zip(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']))
    
    # # Aggregate check-ins seasonally
    # g_month_cnts = []
    # e_month_cnts = []

    # for m in months:
    #     g_month_cnts.append(g_counts[g_filter['month'][m]].tot_checkins.mean())
    #     e_month_cnts.append(e_counts[e_filter['month'][m]].tot_checkins.mean())
    
    # x = month_strs.values()
    # g_y = g_month_cnts
    # e_y = e_month_cnts
    
    # # Plot
    # fig, ax = plt.subplots(figsize=(10, 7))


    # bar_plot(ax, x, g_y, 'Golden Check-ins', 'Month', 'Average check-ins/hr')
    # plt.savefig(os.path.join(img_path, 'gol_check-ins_by_month.png'))

    
    # fig, ax = plt.subplots(figsize=(10, 7))

    
    # bar_plot(ax, x, e_y, 'Englewood Check-ins', 'Month', 'Average check-ins/hr')
    # plt.savefig(os.path.join(img_path, 'eng_check-ins_by_month.png'))
    
    # ### Joint Plots ####
    # ##---------------------------------------------------------------------------------------------------------------#

    # # Hours & Day of Week joint KDE plot
    
    # # Get raw check-ins data for joint plot
    # g_checks = parse_datetime(g_sql.checkins(), col='datetime')
    # e_checks = parse_datetime(e_sql.checkins(), col='datetime')
    
    # # Golden
    # x = g_checks.dow.values
    # y = g_checks.hour.values
    
    # sns.set(rc={'axes.labelsize':20,
    #         'figure.figsize':(10, 10),
    #         'xtick.labelsize':14,
    #         'ytick.labelsize':14,
    #         'axes.titlesize':24})

    # joint_plot(x, y, 'Golden Checkins')
    # plt.tight_layout()
    # plt.savefig(os.path.join(img_path, 'gol_joint_kde.png'))
    
    # # Englewood
    # x = e_checks.dow.values
    # y = e_checks.hour.values
    
    # joint_plot(x, y, 'Englewood Checkins')
    # fig = plt.gcf()
    # plt.tight_layout()
    # plt.savefig(os.path.join(img_path, 'eng_joint_kde.png'))

    # ## Box Plots
    
    # fig, axes = plt.subplots(3, 1, figsize=(8, 16))
    # # fig, ax = plt.subplots(figsize=(8, 8))

    
    # is_mon_thur = ((e_checks.dow == 0) | (e_checks.dow == 1) | (e_checks.dow == 2) | (e_checks.dow == 3))
    # is_friday = (e_checks.dow == 4)
    # is_weekend = ((e_checks.dow == 5) | (e_checks.dow == 6))
    
    # # box_plot(ax, e_checks, 'Englewood All')
    # box_plot(axes[0], e_checks[is_mon_thur], 'Englewood Mon-Thur')
    # box_plot(axes[1], e_checks[is_friday], 'Englewood Friday')
    # box_plot(axes[2], e_checks[is_weekend], 'Englewood Weekends')
    # plt.tight_layout()
    
    # fig, axes = plt.subplots(3, 1, figsize=(8, 16))
    # # fig, ax = plt.subplots(figsize=(8, 8))

    
    # is_mon_thur = ((g_checks.dow == 0) | (g_checks.dow == 1) | (g_checks.dow == 2) | (g_checks.dow == 3))
    # is_friday = (g_checks.dow == 4)
    # is_weekend = ((g_checks.dow == 5) | (g_checks.dow == 6))
    
    # # box_plot(ax, g_checks, 'Englewood All')
    # box_plot(axes[0], g_checks[is_mon_thur], 'Golden Mon-Thur')
    # box_plot(axes[1], g_checks[is_friday], 'Golden Friday')
    # box_plot(axes[2], g_checks[is_weekend], 'Golden Weekends')
    # plt.tight_layout()

    # ## Decomposition Plots ####
    # #---------------------------------------------------------------------------------------------------------------#

    # ## Resampling options plot
    # plt.rcParams['font.size'] = 30

    
    # fig, axes = plt.subplots(figsize=(20, 20))
    
    # axes = []
    
    # axes.append(plt.subplot2grid((3, 2), (0, 0)))
    # axes.append(plt.subplot2grid((3, 2), (0, 1)))
    # axes.append(plt.subplot2grid((3, 2), (1, 0)))
    # axes.append(plt.subplot2grid((3, 2), (1, 1)))
    # axes.append(plt.subplot2grid((3, 2), (2, 0), colspan=2))
    
    # resampling_plot(fig, axes, g_counts, 'Golden Check-ins')
    
    # plt.savefig(os.path.join(img_path, 'gol_resampling_plot.png'))
    
    # fig, axes = plt.subplots(figsize=(20, 20))
    
    # axes = []
    
    # axes.append(plt.subplot2grid((3, 2), (0, 0)))
    # axes.append(plt.subplot2grid((3, 2), (0, 1)))
    # axes.append(plt.subplot2grid((3, 2), (1, 0)))
    # axes.append(plt.subplot2grid((3, 2), (1, 1)))
    # axes.append(plt.subplot2grid((3, 2), (2, 0), colspan=2))
    
    # resampling_plot(fig, axes, e_counts, 'Englewood Check-ins')
    
    # plt.savefig(os.path.join(img_path, 'eng_resampling_plot.png'))
    
    # ## Decomposition plots
    
    # g_decomp_days = Decomp(resample(g_counts, 'D', np.sum))
    # e_decomp_days = Decomp(resample(e_counts, 'D', np.sum))
    
    # fig, axes = plt.subplots(5, 1, figsize=(20, 30), sharey=True)
    # axes = axes.flatten()
    
    # decomp_plot(fig, axes, g_decomp_days, 'Golden Decomposition')
    
    # plt.tight_layout()
    # plt.savefig(os.path.join(img_path, 'gol_decomp_plot.png'))
    
    # fig, axes = plt.subplots(5, 1, figsize=(20, 30), sharey=True)
    # axes = axes.flatten()
    
    # xtick_labels = ['Sep 2018', 'Nov 2018', 'Jan 2019', 'Mar 2019', 'May 2019', 'Jul 2019', 'Sep 2019', 'Nov 2019', 'Jan 2020']

    # decomp_plot(fig, axes, e_decomp_days, 'Englewood Decomposition')
    
    # for i, ax in enumerate(axes):
    #     if i < 3:
    #       for tick in ax.get_xticklabels():
    #             tick.set_rotation(45)
    #     else:
    #         for tick in ax.get_xticklabels():
    #             tick.set_rotation(45)

    # plt.tight_layout()
    # plt.savefig(os.path.join(img_path, 'eng_decomp_plot.png'))
    
    ### Hypothesis Test Plots ####
    ##---------------------------------------------------------------------------------------------------------------#
    plt.rcParams['font.size'] = 14
    
    g_results = run_hyp_tests('gol')
    e_results = run_hyp_tests('eng')
    
    ## Hypothesis plots grid
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 20), constrained_layout=True)
    
    hypothesis_plot(fig, axes, g_results, 'gol')
    
    for ax in axes[:, 0]:
        ax.legend(loc='center left')

    plt.savefig(os.path.join(img_path, 'gol_hyp_plots.png'))

    
    fig, axes = plt.subplots(3, 2, figsize=(20, 20), constrained_layout=True)
    
    hypothesis_plot(fig, axes, e_results, 'eng')
    
    for ax in axes[:, 0]:
        ax.legend(loc='center left')
    
    plt.savefig(os.path.join(img_path, 'eng_hyp_plots.png'))
    
    ## Hypothesis plots sample
    dpi_scale = 0.25
    
    g_results = run_hyp_tests('gol')
    e_results = run_hyp_tests('eng')
    
    # Hypothesis plots grid
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    single_hyp_plot(axes[0], g_results['temp']['weekends'], 'Golden Adjusted Checkin \nDistributions on Weekends \nGood & Bad Temperature Days', label=False)
    single_hyp_plot(axes[1], g_results['temp']['weekdays'], 'Golden Adjusted Checkin \nDistributions on Weekdays \nGood & Bad Temperature Days')
    custom_lines = [Line2D([0], [0], color='green', lw=2),
                Line2D([0], [0], color='brown', lw=2)]
    
    axes[0].legend(loc='upper right', handles=custom_lines, labels=['Good Weather Distribution', 'Bad Weather Distribution'])

    for ax in axes:
        ax.set_yticklabels([], [])
        ax.set_xticklabels([], [])


    plt.tight_layout()    
    plt.savefig(os.path.join(img_path, 'gol_hyp_2_plots.png'))

    g_hyp_df = pd.DataFrame(g_results)
    e_hyp_df = pd.DataFrame(e_results)
    
    g_res_strs = g_hyp_df.applymap(
        lambda hyp: f'{hyp.mean_diff:.0f} more checkins for bad weather with a p value: {hyp.p:.3f}'
        ).transpose()
    
    e_res_strs = e_hyp_df.applymap(
        lambda hyp: f'{hyp.mean_diff:.0f} more checkins for bad weather with a p value: {hyp.p:.3f}'
        ).transpose()

    print(tabulate(g_res_strs, tablefmt="pipe", headers="keys"))
    print(tabulate(e_res_strs, tablefmt="pipe", headers="keys"))
    