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
from src.sql_exec import SqlExec
from src.model import DesignMatrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import sparse

class Grouper():
    
    def __init__(self):
        
        self.user_freq_dates = None
        self.user_freq_dow = None
        
    def fit_user_freq(self, df_checks):
        
        self.df_checks = df_checks
        
        dates = df_checks.datetime.dt.date
        guids = df_checks.guid
        dows = df_checks.datetime.dt.dayofweek
        
        user_freq = pd.concat([dates, guids, dows], axis=1)
        user_freq.columns = ['date', 'guid', 'dow']
        
        self.user_freq_dates = user_freq.pivot_table(index='guid', columns='date', aggfunc='size', fill_value=0)
        
        self.user_freq_dow = user_freq.pivot_table(index='guid', columns='dow', aggfunc='size', fill_value=0)
    
        return self.user_freq_dates, self.user_freq_dow
    
    def scale_data(self, X):
        
        ss = StandardScaler()
        X_centered = ss.fit_transform(X)
        
        return X_centered
    
    def pca_fit_transform(self, pca, X):
        
        X_pca = pca.fit_transform(X)   
        self.pca = pca
        return X_pca
    
    def k_means(self, kmeans, X):
        
        kmeans.fit(X)
        
    def join_user(self, df):
        
        last_checkins = self.df_checks.groupby('guid').max('datetime').reset_index()
        
        pd.merge(last_checkins, df, left_on=['guid', 'datetime'], right_on=['guid', 'datetime'], )
        
    def scree_plot(self, ax, pca=None, n_components_to_plot=8, title=None):
        """Make a scree plot showing the variance explained (i.e. varaince of the projections) for the principal components in a fit sklearn PCA object.
    
        Parameters
        ----------
        ax: matplotlib.axis object
        The axis to make the scree plot on.
        
        pca: sklearn.decomposition.PCA object.
        A fit PCA object.
        
        n_components_to_plot: int
        The number of principal components to display in the skree plot.
        
        title: str
        A title for the skree plot.
        """
        if not pca:
            pca = self.pca
        
        num_components = pca.n_components_
        ind = np.arange(num_components)
        vals = np.cumsum(pca.explained_variance_ratio_*100)
        ax.plot(ind, vals, color='blue')
        ax.scatter(ind, vals, color='blue', s=50)

        for i in range(num_components):
            ax.annotate(r"{:2.2f}%".format(vals[i]), 
                    (ind[i]+0.2, vals[i]+0.005), 
                    va="bottom", 
                    ha="center", 
                    fontsize=12)

        ax.set_xticklabels(ind, fontsize=12)
        ax.set_ylim(0, max(vals) + 0.05)
        ax.set_xlim(0 - 0.45, n_components_to_plot + 0.45)
        ax.set_xlabel("Principal Component", fontsize=12)
        ax.set_ylabel("Variance Explained (%)", fontsize=12)
        if title is not None:
            ax.set_title(title, fontsize=16)

if __name__ == '__main__':
    
    sql = SqlExec('gol')
    
    des_mat = DesignMatrix(sql)
    
    train_dates, test_dates, hold_dates = des_mat.get_date_splits()
    
    df_checks = sql.checkins()
    
    df_checks_train = df_checks[df_checks.datetime.dt.date < max(train_dates).to_timestamp()]
    
    gol_grouper = Grouper()
    
    gol_grouper.fit_user_freq(df_checks_train)
    
    X_centered = gol_grouper.scale_data(gol_grouper.user_freq_dates)
    
    pca = PCA(n_components=2, svd_solver='full')
    
    X_pca = gol_grouper.pca_fit_transform(pca, X_centered)
    pca = gol_grouper.pca
    
    pca_train_path = os.path.join(data_path, 'model/X_train_pca.npy')
    np.save(pca_train_path, X_pca)
    
    fig, ax = plt.subplots()
    
    gol_grouper.scree_plot(ax, pca, 2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(prj_path, 'images/train_scree_plot.png'))
    
    
    