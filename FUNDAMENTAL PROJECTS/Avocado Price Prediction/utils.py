import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import column 

#===================================================================================
# LOAD DATASET
#===================================================================================
def load_data(path= 'data\\avocado.csv'):
    df = pd.read_csv(path)
    
    return df


#===================================================================================
# Data Cleaning and Feature Engineer 
#===================================================================================
def drop_index(df):
    df = df.drop('Unnamed: 0', axis=1)
    return df

def parse_data(df):
    df_clean = df.copy()
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    df_clean['Month'] = df_clean['Date'].dt.month
    df_clean['Week'] = df_clean['Date'].dt.isocalendar().week.astype(int)
    
    return df_clean

def rename_plu(df):
    df_clean = df.copy()
    
    df_clean = df_clean.rename(columns = {
        "Total Volume": "total_volume",
        "4046": "small_hass",
        "4225": "large_hass",
        "4770": "xl_hass",
        "Total Bags": "total_bags",
        "Small Bags": "small_bags",
        "Large Bags": "large_bags",
        "XLarge Bags": "xl_bags",
        "AveragePrice": "AveragePrice",
        "Date": "Date",
        "type": "type",
        "year": "year",
        "region": "region",
    })
    
    return df_clean

#===================================================================================
# Feature Engineering
#===================================================================================

def create_feature(df):
    df_feat = df.copy()
    
    df_feat['log_total_volume'] = np.log1p(df_feat['total_volume'])
    df_feat['log_total_bags'] = np.log1p(df_feat['total_bags'])
    df_feat['log_small_bags'] = np.log1p(df_feat['small_bags'])
    df_feat['log_large_bags'] = np.log1p(df_feat['large_bags'])
    df_feat['log_xl_bags'] = np.log1p(df_feat['xl_bags'])



    t_bags = df_feat[['small_hass','large_hass','xl_hass']].sum(axis=1).replace(0, np.nan)
    df_feat['small_share'] = df_feat['small_hass'] / t_bags
    df_feat['large_share'] = df_feat['large_hass'] / t_bags
    df_feat['xl_share'] = df_feat['xl_hass'] / t_bags

    return df_feat
