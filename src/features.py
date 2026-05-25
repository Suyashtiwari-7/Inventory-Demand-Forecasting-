import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def add_lag_features(df):
    df = df.copy()
    
    # Sort values chronologically per store/sku combo
    if 'week_dt' not in df.columns and 'week' in df.columns:
        df['week_dt'] = pd.to_datetime(df['week'], format='%d/%m/%y')
    
    # Ensure correct sorting
    original_idx = df.index
    if 'week_dt' in df.columns:
        df = df.sort_values(['store_id', 'sku_id', 'week_dt'])
    
    # Group and shift
    group = df.groupby(['store_id', 'sku_id'], observed=False)
    df['total_price_lag_1'] = group['total_price'].shift(1)
    df['total_price_lag_2'] = group['total_price'].shift(2)
    df['is_featured_sku_lag_1'] = group['is_featured_sku'].shift(1)
    df['is_display_sku_lag_1'] = group['is_display_sku'].shift(1)
    
    # Lag of discount ratio
    price_diff = df['base_price'] - df['total_price']
    discount_ratio = price_diff / (df['base_price'] + 1e-5)
    df['discount_ratio_lag_1'] = discount_ratio.groupby([df['store_id'], df['sku_id']], observed=False).shift(1)
    
    # Impute missing values from shifting (leading weeks per group)
    df['total_price_lag_1'] = df['total_price_lag_1'].fillna(df['total_price'])
    df['total_price_lag_2'] = df['total_price_lag_2'].fillna(df['total_price'])
    df['is_featured_sku_lag_1'] = df['is_featured_sku_lag_1'].fillna(df['is_featured_sku'])
    df['is_display_sku_lag_1'] = df['is_display_sku_lag_1'].fillna(df['is_display_sku'])
    df['discount_ratio_lag_1'] = df['discount_ratio_lag_1'].fillna(discount_ratio)
    
    # Restore original index ordering to prevent any misalignment
    df = df.loc[original_idx]
    
    return df

def preprocess_data(df):
    """
    Applies basic preprocessing, date extraction, lag features, and discount feature engineering.
    """
    df = df.copy()
    
    # Impute missing total_price with base_price
    if 'total_price' in df.columns:
        df['total_price'] = df['total_price'].fillna(df['base_price'])
        
    # Convert week to datetime
    if 'week' in df.columns:
        df['week_dt'] = pd.to_datetime(df['week'], format='%d/%m/%y')
        df['day'] = df['week_dt'].dt.day
        df['month'] = df['week_dt'].dt.month
        df['year'] = df['week_dt'].dt.year
        df['week_of_year'] = df['week_dt'].dt.isocalendar().week.astype(int)
        df['day_of_year'] = df['week_dt'].dt.dayofyear
    
    # Feature engineering for discounts
    if 'total_price' in df.columns and 'base_price' in df.columns:
        df['price_difference'] = df['base_price'] - df['total_price']
        df['discount_ratio'] = df['price_difference'] / (df['base_price'] + 1e-5)
        df['is_discounted'] = (df['total_price'] < df['base_price']).astype(int)
        
    # Add lag features
    if 'total_price' in df.columns and 'base_price' in df.columns:
        df = add_lag_features(df)
        
    # Cast identifiers as categorical dtypes for LightGBM
    if 'store_id' in df.columns:
        df['store_id'] = df['store_id'].astype('category')
    if 'sku_id' in df.columns:
        df['sku_id'] = df['sku_id'].astype('category')
        
    return df

def compute_target_encodings(train_df):
    """
    Computes out-of-fold target encoding for store-sku, store, and sku combinations
    to prevent target leakage during training.
    """
    train_df = train_df.copy()
    train_df = train_df.reset_index(drop=True)
    
    # Initialize OOF encoding columns
    train_df['store_sku_mean'] = 0.0
    train_df['store_mean'] = 0.0
    train_df['sku_mean'] = 0.0
    
    global_mean = train_df['units_sold'].mean()
    
    # K-Fold Out-of-fold calculation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_df):
        tr = train_df.iloc[train_idx].copy()
        val = train_df.iloc[val_idx].copy()
        
        # Compute means
        store_sku_oof = tr.groupby(['store_id', 'sku_id'], observed=False)['units_sold'].mean()
        store_oof = tr.groupby('store_id', observed=False)['units_sold'].mean()
        sku_oof = tr.groupby('sku_id', observed=False)['units_sold'].mean()
        
        # Rename series to prevent name collisions during joining
        store_sku_oof.name = 'oof_store_sku'
        store_oof.name = 'oof_store'
        sku_oof.name = 'oof_sku'
        
        # Map to validation fold
        val_store_sku = val.join(store_sku_oof, on=['store_id', 'sku_id'])['oof_store_sku']
        train_df.loc[val_idx, 'store_sku_mean'] = val_store_sku.values
        
        val_store = val.join(store_oof, on='store_id')['oof_store']
        train_df.loc[val_idx, 'store_mean'] = val_store.values
        
        val_sku = val.join(sku_oof, on='sku_id')['oof_sku']
        train_df.loc[val_idx, 'sku_mean'] = val_sku.values
        
    # Fill any NaNs in OOF columns
    train_df['sku_mean'] = train_df['sku_mean'].fillna(global_mean)
    train_df['store_mean'] = train_df['store_mean'].fillna(global_mean)
    train_df['store_sku_mean'] = train_df['store_sku_mean'].fillna(train_df['sku_mean']).fillna(global_mean)
    
    # Compute full mapping tables for inference
    store_sku_map = train_df.groupby(['store_id', 'sku_id'], observed=False)['units_sold'].mean().reset_index()
    store_sku_map.rename(columns={'units_sold': 'store_sku_mean_sales'}, inplace=True)
    
    store_map = train_df.groupby('store_id', observed=False)['units_sold'].mean().reset_index()
    store_map.rename(columns={'units_sold': 'store_mean_sales'}, inplace=True)
    
    sku_map = train_df.groupby('sku_id', observed=False)['units_sold'].mean().reset_index()
    sku_map.rename(columns={'units_sold': 'sku_mean_sales'}, inplace=True)
    
    # Store category lists for inference alignment
    store_categories = train_df['store_id'].cat.categories.tolist()
    sku_categories = train_df['sku_id'].cat.categories.tolist()
    
    mappings = {
        'global_mean': global_mean,
        'store_sku_map': store_sku_map,
        'store_map': store_map,
        'sku_map': sku_map,
        'store_categories': store_categories,
        'sku_categories': sku_categories
    }
    
    # Also rename the OOF columns in train_df to match mapping names
    train_df.rename(columns={
        'store_sku_mean': 'store_sku_mean_sales',
        'store_mean': 'store_mean_sales',
        'sku_mean': 'sku_mean_sales'
    }, inplace=True)
    
    return train_df, mappings

def apply_target_encodings(df, mappings):
    """
    Applies pre-computed target encoding mappings to a dataset (validation or test).
    """
    df = df.copy()
    
    # Align categorical features using stored categories to prevent LightGBM mismatches
    if 'store_categories' in mappings and 'sku_categories' in mappings:
        df['store_id'] = pd.Categorical(df['store_id'], categories=mappings['store_categories'])
        df['sku_id'] = pd.Categorical(df['sku_id'], categories=mappings['sku_categories'])
    
    # Join the mappings
    df = df.merge(mappings['store_sku_map'], on=['store_id', 'sku_id'], how='left')
    df = df.merge(mappings['store_map'], on='store_id', how='left')
    df = df.merge(mappings['sku_map'], on='sku_id', how='left')
    
    # Fallback missing mapping values
    global_mean = mappings['global_mean']
    df['sku_mean_sales'] = df['sku_mean_sales'].fillna(global_mean)
    df['store_mean_sales'] = df['store_mean_sales'].fillna(global_mean)
    df['store_sku_mean_sales'] = df['store_sku_mean_sales'].fillna(df['sku_mean_sales']).fillna(global_mean)
    
    return df
