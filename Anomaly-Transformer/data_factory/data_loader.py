import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


class NASABatteryLoader(object):
    """
    NASA Battery Dataset Loader for Anomaly Transformer
    
    Args:
        data_path: path to NASA battery discharge data file
        win_size: window size (default: 25)
        step: stride for sliding window (default: 1)
        drop_columns: columns to drop from data
    """
    
    def __init__(self, data_path, win_size=25, step=1,
                 drop_columns=['type', 'start_time_raw', 'Time', 'ambient_temperature', 'cycle_idx']):
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        # Load discharge data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
            
            # Filter only discharge data
            if 'type' in data.columns:
                print(f"Original data shape: {data.shape}")
                data = data[data['type'] == 'discharge'].copy()
                print(f"After filtering discharge: {data.shape}")
            
            # Drop specified columns
            columns_to_drop = [col for col in drop_columns if col in data.columns]
            if columns_to_drop:
                print(f"Dropping columns: {columns_to_drop}")
                data = data.drop(columns=columns_to_drop)
            
            data = data.values
            
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
        else:
            raise ValueError("Data file must be .csv or .npy format")
        
        # Handle NaN values
        data = np.nan_to_num(data)
        
        print(f"Final data shape: {data.shape}")
        print(f"Features: {data.shape[1]}")
        
        # Fit scaler and transform
        self.scaler.fit(data)
        data_normalized = self.scaler.transform(data)
        
        # Use all data for training
        self.train = data_normalized
        self.all_data = data_normalized
        self.train_labels = np.zeros(len(self.train))
        
        print(f"\n--- Data Info ---")
        print(f"Total timesteps: {len(data_normalized)}")
        print(f"✅ Using ALL data for training")
        
        # Check if cycle 598 exists
        if len(data_normalized) > 598:
            print(f"✅ Timestep 598 exists in dataset")
        else:
            print(f"⚠️  Dataset has only {len(data_normalized)} timesteps (< 598)")
    
    def __len__(self):
        """Number of windows in the dataset"""
        return (self.train.shape[0] - self.win_size) // self.step + 1
    
    def __getitem__(self, index):
        """Get a single window of data"""
        start_idx = index * self.step
        window_data = self.train[start_idx:start_idx + self.win_size]
        window_label = self.train_labels[start_idx:start_idx + self.win_size]
        return np.float32(window_data), np.float32(window_label)
    
    def get_all_data(self):
        """Get all normalized data for full prediction"""
        return self.all_data


def get_nasa_loader(data_path, batch_size=32, win_size=25, step=1,
                    drop_columns=['type', 'start_time_raw', 'Time', 'ambient_temperature', 'cycle_idx']):
    """
    Get DataLoader for NASA Battery discharge data
    
    Args:
        data_path: path to NASA battery data file
        batch_size: batch size for DataLoader
        win_size: window size (default: 25)
        step: stride for sliding window (default: 1)
        drop_columns: columns to drop from data
    
    Returns:
        (DataLoader, Dataset) tuple
    """
    dataset = NASABatteryLoader(
        data_path=data_path,
        win_size=win_size,
        step=step,
        drop_columns=drop_columns
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,  # 항상 shuffle
        num_workers=0
    )
    
    return data_loader, dataset