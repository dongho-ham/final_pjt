from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class NASABatteryLoader(object):
    """
    NASA Battery Dataset Loader for Anomaly Transformer
    """
    
    def __init__(self, data_path, win_size=25, step=1):
        self.step = step
        self.win_size = win_size
        
        # Load preprocessed data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            print(f"Loaded data shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Separate cycle_idx for tracking
            if 'cycle_idx' in df.columns:
                self.cycle_idx = df['cycle_idx'].values
                df = df.drop(columns=['cycle_idx'])
            else:
                self.cycle_idx = None
            
            # Convert to numpy
            data = df.values
            
        elif data_path.endswith('.npy'):
            data = np.load(data_path)
            self.cycle_idx = None
        else:
            raise ValueError("Data file must be .csv or .npy format")
        
        # Handle NaN values
        if np.isnan(data).any():
            print(f"Warning: {np.isnan(data).sum()} NaN values found, replacing with 0")
            data = np.nan_to_num(data)
        
        print(f"Final data shape: {data.shape}")
        print(f"Features: {data.shape[1]}")
        
        # Use all data for training (unsupervised)
        self.train = data
        self.all_data = data
        
        # Label 생성: cycle 42와 596을 anomaly로 설정
        if self.cycle_idx is not None:
            self.labels = np.zeros(len(self.cycle_idx))
            self.labels[(self.cycle_idx == 42) | (self.cycle_idx == 596)] = 1
            
            print(f"\n--- Anomaly Label Info ---")
            print(f"Anomaly samples: {int(self.labels.sum())}")
            print(f"Anomaly ratio: {self.labels.mean()*100:.2f}%")
            print(f"Normal samples: {int((self.labels == 0).sum())}")
        else:
            self.labels = np.zeros(len(self.train))
        
        print(f"\n--- Data Info ---")
        print(f"Total timesteps: {len(data)}")
        print(f"Number of windows: {(len(data) - win_size) // self.step + 1}")
    
    def __len__(self):
        """Number of windows in the dataset"""
        return (self.train.shape[0] - self.win_size) // self.step + 1
    
    def __getitem__(self, index):
        """Get a single window of data"""
        start_idx = index * self.step
        window_data = self.train[start_idx:start_idx + self.win_size]
        window_label = self.labels[start_idx:start_idx + self.win_size]
        return np.float32(window_data), np.float32(window_label)
    
    def get_all_data(self):
        """Get all data for full prediction"""
        return self.all_data


def get_nasa_loader(data_path, batch_size=32, win_size=25, step=1):
    """
    Get DataLoader for NASA Battery discharge data
    
    Args:
        data_path: path to NASA battery data file
        batch_size: batch size for DataLoader
        win_size: window size (default: 25)
        step: stride for sliding window (default: 1)
    
    Returns:
        (DataLoader, Dataset) tuple
    """
    dataset = NASABatteryLoader(
        data_path=data_path,
        win_size=win_size,
        step=step
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return data_loader, dataset

def get_loader_segment(data_path, batch_size, win_size, mode='train', dataset='nasa_battery'):
    """NASA Battery용 wrapper - solver.py와 호환"""
    if dataset == 'nasa_battery':
        # mode에 관계없이 전체 데이터 반환 (unsupervised)
        dataset_obj = NASABatteryLoader(
            data_path=data_path,
            win_size=win_size,
            step=1
        )
        
        # test/thre mode에서는 shuffle=False (순서 유지)
        shuffle = True if mode == 'train' else False
        
        data_loader = DataLoader(
            dataset=dataset_obj,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        return data_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")