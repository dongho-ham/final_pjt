from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


class NASABatteryLoader(object):
    """
    NASA Battery Dataset Loader for Anomaly Transformer
    """
    
    def __init__(self, data_path, win_size=25, step=1, mode='train', train_ratio=0.8):
        self.step = step
        self.win_size = win_size
        self.mode = mode
        
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
            
            if 'cycle_idx' not in df.columns:
                raise ValueError("cycle_idx column is required for temporal split")
            
            total_rows = len(df)
            unique_cycles = sorted(df['cycle_idx'].unique())
            total_cycles = len(unique_cycles)
            
            n_train_cycles = int(total_cycles * train_ratio)
            train_cycles = unique_cycles[:n_train_cycles]
            test_cycles = unique_cycles[n_train_cycles:]
            
            if mode == 'train':
                # Train set: first 80% cycles
                df = df[df['cycle_idx'].isin(train_cycles)]
                cycle_range = f"{train_cycles[0]} ~ {train_cycles[-1]}"
                n_cycles = len(train_cycles)
            elif mode in ['test', 'val', 'thre']:
                # Test/Val/Thre set: last 20% cycles
                df = df[df['cycle_idx'].isin(test_cycles)]
                cycle_range = f"{test_cycles[0]} ~ {test_cycles[-1]}"
                n_cycles = len(test_cycles)
            
            # Extract cycle_idx and drop from features
            self.cycle_idx = df['cycle_idx'].values
            df = df.drop(columns=['cycle_idx'])

            drop_cols = ['type', 'ambient_temperature', 'start_time_raw', 'Time', 'Capacity']

            # 존재하는 컬럼만 제거
            cols_to_drop = [col for col in drop_cols if col in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)

            data = df.values.astype(np.float32)
            
            # Print dataset info
            split_ratio = len(data) / total_rows * 100
            print(f"\n[{mode.upper()}]")
            print(f"  Cycles: {cycle_range} ({n_cycles} / {total_cycles} cycles)")
            print(f"  Rows: {len(data):,} / {total_rows:,} ({split_ratio:.1f}%)")
            print(f"  Windows: {(len(data) - win_size) // step + 1:,}")
            
        else:
            raise ValueError("Data file must be .csv format")
        
        if np.isnan(data).any():
            print(f"  Warning: {np.isnan(data).sum()} NaN values, replacing with 0")
            data = np.nan_to_num(data)
        
        self.data = data
    
    def __len__(self):
        return (len(self.data) - self.win_size) // self.step + 1
    
    def __getitem__(self, index):
        start_idx = index * self.step
        end_idx = start_idx + self.win_size
        
        window_data = self.data[start_idx:end_idx]
        window_label = np.zeros(self.win_size)
        
        return np.float32(window_data), np.float32(window_label)
    
    def get_all_data(self):
        """Get all data for full prediction"""
        return self.data


def get_nasa_loader(data_path, batch_size, win_size, step=1, mode='train', 
                       dataset='nasa_battery', train_ratio=0.8):
    """
    Get DataLoader for NASA Battery discharge data
    
    Args:
        data_path: path to NASA battery data file
        batch_size: batch size for DataLoader
        win_size: window size (default: 25)
        step: stride for sliding window (default: 1)
        mode: 'train', 'test', 'val', or 'thre' (default: 'train')
        dataset: dataset name (default: 'nasa_battery')
        train_ratio: ratio of data used for training (default: 0.8)
    
    Returns:
        (DataLoader, Dataset) tuple
    """
    dataset = NASABatteryLoader(
        data_path=data_path,
        win_size=win_size,
        step=step,
        mode=mode,
        train_ratio=train_ratio
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # 시계열 순서 유지
        num_workers=0
    )
    
    return data_loader, dataset

def get_loader_segment(data_path, batch_size, win_size, step=1, mode='train', 
                       dataset='nasa_battery', train_ratio=0.8):
    """NASA Battery용 wrapper - solver.py와 호환"""
    if dataset == 'nasa_battery':
        dataset_obj = NASABatteryLoader(
            data_path=data_path,
            win_size=win_size,
            step=step,
            mode=mode,
            train_ratio=train_ratio
        )
        
        # Train만 shuffle, 나머지는 순서 유지
        shuffle = (mode == 'train')
        
        data_loader = DataLoader(
            dataset=dataset_obj,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
        return data_loader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")