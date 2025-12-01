import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Tuple
import warnings

class BatteryDataPreprocessor:
    def __init__(self, window_size: int = 50, stride: int = 25, lowess_frac: float = 0.05):
        self.window_size = window_size
        self.stride = stride
        self.lowess_frac = lowess_frac
        self.target_cols = [
            'Voltage_measured', 'Current_measured', 'Temperature_measured',
            'Current', 'Voltage'
        ]
        self.scalers = {}
        
    def load_and_merge(self, charge_path: str, discharge_path: str) -> pd.DataFrame:
        """데이터 로드 및 병합"""
        charge = pd.read_csv(charge_path)
        discharge = pd.read_csv(discharge_path)
        
        merged = pd.concat([charge, discharge]).sort_values(['cycle_idx', 'Time'])
        merged = merged.reset_index(drop=True)  # ← 인덱스 리셋 추가!
        
        # 데이터 검증
        assert merged['cycle_idx'].is_monotonic_increasing
        
        return merged
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """기본 전처리"""
        df = df.copy()
        
        # Current/Voltage 통합
        df['Current'] = df['Current_charge'].fillna(df['Current_load'])
        df['Voltage'] = df['Voltage_charge'].fillna(df['Voltage_load'])
        
        assert df['Current'].isna().sum() == 0
        assert df['Voltage'].isna().sum() == 0
        
        # Phase 정보
        df['is_charging'] = (df['type'] == 'charge').astype(int)
        
        # Capacity forward fill
        df['Capacity'] = df['Capacity'].ffill()
        first_capacity = df['Capacity'].dropna().iloc[0]
        df['Capacity'] = df['Capacity'].fillna(first_capacity)
        
        # 불필요한 컬럼 제거
        drop_cols = ['Current_charge', 'Voltage_charge', 'type', 
                     'start_time_raw', 'Capacity', 'Current_load', 'Voltage_load']
        df = df.drop(drop_cols, axis=1)
        
        return df
    
    def apply_lowess(self, df: pd.DataFrame) -> pd.DataFrame:
        """LOWESS 기반 feature 생성 - 수정 버전"""
        result = df.copy()
        
        for col in self.target_cols:
            print(f"처리 중: {col}")
            
            # 빈 리스트로 시작
            smooth_data = []
            residual_data = []
            trend_data = []
            indices = []
            
            # 사이클별로 직접 처리
            for cycle in sorted(df['cycle_idx'].unique()):
                mask = df['cycle_idx'] == cycle
                cycle_idx = df[mask].index
                values = df.loc[mask, col].values
                
                n = len(values)
                if n == 0:
                    continue
                
                # LOWESS 적용
                time_idx = np.arange(n)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    smoothed = lowess(values, time_idx, 
                                    frac=self.lowess_frac, 
                                    return_sorted=False)
                
                residual = values - smoothed
                trend = np.gradient(smoothed)
                
                # 리스트에 추가
                smooth_data.extend(smoothed)
                residual_data.extend(residual)
                trend_data.extend(trend)
                indices.extend(cycle_idx)
            
            # Series로 변환 후 할당
            result[f'{col}_smooth'] = pd.Series(smooth_data, index=indices).reindex(result.index)
            result[f'{col}_residual'] = pd.Series(residual_data, index=indices).reindex(result.index)
            result[f'{col}_trend'] = pd.Series(trend_data, index=indices).reindex(result.index)
        
        return result
    
    def normalize(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """MinMaxScaler 적용"""
        df_scaled = df.copy()
        
        scale_cols = [
            'Voltage_measured', 'Current_measured', 'Temperature_measured',
            'Time', 'ambient_temperature', 'Current', 'Voltage'
        ]
        
        for col in self.target_cols:
            scale_cols.extend([f'{col}_smooth', f'{col}_residual', f'{col}_trend'])
        
        for col in scale_cols:
            if col not in df_scaled.columns:
                continue
                
            if is_train:
                scaler = MinMaxScaler(feature_range=(0, 1))
                df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
                self.scalers[col] = scaler
            else:
                df_scaled[col] = self.scalers[col].transform(df_scaled[[col]])
        
        return df_scaled
    
    def create_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """시계열 윈도우 생성"""
        base_features = ['Voltage_measured', 'Current_measured', 'Temperature_measured',
                        'Time', 'cycle_idx', 'ambient_temperature',
                        'Current', 'Voltage', 'is_charging']
        
        lowess_features = []
        for col in self.target_cols:
            lowess_features.extend([f'{col}_smooth', f'{col}_residual', f'{col}_trend'])
        
        features = base_features + lowess_features
        
        X, y_cycle = [], []
        
        for cycle in sorted(df['cycle_idx'].unique()):
            cycle_data = df[df['cycle_idx'] == cycle].reset_index(drop=True)
            
            if len(cycle_data) < self.window_size:
                continue
            
            for i in range(0, len(cycle_data) - self.window_size + 1, self.stride):
                window = cycle_data.iloc[i:i+self.window_size][features].values
                X.append(window)
                y_cycle.append(cycle)
        
        return np.array(X), np.array(y_cycle)
    

# 데이터셋이 있는 디렉토리 주소를 넣어주세요.
BASE_DIR = r"\dataset"
preprocessor = BatteryDataPreprocessor(window_size=50, stride=25)

merged = preprocessor.load_and_merge(f"{BASE_DIR}/B0005_charge.csv", f"{BASE_DIR}/B0005_discharge.csv")
print('--------데이터 전처리 중--------')
merged = preprocessor.preprocess(merged)
print('--------lowess 연산 중--------')
merged_lowess = preprocessor.apply_lowess(merged)
print('--------정규화 중--------')
merged_normalized = preprocessor.normalize(merged_lowess, is_train=True)
print('--------window 생성 중--------')
X, y_cycle = preprocessor.create_windows(merged_normalized)

print(f"X shape: {X.shape}")
print(f"Features per window: {X.shape[2]}")

print('--------데이터 저장 중--------')
# NumPy 형식으로 저장 (가장 빠르고 용량 작음)
np.save('X_train_B0005.npy', X)
np.save('y_cycle_B0005.npy', y_cycle)

# 전처리된 DataFrame도 저장 (나중에 다시 사용 가능)
merged_normalized.to_pickle('merged_normalized_B0005.pkl')

# Scaler 정보도 저장 (test 데이터 처리용)
with open('scalers_B0005.pkl', 'wb') as f:
    pickle.dump(preprocessor.scalers, f)