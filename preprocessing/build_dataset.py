import pandas as pd
import numpy as np
import os
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings

class BatteryDataPreprocessor:
    def __init__(self, lowess_frac: float = 0.05):
        self.lowess_frac = lowess_frac
        self.target_cols = [
            'Voltage_measured', 'Current_measured', 'Temperature_measured',
            'Current_load', 'Voltage_load'
        ]
        
    def load_and_preprocess(self, discharge_path: str) -> pd.DataFrame:
        """데이터 로드 및 전처리"""
        df = pd.read_csv(discharge_path)
        df = df.sort_values(['cycle_idx']).reset_index(drop=True)
        
        # 불필요한 컬럼 제거
        drop_cols = ['start_time_raw', 'Capacity', 'type', 'ambient_temperature', 'Time']
        df = df.drop([col for col in drop_cols if col in df.columns], axis=1)
        
        return df
    
    def apply_lowess(self, df: pd.DataFrame) -> pd.DataFrame:
        """LOWESS 기반 feature 생성 - 원본 데이터에 컬럼 추가"""
        result = df.copy()
        
        for col in self.target_cols:
            print(f"처리 중: {col}")
            
            smooth_data = []
            residual_data = []
            trend_data = []
            indices = []
            
            # 사이클별로 처리
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
                
                smooth_data.extend(smoothed)
                residual_data.extend(residual)
                trend_data.extend(trend)
                indices.extend(cycle_idx)
            
            # 새로운 컬럼으로 추가
            result[f'{col}_smooth'] = pd.Series(smooth_data, index=indices).reindex(result.index)
            result[f'{col}_residual'] = pd.Series(residual_data, index=indices).reindex(result.index)
            result[f'{col}_trend'] = pd.Series(trend_data, index=indices).reindex(result.index)
        
        return result


# 메인 실행
if __name__ == "__main__":
    BASE_DIR = r"dataset"
    
    preprocessor = BatteryDataPreprocessor(lowess_frac=0.05)
    
    print('--------데이터 로드 중--------')
    df = preprocessor.load_and_preprocess(f"{BASE_DIR}/B0005_discharge.csv")
    print(f"원본 데이터 shape: {df.shape}")
    
    print('--------LOWESS Feature 생성 중--------')
    df_with_lowess = preprocessor.apply_lowess(df)
    print(f"LOWESS 적용 후 shape: {df_with_lowess.shape}")
    print(f"추가된 feature 수: {df_with_lowess.shape[1] - df.shape[1]}")
    
    print('--------데이터 저장 중--------')
    output_path = f"{BASE_DIR}\B0005_discharge_with_lowess_features.csv"
    df_with_lowess.to_csv(output_path, index=False)
    
    print('--------완료--------')
    print(f"생성된 컬럼: {list(df_with_lowess.columns)}")