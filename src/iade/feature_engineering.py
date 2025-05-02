from dataclasses import dataclass
from typing import Tuple
import pandas as pd
import numpy as np
from src.common.data_loader import DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

@dataclass
class IadeFeatureConfig:
    """Configuration class for return risk feature engineering"""
    discount_threshold: float = 0.2
    spending_threshold: float = 100
    target_column: str = "is_risky"
    feature_columns: list = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'discount',
                'quantity',
                'total_spending'
            ]

class IadeFeatureEngineering:
    """Feature engineering class for return risk prediction"""
    
    def __init__(self, config: IadeFeatureConfig = None):
        self.config = config or IadeFeatureConfig()
        self.data_loader = DataLoader()
        self.scaler = RobustScaler()

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare order details data"""
        order_details = self.data_loader.fetch_order_details()
        
        # Calculate total spending
        order_details['total_spending'] = order_details['quantity'] * order_details['unit_price']
        
        return order_details

    def _create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create labels based on discount and spending thresholds"""
        data[self.config.target_column] = (
            (data['discount'] > self.config.discount_threshold) & 
            (data['total_spending'] < self.config.spending_threshold)
        ).astype(int)
        return data

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using RobustScaler"""
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        # Save the scaler for later use
        joblib.dump(self.scaler, 'models/iade_scaler.joblib')
        return X_scaled

    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare data for modeling"""
        # Load and process data
        data = self._load_data()
        data = self._create_labels(data)
        
        # Select features and target
        X = data[self.config.feature_columns]
        y = data[self.config.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self._scale_features(X_train)
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Print data statistics
        print("\nData Statistics:")
        print(f"Total samples: {len(data)}")
        print(f"Risky orders: {y.sum()} ({y.mean()*100:.2f}%)")
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test 