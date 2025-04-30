# feature_engineering.py dosyanızda

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from data_loader import DataLoader
import os

@dataclass
class FeatureConfig:
    """Configuration class for feature engineering parameters"""
    churn_threshold_months: int = 6
    reference_date: str = "1998-05-06"
    target_column: str = "Churn"
    feature_columns: List[str] = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'total_order_value',
                'order_count',
                'average_order_value',
                'recency_days',
                'frequency_score',
                'monetary_score'
            ]

class FeatureProcessor(ABC):
    """Abstract base class for feature processing"""
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class TemporalFeatureProcessor(FeatureProcessor):
    """Process temporal features from order data"""
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # Ensure order_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['order_date']):
            data['order_date'] = pd.to_datetime(data['order_date'])
            
        # Extract temporal features
        data['order_month'] = data['order_date'].dt.month
        data['order_quarter'] = data['order_date'].dt.quarter
        data['order_year'] = data['order_date'].dt.year
        
        return data

class RFMFeatureProcessor(FeatureProcessor):
    """Process RFM (Recency, Frequency, Monetary) features"""
    def __init__(self, reference_date: str):
        self.reference_date = pd.to_datetime(reference_date)

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate last order date for each customer
        data['order_date'] = pd.to_datetime(data['order_date'])
        data['last_order_date'] = data.groupby('customer_id')['order_date'].transform('max')
        
        # Recency
        data['recency_days'] = (self.reference_date - data['last_order_date']).dt.days
        
        # Frequency
        data['frequency_score'] = data['order_count']
        
        # Monetary
        data['monetary_score'] = data['total_order_value']
        
        return data

class FeatureEngineering:
    """Feature engineering for customer churn prediction"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV files"""
        # Load customer data
        customers_df = pd.read_csv('data/customers.csv')
        
        # Load order data
        orders_df = pd.read_csv('data/orders.csv')
        
        # Load order details
        order_details_df = pd.read_csv('data/order_details.csv')
        
        return customers_df, orders_df, order_details_df
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for modeling"""
        # Load data
        customers_df, orders_df, order_details_df = self.load_data()
        
        # Calculate features
        features_df = self._calculate_features(customers_df, orders_df, order_details_df)
        
        # Split features and target
        X = features_df[self.config.feature_columns].values
        y = features_df[self.config.target_column].values
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def _calculate_features(self, customers_df: pd.DataFrame, 
                          orders_df: pd.DataFrame, 
                          order_details_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features for modeling"""
        # Merge order details with orders
        order_details = pd.merge(
            order_details_df,
            orders_df[['order_id', 'customer_id', 'order_date']],
            on='order_id'
        )
        
        # Calculate total order value and count per customer
        customer_orders = order_details.groupby('customer_id').agg({
            'unit_price': 'sum',
            'order_id': 'count'
        }).rename(columns={
            'unit_price': 'total_order_value',
            'order_id': 'order_count'
        })
        
        # Calculate average order value
        customer_orders['average_order_value'] = (
            customer_orders['total_order_value'] / customer_orders['order_count']
        )
        
        # Calculate recency (days since last order)
        latest_order = order_details.groupby('customer_id')['order_date'].max()
        latest_order = pd.to_datetime(latest_order)
        current_date = pd.to_datetime('2024-01-01')  # Example current date
        customer_orders['recency_days'] = (current_date - latest_order).dt.days
        
        # Calculate frequency score (1-5)
        customer_orders['frequency_score'] = pd.qcut(
            customer_orders['order_count'],
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Calculate monetary score (1-5)
        customer_orders['monetary_score'] = pd.qcut(
            customer_orders['total_order_value'],
            q=5,
            labels=[1, 2, 3, 4, 5]
        ).astype(int)
        
        # Merge with customers
        features_df = pd.merge(
            customers_df,
            customer_orders,
            on='customer_id',
            how='left'
        )
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        return features_df

if __name__ == "__main__":
    # Initialize feature engineering
    config = FeatureConfig()
    feature_engineering = FeatureEngineering(config)
    
    # Prepare data
    X, y = feature_engineering.prepare_data()
    
    # Print sample results
    print("\nFeature Statistics:")
    print(X.describe())
    print("\nClass Distribution:")
    print(y.value_counts(normalize=True))
