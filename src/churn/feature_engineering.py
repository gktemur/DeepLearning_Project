# feature_engineering.py dosyanızda

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from src.common.data_loader import DataLoader
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
import joblib

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
                'monetary_score',
                'is_recent',
                'is_one_time_customer',
                'is_high_value',
                'is_frequent', 
                'is_low_spender', 
                'has_large_order'
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
    """Main class for feature engineering process"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.data_loader = DataLoader()
        self.scaler = StandardScaler()

    def _load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load data from database"""
        return (
            self.data_loader.fetch_customers(),
            self.data_loader.fetch_orders(),
            self.data_loader.fetch_order_details()
        )

    def _calculate_order_metrics(self, orders_df: pd.DataFrame, order_details_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic order metrics"""
        # Calculate total order value
        order_details_df['total_order_value'] = order_details_df['quantity'] * order_details_df['unit_price']
        total_spending = order_details_df.groupby('order_id')['total_order_value'].sum().reset_index()
        
        # Calculate order count
        order_count = orders_df.groupby('customer_id').size().reset_index(name='order_count')
        
        # Merge metrics
        order_summary = orders_df.merge(order_count, on='customer_id', how='left')
        order_summary = order_summary.merge(total_spending, on='order_id', how='left')
        
        # Calculate average order value
        order_summary['average_order_value'] = order_summary['total_order_value'] / order_summary['order_count']
        
        return order_summary

    def _process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process features using various processors"""
        processors = [
            TemporalFeatureProcessor(),
            RFMFeatureProcessor(self.config.reference_date)
        ]
        
        for processor in processors:
            data = processor.process(data)
        
        return data

    def _clean_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean data by handling missing values"""
        # Drop rows with NaN values
        X_clean = X.dropna()
        y_clean = y[X_clean.index]
        
        # Reset index after dropping rows
        X_clean = X_clean.reset_index(drop=True)
        y_clean = y_clean.reset_index(drop=True)
        
        return X_clean, y_clean

    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using RobustScaler for better handling of outliers"""
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Save the scaler for later use
        joblib.dump(self.scaler, 'models/scaler.joblib')
        return X_scaled

    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using SMOTE"""
        # First clean the data
        X_clean, y_clean = self._clean_data(X, y)
        
        # Then apply SMOTE with adjusted sampling strategy
        smote = SMOTE(
            random_state=42,
            sampling_strategy=0.8,  # Increase minority class to 80% of majority
            k_neighbors=3  # Reduced neighbors for more distinct samples
        )
        X_resampled, y_resampled = smote.fit_resample(X_clean, y_clean)
        return X_resampled, y_resampled

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Main method to prepare data for modeling"""
        # Load data
        customers_df, orders_df, order_details_df = self._load_data()
        
        # Calculate basic metrics
        order_summary = self._calculate_order_metrics(orders_df, order_details_df)
        
        # Process features
        processed_data = self._process_features(order_summary)
        
        # Calculate churn
        processed_data['months_since_last_order'] = (
            pd.to_datetime(self.config.reference_date) - processed_data['last_order_date']
        ).dt.days // 30
        
        processed_data[self.config.target_column] = processed_data['months_since_last_order'].apply(
            lambda x: 1 if x >= self.config.churn_threshold_months else 0
        )

        # Add binary features with adjusted thresholds
        processed_data['is_recent'] = (processed_data['recency_days'] < 30).astype(int)  # Reduced from 60
        processed_data['is_one_time_customer'] = (processed_data['order_count'] == 1).astype(int)
        processed_data['is_high_value'] = (processed_data['monetary_score'] > processed_data['monetary_score'].quantile(0.8)).astype(int)
        processed_data['is_frequent'] = (processed_data['frequency_score'] > 7).astype(int)
        processed_data['is_low_spender'] = (processed_data['monetary_score'] < processed_data['monetary_score'].quantile(0.2)).astype(int)
        processed_data['has_large_order'] = (processed_data['average_order_value'] > 1500).astype(int)

        # Add recency score as a new feature
        processed_data['recency_score'] = np.where(
            processed_data['recency_days'] < 30, 1,
            np.where(processed_data['recency_days'] < 90, 0.5,
            np.where(processed_data['recency_days'] < 180, 0.2, 0))
        )

        # Update feature columns to include recency_score
        self.config.feature_columns.append('recency_score')

        # Final feature set
        X = processed_data[self.config.feature_columns]
        y = processed_data[self.config.target_column]

        # Print data quality information
        print("\nData Quality Information:")
        print("Number of samples before cleaning:", len(X))
        print("Number of missing values per feature:")
        print(X.isnull().sum())
        print("\nClass distribution before balancing:")
        print(y.value_counts(normalize=True))

        # Handle class imbalance using SMOTE
        X_resampled, y_resampled = self._handle_class_imbalance(X, y)
        
        print("\nData Quality Information after cleaning:")
        print("Number of samples after cleaning:", len(X_resampled))
        print("\nClass distribution after balancing:")
        print(y_resampled.value_counts(normalize=True))
        
        # Scale features
        X_scaled = self._scale_features(X_resampled)
        
        return X_scaled, y_resampled


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
